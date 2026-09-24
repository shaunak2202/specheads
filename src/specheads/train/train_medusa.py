"""Phase 3: train the Medusa heads against the target's own greedy responses.

Only the heads train. The target is frozen and runs under `no_grad`, and the LM
head is shared and frozen -- which on Qwen2.5 is mandatory rather than merely
thrifty, since tied embeddings mean a gradient there would rewrite the target's
input embeddings.

Two details carry the memory budget:

* **Loss is masked to response tokens.** The drafter is only ever used while
  generating a response, so training it to predict prompt continuations spends
  capacity on a distribution it will never face.
* **Cross-entropy is chunked over the sequence.** At vocab 151936 a single fp32
  1024-token logit tensor is 622 MB, and Medusa needs one per head. Chunking caps
  peak logit memory at the chunk ratio for a small throughput cost.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ..model.medusa_heads import MedusaHeads


@dataclass
class TrainConfig:
    num_heads: int = 5
    num_resblocks: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 3
    warmup_steps: int = 50
    grad_clip: float = 1.0
    head_loss_decay: float = 0.8
    ce_chunk_size: int = 256
    max_seq_len: int = 1024
    seed: int = 1234
    log_every: int = 25


@dataclass
class Example:
    """One distilled sequence, with the boundary the loss is masked on."""

    input_ids: torch.Tensor  # [seq]
    prompt_len: int
    domain: str

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])


def build_examples(
    records: Iterable[dict], max_seq_len: int, device: torch.device
) -> list[Example]:
    """Concatenate prompt and response ids into training sequences."""
    examples: list[Example] = []
    for record in records:
        prompt_ids = record["prompt_ids"]
        response_ids = record["response_ids"]
        if len(response_ids) < 8:
            # Too short to supply targets for the deeper heads.
            continue
        ids = (prompt_ids + response_ids)[:max_seq_len]
        if len(ids) < 16:
            continue
        examples.append(
            Example(
                input_ids=torch.tensor(ids, dtype=torch.long, device=device),
                prompt_len=min(len(prompt_ids), len(ids)),
                domain=record["domain"],
            )
        )
    return examples


def chunked_cross_entropy(
    features: torch.Tensor,
    lm_head: nn.Module,
    targets: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy through the frozen LM head, in sequence chunks.

    Returns the summed loss and the token count, so the caller can weight heads
    by their own token counts rather than by chunk arithmetic.
    """
    if features.shape[0] == 0:
        return features.new_zeros(()), 0

    # The heads carry fp32 master weights while the frozen LM head is fp16, so
    # the cast happens here, at the boundary. Logits come straight back to fp32
    # before the softmax: an fp16 softmax over 151936 classes loses enough of
    # the tail to distort the loss.
    weight_dtype = next(lm_head.parameters()).dtype

    total = features.new_zeros(())
    count = 0
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        logits = lm_head(features[start:stop].to(weight_dtype)).float()
        chunk_targets = targets[start:stop]
        total = total + F.cross_entropy(logits, chunk_targets, reduction="sum")
        count += int(chunk_targets.numel())
    return total, count


@dataclass
class HeadStats:
    """Per-head top-1 / top-5 accuracy accumulator."""

    top1: list[int] = field(default_factory=list)
    top5: list[int] = field(default_factory=list)
    seen: list[int] = field(default_factory=list)

    def ensure(self, num_heads: int) -> None:
        if not self.seen:
            self.top1 = [0] * num_heads
            self.top5 = [0] * num_heads
            self.seen = [0] * num_heads

    def update(self, head: int, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if targets.numel() == 0:
            return
        top = torch.topk(logits, k=5, dim=-1).indices
        self.top1[head] += int((top[:, 0] == targets).sum())
        self.top5[head] += int((top == targets.unsqueeze(-1)).any(dim=-1).sum())
        self.seen[head] += int(targets.numel())

    def as_dict(self) -> dict:
        return {
            "top1_accuracy": [
                (t / s if s else float("nan")) for t, s in zip(self.top1, self.seen)
            ],
            "top5_accuracy": [
                (t / s if s else float("nan")) for t, s in zip(self.top5, self.seen)
            ],
            "tokens_scored": list(self.seen),
        }


@torch.no_grad()
def evaluate(
    target_model,
    heads: MedusaHeads,
    lm_head: nn.Module,
    examples: list[Example],
    config: TrainConfig,
) -> dict:
    """Per-head top-1/top-5 accuracy on held-out examples."""
    heads.eval()
    stats = HeadStats()
    stats.ensure(config.num_heads)
    weight_dtype = next(lm_head.parameters()).dtype

    for example in examples:
        ids = example.input_ids.unsqueeze(0)
        hidden = target_model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        features = heads(hidden.float())

        # Indexing must match the training loop exactly: head k predicts t+k+2,
        # and only response positions are scored.
        for head in range(config.num_heads):
            shift = head + 2
            start = max(example.prompt_len - 1, 0)
            stop = example.length - shift
            if stop <= start:
                continue
            head_features = features[head][start:stop]
            head_target_ids = example.input_ids[start + shift : stop + shift]
            logits = lm_head(head_features.to(weight_dtype)).float()
            stats.update(head, logits, head_target_ids)

    heads.train()
    return stats.as_dict()


def train_medusa(
    target,
    train_examples: list[Example],
    val_examples: list[Example],
    config: TrainConfig,
    out_dir: Path,
    resume: bool = True,
    verbose: bool = True,
) -> dict:
    """Train the heads, checkpointing each epoch. Returns the training record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    device = target.device
    lm_head = target.lm_head
    for param in lm_head.parameters():
        param.requires_grad_(False)

    heads = MedusaHeads(
        target.hidden_size, num_heads=config.num_heads, num_resblocks=config.num_resblocks
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        heads.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    start_epoch = 0
    checkpoint_path = out_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        heads.load_state_dict(state["heads"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        if verbose:
            print(f"resumed from epoch {start_epoch}", flush=True)

    weights = [config.head_loss_decay**k for k in range(config.num_heads)]
    total_steps = max(1, config.epochs * len(train_examples))
    history: list[dict] = []
    step = start_epoch * len(train_examples)
    started = time.perf_counter()

    for epoch in range(start_epoch, config.epochs):
        running = 0.0
        counted = 0
        for index, example in enumerate(train_examples):
            # Learning-rate schedule: linear warmup then cosine decay.
            if step < config.warmup_steps:
                lr_scale = (step + 1) / config.warmup_steps
            else:
                progress = (step - config.warmup_steps) / max(
                    1, total_steps - config.warmup_steps
                )
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = config.lr * lr_scale

            ids = example.input_ids.unsqueeze(0)
            with torch.no_grad():
                hidden = target.model.model(
                    input_ids=ids, use_cache=False
                ).last_hidden_state[0].float()

            features = heads(hidden)

            loss = features.new_zeros(())
            tokens = 0
            for head in range(config.num_heads):
                shift = head + 2
                start = max(example.prompt_len - 1, 0)
                stop = example.length - shift
                if stop <= start:
                    continue
                head_features = features[head][start:stop]
                head_targets_ids = example.input_ids[start + shift : stop + shift]
                head_loss, count = chunked_cross_entropy(
                    head_features, lm_head, head_targets_ids, config.ce_chunk_size
                )
                if count:
                    loss = loss + weights[head] * (head_loss / count)
                    tokens += count

            if tokens == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), config.grad_clip)
            optimizer.step()

            running += float(loss.detach())
            counted += 1
            step += 1

            if verbose and config.log_every and (index + 1) % config.log_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch} step {index + 1}/{len(train_examples)} "
                    f"loss {running / max(1, counted):.4f} "
                    f"lr {config.lr * lr_scale:.2e} ({elapsed:.0f}s)",
                    flush=True,
                )

        metrics = evaluate(target.model, heads, lm_head, val_examples, config)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, counted),
            "val": metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if verbose:
            top1 = ", ".join(f"h{i}={a:.3f}" for i, a in enumerate(metrics["top1_accuracy"]))
            print(f"epoch {epoch}: loss {record['train_loss']:.4f} | val top1 {top1}", flush=True)

        torch.save(
            {
                "heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": config.__dict__,
            },
            checkpoint_path,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({"heads": heads.state_dict(), "config": config.__dict__}, out_dir / "heads.pt")
    return {"history": history, "num_parameters": heads.num_parameters}
