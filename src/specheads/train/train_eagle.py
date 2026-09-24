"""Phase 5: train the EAGLE drafter, and sweep its loss weighting.

The loss has two terms with genuinely different jobs:

* **feature regression** (smooth L1) against the target's real next hidden state,
  which is what keeps the drafter's *autoregressive* rollout from drifting -- at
  depth d it consumes its own depth-(d-1) prediction, so an error in feature
  space compounds in a way a token-level loss alone never sees;
* **cross-entropy** through the frozen LM head, which is what actually decides
  the drafted token and therefore the acceptance rate.

Their scales are unrelated -- one is a distance between 1536-dim vectors, the
other a log-probability over 151936 classes -- so the weighting cannot be
reasoned about in advance. `sweep_loss_weight` trains one short run per candidate
on an identical slice with an identical seed, and the winner is chosen on
validation, never on the eval prompts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..model.eagle_drafter import EagleDrafter
from .train_medusa import Example


@dataclass
class EagleConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    epochs: int = 2
    warmup_steps: int = 50
    grad_clip: float = 1.0
    w_regression: float = 1.0
    w_cross_entropy: float = 0.1
    ce_chunk_size: int = 256
    intermediate_size: int | None = None
    seed: int = 1234
    log_every: int = 25
    max_steps: int | None = None


def eagle_losses(
    drafter: EagleDrafter,
    target,
    example: Example,
    config: EagleConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Regression and cross-entropy terms for one teacher-forced sequence.

    Alignment: the drafter sees feature ``h_t`` and the embedding of ``tok_{t+1}``
    and predicts ``h_{t+1}``; pushing that through the LM head predicts
    ``tok_{t+2}``. Being off by one here trains the drafter on the wrong horizon
    while the loss still falls, so the slices are written out explicitly.
    """
    ids = example.input_ids.unsqueeze(0)
    with torch.no_grad():
        hidden = target.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]

    length = example.length
    start = max(example.prompt_len - 1, 0)
    stop = length - 2
    if stop <= start:
        zero = hidden.new_zeros(())
        return zero, zero, 0

    drafter_dtype = next(drafter.parameters()).dtype
    features = hidden[start:stop].to(drafter_dtype)                    # h_t
    next_tokens = example.input_ids[start + 1 : stop + 1]              # tok_{t+1}
    true_next = hidden[start + 1 : stop + 1].to(drafter_dtype)         # h_{t+1}
    ce_targets = example.input_ids[start + 2 : stop + 2]               # tok_{t+2}

    embeddings = target.model.model.embed_tokens(next_tokens).to(drafter_dtype)
    positions = torch.arange(features.shape[0], device=features.device).unsqueeze(0)
    rope = target.model.model.rotary_emb(features.unsqueeze(0), positions)

    predicted = drafter(features.unsqueeze(0), embeddings.unsqueeze(0), rope)[0]

    regression = F.smooth_l1_loss(predicted, true_next)

    head_dtype = next(target.lm_head.parameters()).dtype
    total = predicted.new_zeros(())
    count = 0
    for chunk in range(0, predicted.shape[0], config.ce_chunk_size):
        end = min(chunk + config.ce_chunk_size, predicted.shape[0])
        logits = target.lm_head(predicted[chunk:end].to(head_dtype)).float()
        total = total + F.cross_entropy(logits, ce_targets[chunk:end], reduction="sum")
        count += end - chunk
    cross_entropy = total / max(1, count)
    return regression, cross_entropy, count


@torch.no_grad()
def evaluate_eagle(drafter, target, examples: list[Example], config: EagleConfig) -> dict:
    """Validation losses and top-1 token accuracy of the drafted feature."""
    drafter.eval()
    reg_total = ce_total = 0.0
    correct = seen = 0

    for example in examples:
        regression, cross_entropy, count = eagle_losses(drafter, target, example, config)
        if count == 0:
            continue
        reg_total += float(regression)
        ce_total += float(cross_entropy)

        ids = example.input_ids.unsqueeze(0)
        hidden = target.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        start = max(example.prompt_len - 1, 0)
        stop = example.length - 2
        drafter_dtype = next(drafter.parameters()).dtype
        features = hidden[start:stop].to(drafter_dtype)
        next_tokens = example.input_ids[start + 1 : stop + 1]
        embeddings = target.model.model.embed_tokens(next_tokens).to(drafter_dtype)
        positions = torch.arange(features.shape[0], device=features.device).unsqueeze(0)
        rope = target.model.model.rotary_emb(features.unsqueeze(0), positions)
        predicted = drafter(features.unsqueeze(0), embeddings.unsqueeze(0), rope)[0]
        head_dtype = next(target.lm_head.parameters()).dtype
        logits = target.lm_head(predicted.to(head_dtype)).float()
        targets = example.input_ids[start + 2 : stop + 2]
        correct += int((logits.argmax(dim=-1) == targets).sum())
        seen += int(targets.numel())

    drafter.train()
    n = max(1, len(examples))
    return {
        "val_regression": reg_total / n,
        "val_cross_entropy": ce_total / n,
        "val_top1_accuracy": correct / seen if seen else float("nan"),
        "tokens_scored": seen,
    }


def train_eagle(
    target,
    train_examples: list[Example],
    val_examples: list[Example],
    config: EagleConfig,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """Train the drafter, checkpointing each epoch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    drafter = EagleDrafter(target.config, intermediate_size=config.intermediate_size)
    drafter = drafter.to(device=target.device, dtype=torch.float32)

    for param in target.lm_head.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        drafter.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history: list[dict] = []
    step = 0
    started = time.perf_counter()
    stop_early = False

    for epoch in range(config.epochs):
        running = 0.0
        counted = 0
        for example in train_examples:
            if config.max_steps is not None and step >= config.max_steps:
                stop_early = True
                break

            lr_scale = min(1.0, (step + 1) / max(1, config.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = config.lr * lr_scale

            regression, cross_entropy, count = eagle_losses(drafter, target, example, config)
            if count == 0:
                continue
            loss = config.w_regression * regression + config.w_cross_entropy * cross_entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drafter.parameters(), config.grad_clip)
            optimizer.step()

            running += float(loss.detach())
            counted += 1
            step += 1

            if verbose and config.log_every and step % config.log_every == 0:
                print(
                    f"  step {step} loss {running / max(1, counted):.4f} "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )

        metrics = evaluate_eagle(drafter, target, val_examples, config)
        history.append({"epoch": epoch, "train_loss": running / max(1, counted), **metrics})
        if verbose:
            print(
                f"epoch {epoch}: loss {history[-1]['train_loss']:.4f} "
                f"val_top1 {metrics['val_top1_accuracy']:.4f}",
                flush=True,
            )
        torch.save({"drafter": drafter.state_dict(), "config": config.__dict__},
                   out_dir / "drafter.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if stop_early:
            break

    return {"history": history, "num_parameters": drafter.num_parameters}
