"""Self-distillation: the target's own greedy responses to training prompts.

The drafter learns to predict what *this* model actually says, not what a
dataset's reference answer says. That distinction is the whole mechanism --
acceptance rate is agreement with the target's own continuation, so training on
human-written answers would optimise the wrong thing.

Generation is batched, which the benchmark loop deliberately is not. Batch-1 is
the regime the *measurements* are about; data generation just needs throughput,
so it uses HF `generate` with left padding. Decoder-only models must be
left-padded or the prompt ends up separated from the generation by pad tokens
and every response is conditioned on garbage.

Work is checkpointed every `checkpoint_every` prompts and resumed by reading back
what is already on disk, so a preempted session loses minutes rather than hours.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class DistillRecord:
    """One prompt and the target's greedy continuation, as token ids."""

    domain: str
    prompt: str
    prompt_ids: list[int]
    response_ids: list[int]

    @property
    def n_response_tokens(self) -> int:
        return len(self.response_ids)

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "prompt": self.prompt,
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
        }


def load_existing(path: Path) -> list[DistillRecord]:
    """Read records already generated, for resume."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            DistillRecord(
                domain=row["domain"],
                prompt=row["prompt"],
                prompt_ids=row["prompt_ids"],
                response_ids=row["response_ids"],
            )
        )
    return records


def _batches(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.no_grad()
def generate_responses(
    target,
    prompts: list[tuple[str, str]],
    out_path: Path,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    checkpoint_every: int = 64,
    verbose: bool = True,
) -> list[DistillRecord]:
    """Greedily generate the target's response to each ``(domain, prompt)``.

    Resumes from `out_path`: prompts already present are skipped, so re-running
    after a preemption continues rather than restarting.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(out_path)
    done = {record.prompt for record in existing}
    todo = [(domain, prompt) for domain, prompt in prompts if prompt not in done]

    if verbose:
        print(f"{len(existing)} already done, {len(todo)} to generate", flush=True)
    if not todo:
        return existing

    tokenizer = target.tokenizer
    # Left padding: with right padding the prompt and its continuation end up
    # separated by pads and every response is conditioned on the wrong context.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = list(existing)
    handle = out_path.open("a")
    started = time.perf_counter()
    generated_tokens = 0

    try:
        for batch in _batches(todo, batch_size):
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for _, prompt in batch
            ]
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(target.device)

            outputs = target.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

            prompt_len = encoded["input_ids"].shape[1]
            for (domain, prompt), row, mask in zip(
                batch, outputs, encoded["attention_mask"]
            ):
                # Strip left padding from the prompt side before recording, so
                # prompt_ids is the real prompt rather than pad-prefixed.
                keep = int(mask.sum())
                prompt_ids = row[prompt_len - keep : prompt_len].tolist()
                response_ids = row[prompt_len:].tolist()
                while response_ids and response_ids[-1] == tokenizer.pad_token_id:
                    response_ids.pop()

                record = DistillRecord(domain, prompt, prompt_ids, response_ids)
                records.append(record)
                generated_tokens += len(response_ids)
                handle.write(json.dumps(record.as_dict()) + "\n")

            if len(records) % checkpoint_every < batch_size:
                handle.flush()

            if verbose:
                elapsed = time.perf_counter() - started
                rate = generated_tokens / elapsed if elapsed else 0.0
                print(
                    f"  {len(records)}/{len(prompts)} prompts | "
                    f"{generated_tokens} tokens | {rate:.0f} tok/s",
                    flush=True,
                )
    finally:
        handle.close()
        tokenizer.padding_side = original_side

    return records


def token_counts(records: list[DistillRecord]) -> dict[str, int]:
    """Response tokens per domain -- the quantity the mixtures are balanced on."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.domain] = counts.get(record.domain, 0) + record.n_response_tokens
    return counts


def build_mixture(
    records: list[DistillRecord], mixture: dict[str, float], budget_tokens: int
) -> list[DistillRecord]:
    """Select records hitting `budget_tokens` total, split by `mixture`.

    Balanced on response tokens rather than prompt count: code answers are much
    longer than chat answers, so equal prompt counts would give a code-containing
    mixture far more training signal and confound the domain comparison.
    """
    by_domain: dict[str, list[DistillRecord]] = {}
    for record in records:
        by_domain.setdefault(record.domain, []).append(record)

    selected: list[DistillRecord] = []
    for domain, share in mixture.items():
        target_tokens = int(budget_tokens * share)
        running = 0
        for record in by_domain.get(domain, []):
            if running >= target_tokens:
                break
            selected.append(record)
            running += record.n_response_tokens
    return selected
