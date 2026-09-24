#!/usr/bin/env python3
"""Phase 3: train Medusa heads on a chosen domain mixture.

Two runs make the domain-shift experiment: `--mixture chat` (chat only) and
`--mixture mixed` (equal token counts across chat, code and math). Both are
capped to the same token budget so the comparison is about mixture, not volume.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

from specheads.model.target import load_target
from specheads.train.distill_data import DistillRecord, build_mixture, load_existing
from specheads.train.train_medusa import TrainConfig, build_examples, train_medusa
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything

MIXTURES = {
    "chat": {"chat": 1.0},
    "mixed": {"chat": 0.3334, "code": 0.3333, "math": 0.3333},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", choices=sorted(MIXTURES), required=True)
    parser.add_argument("--budget-tokens", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--responses", type=Path, default=Path("data/distill/responses.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or Path(f"results/medusa_{args.mixture}")
    seed_everything(args.seed)

    records = load_existing(args.responses)
    print(f"{len(records)} distilled records available", flush=True)

    selected = build_mixture(records, MIXTURES[args.mixture], args.budget_tokens)
    tokens = sum(r.n_response_tokens for r in selected)
    by_domain: dict[str, int] = {}
    for record in selected:
        by_domain[record.domain] = by_domain.get(record.domain, 0) + r_tokens(record)
    print(f"mixture={args.mixture}: {len(selected)} records, {tokens} response tokens")
    print(f"  by domain: {by_domain}")

    # Validation split is held out for hyperparameter choices only; the eval
    # prompt sets are never touched here.
    rng = random.Random(args.seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    n_val = max(4, int(len(shuffled) * args.val_fraction))
    val_records, train_records = shuffled[:n_val], shuffled[n_val:]

    target = load_target(dtype=args.dtype, device=args.device)
    train_examples = build_examples(
        (r.as_dict() for r in train_records), args.max_seq_len, target.device
    )
    val_examples = build_examples(
        (r.as_dict() for r in val_records), args.max_seq_len, target.device
    )
    print(f"train={len(train_examples)} val={len(val_examples)}", flush=True)

    config = TrainConfig(
        num_heads=args.num_heads,
        lr=args.lr,
        epochs=args.epochs,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    result = train_medusa(target, train_examples, val_examples, config, out_dir)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "mixture": args.mixture,
        "mixture_weights": MIXTURES[args.mixture],
        "budget_tokens": args.budget_tokens,
        "selected_records": len(selected),
        "selected_tokens": tokens,
        "tokens_by_domain": by_domain,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "config": config.__dict__,
        "num_parameters": result["num_parameters"],
        "history": result["history"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


def r_tokens(record: DistillRecord) -> int:
    return record.n_response_tokens


if __name__ == "__main__":
    raise SystemExit(main())
