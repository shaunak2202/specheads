#!/usr/bin/env python3
"""Phase 5: train the EAGLE drafter with the swept loss weight."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from specheads.model.target import load_target
from specheads.train.distill_data import build_mixture, load_existing
from specheads.train.train_eagle import EagleConfig, train_eagle
from specheads.train.train_medusa import build_examples
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything

MIXTURES = {"chat": {"chat": 1.0}, "mixed": {"chat": 0.3334, "code": 0.3333, "math": 0.3333}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", default="mixed", choices=sorted(MIXTURES))
    parser.add_argument("--w-cross-entropy", type=float, default=None,
                        help="default: read the winner from results/eagle_loss_sweep/sweep.json")
    parser.add_argument("--budget-tokens", type=int, default=60000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    weight = args.w_cross_entropy
    if weight is None:
        sweep = json.loads(Path("results/eagle_loss_sweep/sweep.json").read_text())
        weight = sweep["best"]["w_cross_entropy"]
        print(f"using swept w_cross_entropy = {weight}")

    out_dir = args.out or Path(f"results/eagle_{args.mixture}")
    seed_everything(args.seed)

    records = load_existing(Path("data/distill/responses.jsonl"))
    selected = build_mixture(records, MIXTURES[args.mixture], args.budget_tokens)
    rng = random.Random(args.seed)
    shuffled = list(selected)
    rng.shuffle(shuffled)
    n_val = max(4, int(len(shuffled) * 0.1))
    val_records, train_records = shuffled[:n_val], shuffled[n_val:]

    target = load_target(dtype=args.dtype, device=args.device)
    train_examples = build_examples(
        (r.as_dict() for r in train_records), args.max_seq_len, target.device
    )
    val_examples = build_examples(
        (r.as_dict() for r in val_records), args.max_seq_len, target.device
    )
    print(f"train={len(train_examples)} val={len(val_examples)}", flush=True)

    config = EagleConfig(
        w_regression=1.0,
        w_cross_entropy=weight,
        epochs=args.epochs,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    result = train_eagle(target, train_examples, val_examples, config, out_dir)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "mixture": args.mixture,
        "config": config.__dict__,
        "num_parameters": result["num_parameters"],
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
        "history": result["history"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
