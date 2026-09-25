#!/usr/bin/env python3
"""Phase 5 open question: choose EAGLE's cross-entropy weight empirically.

Trains one short run per candidate weight on an identical slice, with an
identical seed and an identical step count, so the only thing that differs is
the weighting. Selection is on the held-out validation split -- never the eval
prompts -- and the table is written out before any full training run.
"""

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
    parser.add_argument("--weights", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--mixture", default="mixed", choices=sorted(MIXTURES))
    parser.add_argument("--budget-tokens", type=int, default=30000)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/eagle_loss_sweep"))
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
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
    print(f"sweep on {len(train_examples)} train / {len(val_examples)} val examples", flush=True)

    rows = []
    for weight in weights:
        print(f"\n--- w_cross_entropy = {weight} ---", flush=True)
        config = EagleConfig(
            w_regression=1.0,
            w_cross_entropy=weight,
            epochs=1,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        result = train_eagle(
            target,
            train_examples,
            val_examples,
            config,
            args.out / f"w{weight}",
            verbose=True,
        )
        final = result["history"][-1]
        rows.append(
            {
                "w_cross_entropy": weight,
                "w_regression": 1.0,
                "train_loss": final["train_loss"],
                "val_regression": final["val_regression"],
                "val_cross_entropy": final["val_cross_entropy"],
                "val_top1_accuracy": final["val_top1_accuracy"],
                "steps": args.max_steps,
            }
        )
        print(
            f"  -> val_top1 {final['val_top1_accuracy']:.4f} "
            f"val_ce {final['val_cross_entropy']:.4f} val_reg {final['val_regression']:.4f}",
            flush=True,
        )

    # Selection proxy: top-1 accuracy of the drafted token on validation. Mean
    # accepted length is what we ultimately care about, but it is monotone in
    # draft accuracy and far cheaper to compute during a sweep.
    best = max(rows, key=lambda r: r["val_top1_accuracy"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "settings": {
            "mixture": args.mixture,
            "budget_tokens": args.budget_tokens,
            "max_steps": args.max_steps,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "device": str(target.device),
            "dtype": args.dtype,
        },
        "selection_metric": "val_top1_accuracy",
        "rows": rows,
        "best": best,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.json").write_text(json.dumps(payload, indent=2))
    print(f"\nbest w_cross_entropy = {best['w_cross_entropy']} "
          f"(val_top1 {best['val_top1_accuracy']:.4f})")
    print(f"wrote {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
