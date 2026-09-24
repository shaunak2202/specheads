#!/usr/bin/env python3
"""Phase 2: generate the target's own greedy responses to the training prompts.

Resumable -- rerun after a preemption and it continues from the JSONL already on
disk. Writes `data/distill/responses.jsonl` plus a summary to `results/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from specheads.model.target import load_target
from specheads.train.distill_data import generate_responses, token_counts
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-domain", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--out", type=Path, default=Path("data/distill/responses.jsonl"))
    args = parser.parse_args()

    seed_everything(args.seed)

    prompts: list[tuple[str, str]] = []
    for domain in ("chat", "code", "math"):
        path = args.prompts_dir / f"train_{domain}.json"
        items = json.loads(path.read_text())
        if args.limit_per_domain:
            items = items[: args.limit_per_domain]
        prompts.extend((domain, p) for p in items)
    print(f"{len(prompts)} prompts across domains", flush=True)

    target = load_target(dtype=args.dtype, device=args.device)
    records = generate_responses(
        target,
        prompts,
        out_path=args.out,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    counts = token_counts(records)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "settings": {
            "device": str(target.device),
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "n_records": len(records),
        "response_tokens_by_domain": counts,
        "total_response_tokens": sum(counts.values()),
    }
    out_dir = Path("results/distill")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["response_tokens_by_domain"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
