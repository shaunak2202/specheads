#!/usr/bin/env python3
"""Phase 6: evaluate a drafter per domain -- acceptance, losslessness, throughput.

Reports two classes of metric, kept separate on purpose:

* **Hardware-independent** -- mean accepted length, tokens per forward, per-depth
  acceptance, losslessness. These are properties of the drafter and the target,
  so they are comparable across machines and are what the domain-shift claim
  rests on.
* **Hardware-dependent** -- tokens/sec and speedup. Only comparable against a
  vanilla baseline measured in the *same* session on the *same* device, which is
  why one is always re-run here rather than read from an earlier file.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from specheads.bench.metrics import bootstrap_ci, speedup
from specheads.bench.timing import synchronize
from specheads.decode.speculative import speculative_generate
from specheads.decode.tree import TreeSpec
from specheads.decode.vanilla import generate
from specheads.model.medusa_drafter import MedusaDrafter
from specheads.model.medusa_heads import MedusaHeads
from specheads.model.target import encode_chat, load_target
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything

TREES = {
    "chain-2": TreeSpec.chain(2),
    "chain-3": TreeSpec.chain(3),
    "chain-5": TreeSpec.chain(5),
    "tree(3,2)": TreeSpec.from_widths((3, 2)),
    "tree(4,2,2)": TreeSpec.from_widths((4, 2, 2)),
}


def load_heads(path: Path, hidden_size: int, device, dtype=torch.float32) -> MedusaHeads:
    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    heads = MedusaHeads(
        hidden_size,
        num_heads=config.get("num_heads", 5),
        num_resblocks=config.get("num_resblocks", 1),
    ).to(device=device, dtype=dtype)
    heads.load_state_dict(state["heads"])
    heads.eval()
    return heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--label", required=True, help="name for this drafter, e.g. medusa_chat")
    parser.add_argument("--domains", default="chat,code,math")
    parser.add_argument("--trees", default="chain-2,chain-3,chain-5,tree(3,2),tree(4,2,2)")
    parser.add_argument("--n-prompts", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or Path(f"results/eval_{args.label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    target = load_target(dtype=args.dtype, device=args.device)
    eos = target.tokenizer.eos_token_id
    heads = load_heads(args.heads, target.hidden_size, target.device)
    drafter = MedusaDrafter(heads, target.lm_head)

    trees = {name: TREES[name] for name in args.trees.split(",") if name in TREES}
    rows: list[dict] = []

    for domain in args.domains.split(","):
        prompts = json.loads((args.prompts_dir / f"eval_{domain}.json").read_text())
        prompts = prompts[: args.n_prompts]
        print(f"\n=== {domain}: {len(prompts)} prompts ===", flush=True)

        # Vanilla baseline, same session and device as every method below.
        baselines: list[list[int]] = []
        vanilla_tps: list[float] = []
        for prompt in prompts:
            ids = encode_chat(target.tokenizer, prompt, target.device)
            synchronize(target.device)
            start = time.perf_counter()
            result, _, _ = generate(
                target.model, ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos
            )
            synchronize(target.device)
            elapsed = time.perf_counter() - start
            baselines.append(result.tokens)
            vanilla_tps.append(result.num_tokens / elapsed if elapsed else 0.0)

        vanilla_median = sorted(vanilla_tps)[len(vanilla_tps) // 2]
        rows.append(
            {
                "drafter": "vanilla",
                "domain": domain,
                "tree": None,
                "median_tokens_per_second": vanilla_median,
                "mean_tokens_per_forward": 1.0,
                "mean_accepted_length": 0.0,
                "lossless": True,
                "n_prompts": len(prompts),
            }
        )
        print(f"vanilla: {vanilla_median:.1f} tok/s (median)", flush=True)

        for tree_name, spec in trees.items():
            accepted: list[float] = []
            per_forward: list[float] = []
            tps: list[float] = []
            depth_hits: dict[int, int] = {}
            depth_attempts: dict[int, int] = {}
            mismatches = 0

            for prompt, expected in zip(prompts, baselines):
                ids = encode_chat(target.tokenizer, prompt, target.device)
                synchronize(target.device)
                start = time.perf_counter()
                result, stats = speculative_generate(
                    target.model,
                    ids,
                    drafter,
                    spec,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos,
                )
                synchronize(target.device)
                elapsed = time.perf_counter() - start

                if result.tokens != expected:
                    mismatches += 1
                accepted.append(stats.mean_accepted_length)
                per_forward.append(stats.mean_emitted_per_step)
                tps.append(result.num_tokens / elapsed if elapsed else 0.0)
                for depth, count in stats.depth_attempts.items():
                    depth_attempts[depth] = depth_attempts.get(depth, 0) + count
                for depth, count in stats.depth_hits.items():
                    depth_hits[depth] = depth_hits.get(depth, 0) + count

            median_tps = sorted(tps)[len(tps) // 2]
            low, high = bootstrap_ci(accepted, seed=args.seed)
            row = {
                "drafter": args.label,
                "domain": domain,
                "tree": tree_name,
                "mean_accepted_length": sum(accepted) / len(accepted),
                "accepted_ci_low": low,
                "accepted_ci_high": high,
                "mean_tokens_per_forward": sum(per_forward) / len(per_forward),
                "median_tokens_per_second": median_tps,
                "speedup_vs_vanilla": speedup(median_tps, vanilla_median),
                "lossless": mismatches == 0,
                "n_mismatches": mismatches,
                "acceptance_by_depth": {
                    str(d): depth_hits.get(d, 0) / a for d, a in sorted(depth_attempts.items()) if a
                },
                "n_prompts": len(prompts),
            }
            rows.append(row)
            print(
                f"{tree_name:11} accepted={row['mean_accepted_length']:.3f} "
                f"tok/fwd={row['mean_tokens_per_forward']:.3f} "
                f"tok/s={median_tps:.1f} speedup={row['speedup_vs_vanilla']:.2f}x "
                f"lossless={row['lossless']}",
                flush=True,
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "settings": {
            "label": args.label,
            "heads": str(args.heads),
            "device": str(target.device),
            "dtype": args.dtype,
            "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "hardware_note": (
            "tokens/sec and speedup describe THIS device and do not transfer. "
            "mean_accepted_length, tokens_per_forward, acceptance_by_depth and "
            "lossless are properties of the drafter and target and do transfer."
        ),
        "rows": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
