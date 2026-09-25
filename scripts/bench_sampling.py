#!/usr/bin/env python3
"""Phase 6 experiment 3: speculative decoding at temperature > 0.

Reports both verification rules side by side. They are not interchangeable:
`rejection` preserves the target distribution and `typical` does not, so the
higher acceptance of `typical` is bought with fidelity, not won.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from specheads.bench.metrics import bootstrap_ci
from specheads.bench.timing import synchronize
from specheads.decode.sampling import speculative_sample_generate
from specheads.decode.vanilla import generate
from specheads.model.medusa_drafter import MedusaDrafter
from specheads.model.medusa_heads import MedusaHeads
from specheads.model.target import encode_chat, load_target
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything


def load_heads(path: Path, hidden_size: int, device) -> MedusaHeads:
    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    heads = MedusaHeads(hidden_size, num_heads=config.get("num_heads", 5),
                        num_resblocks=config.get("num_resblocks", 1)).to(device, torch.float32)
    heads.load_state_dict(state["heads"])
    heads.eval()
    return heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, default=Path("results/medusa_mixed/heads.pt"))
    parser.add_argument("--label", default="medusa_mixed")
    parser.add_argument("--domains", default="chat,code,math")
    parser.add_argument("--temperatures", default="0.7,1.0")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-prompts", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/sampling"))
    args = parser.parse_args()

    seed_everything(args.seed)
    target = load_target(dtype=args.dtype, device=args.device)
    eos = target.tokenizer.eos_token_id
    drafter = MedusaDrafter(load_heads(args.heads, target.hidden_size, target.device),
                            target.lm_head)

    rows = []
    for domain in args.domains.split(","):
        prompts = json.loads(Path(f"data/prompts/eval_{domain}.json").read_text())[: args.n_prompts]
        for temperature in [float(t) for t in args.temperatures.split(",")]:
            # Vanilla sampling baseline, same session and device.
            tps_vanilla = []
            for prompt in prompts:
                ids = encode_chat(target.tokenizer, prompt, target.device)
                gen = torch.Generator(device="cpu").manual_seed(args.seed)
                synchronize(target.device)
                start = time.perf_counter()
                res, _, _ = generate(target.model, ids, max_new_tokens=args.max_new_tokens,
                                     greedy=False, temperature=temperature,
                                     eos_token_id=eos, generator=gen)
                synchronize(target.device)
                elapsed = time.perf_counter() - start
                tps_vanilla.append(res.num_tokens / elapsed if elapsed else 0.0)
            median_vanilla = sorted(tps_vanilla)[len(tps_vanilla) // 2]
            rows.append({"mode": "vanilla_sampling", "domain": domain,
                         "temperature": temperature, "median_tokens_per_second": median_vanilla,
                         "mean_accepted_length": 0.0, "preserves_distribution": True})

            for mode in ("rejection", "typical"):
                accepted, tps = [], []
                for prompt in prompts:
                    ids = encode_chat(target.tokenizer, prompt, target.device)
                    gen = torch.Generator(device="cpu").manual_seed(args.seed)
                    synchronize(target.device)
                    start = time.perf_counter()
                    res, stats = speculative_sample_generate(
                        target.model, ids, drafter, args.depth, args.max_new_tokens,
                        temperature=temperature, mode=mode, eos_token_id=eos, generator=gen,
                    )
                    synchronize(target.device)
                    elapsed = time.perf_counter() - start
                    accepted.append(stats["mean_accepted_length"])
                    tps.append(res.num_tokens / elapsed if elapsed else 0.0)

                median = sorted(tps)[len(tps) // 2]
                low, high = bootstrap_ci(accepted, seed=args.seed)
                row = {
                    "mode": mode, "domain": domain, "temperature": temperature,
                    "mean_accepted_length": sum(accepted) / len(accepted),
                    "accepted_ci_low": low, "accepted_ci_high": high,
                    "median_tokens_per_second": median,
                    "speedup_vs_vanilla_sampling": median / median_vanilla if median_vanilla else float("nan"),
                    # The distinction that must never be lost in a results table.
                    "preserves_distribution": mode == "rejection",
                    "n_prompts": len(prompts),
                }
                rows.append(row)
                print(f"{domain:5} T={temperature} {mode:10} accepted={row['mean_accepted_length']:.3f} "
                      f"tok/s={median:.1f} speedup={row['speedup_vs_vanilla_sampling']:.2f}x "
                      f"exact={row['preserves_distribution']}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "settings": {"label": args.label, "depth": args.depth, "n_prompts": args.n_prompts,
                     "max_new_tokens": args.max_new_tokens, "device": str(target.device),
                     "dtype": args.dtype, "seed": args.seed},
        "note": (
            "rejection preserves the target distribution; typical does not. Higher "
            "acceptance under typical is a fidelity trade, not a free win, and the "
            "two rows are not comparable as if they were the same algorithm."
        ),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sampling.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out / 'sampling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
