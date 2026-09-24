#!/usr/bin/env python3
"""Hard Rule 2 gate: speculative greedy must equal vanilla greedy, token for token.

Runs against the real target at a chosen precision and writes a JSON record to
`results/`. Any divergence is reported with the prompt, the token index, and the
target's top-2 logits and their gap at that position -- the evidence needed to
tell a genuine fp16 near-tie from a cache or mask bug. It is never summarised
away.

Usage:
  python scripts/check_losslessness.py --device mps --dtype float16 --out results/losslessness_mps_fp16
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from specheads.decode.drafters import OracleDrafter, RandomDrafter
from specheads.decode.speculative import speculative_generate
from specheads.decode.tree import TreeSpec
from specheads.decode.vanilla import generate
from specheads.model.target import encode_chat, load_target
from specheads.utils.env import capture_env
from specheads.utils.seed import seed_everything

PROMPTS = [
    ("chat", "Explain why the sky is blue in two sentences."),
    ("chat", "Give me three tips for writing clear technical documentation."),
    ("code", "Write a Python function that reverses a linked list."),
    ("code", "Implement binary search over a sorted list in Python."),
    ("math", "If a train travels 60 km in 45 minutes, what is its speed in km/h?"),
    ("math", "A shirt costs $40 after a 20% discount. What was the original price?"),
]

TREES = {
    "chain-1": TreeSpec.chain(1),
    "chain-2": TreeSpec.chain(2),
    "chain-5": TreeSpec.chain(5),
    "tree(3,2)": TreeSpec.from_widths((3, 2)),
    "tree(4,2,2)": TreeSpec.from_widths((4, 2, 2)),
}


def divergence_evidence(model, ids, tokens, index, device):
    """Top-2 logits and their gap at the first divergent position.

    A near-zero gap means fp16 reduction order flipped a genuine tie, which is a
    numerics fact about the hardware. A large gap means the bug is ours.
    """
    prefix = torch.cat(
        [ids, torch.tensor([tokens[:index]], dtype=torch.long, device=device)], dim=1
    )
    with torch.no_grad():
        logits = model(input_ids=prefix, use_cache=False, logits_to_keep=1).logits[0, -1]
    top = torch.topk(logits.float(), k=2)
    return {
        "top1_token": int(top.indices[0]),
        "top2_token": int(top.indices[1]),
        "top1_logit": float(top.values[0]),
        "top2_logit": float(top.values[1]),
        "gap": float(top.values[0] - top.values[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/losslessness"))
    args = parser.parse_args()

    seed_everything(args.seed)
    target = load_target(dtype=args.dtype, device=args.device)
    eos = target.tokenizer.eos_token_id

    checks: list[dict] = []
    divergences: list[dict] = []

    for index, (domain, prompt) in enumerate(PROMPTS):
        ids = encode_chat(target.tokenizer, prompt, target.device)
        baseline, _, _ = generate(
            target.model, ids, max_new_tokens=args.max_new_tokens, eos_token_id=eos
        )

        for tree_name, spec in TREES.items():
            drafters = {
                "random": RandomDrafter(target.vocab_size, seed=index),
                "oracle": OracleDrafter(baseline.tokens, target.vocab_size),
            }
            for drafter_name, drafter in drafters.items():
                result, stats = speculative_generate(
                    target.model,
                    ids,
                    drafter,
                    spec,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos,
                )
                lossless = result.tokens == baseline.tokens
                record = {
                    "prompt_index": index,
                    "domain": domain,
                    "tree": tree_name,
                    "drafter": drafter_name,
                    "lossless": lossless,
                    "mean_accepted_length": stats.mean_accepted_length,
                    "mean_tokens_per_forward": stats.mean_emitted_per_step,
                    "forward_passes": result.forward_passes,
                    "vanilla_forward_passes": baseline.forward_passes,
                    "tokens_generated": result.num_tokens,
                }
                if not lossless:
                    first = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(result.tokens, baseline.tokens))
                            if a != b
                        ),
                        min(len(result.tokens), len(baseline.tokens)),
                    )
                    record["divergence"] = {
                        "index": first,
                        "prompt": prompt,
                        "speculative": result.tokens[first : first + 3],
                        "vanilla": baseline.tokens[first : first + 3],
                        "evidence": divergence_evidence(
                            target.model, ids, baseline.tokens, first, target.device
                        ),
                    }
                    divergences.append(record)
                checks.append(record)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "env": capture_env().as_dict(),
        "settings": {
            "device": str(target.device),
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "model": target.model.name_or_path,
        },
        "n_checks": len(checks),
        "n_divergences": len(divergences),
        "all_lossless": not divergences,
        "checks": checks,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "losslessness.json").write_text(json.dumps(payload, indent=2))

    print(f"{len(checks)} checks, {len(divergences)} divergences")
    print(f"wrote {args.out / 'losslessness.json'}")
    return 0 if not divergences else 1


if __name__ == "__main__":
    raise SystemExit(main())
