#!/usr/bin/env python3
"""Root-cause the fp16 losslessness divergences (Hard Rule 2).

Runs the same prompts and drafter at fp16 and fp32 and counts divergences in
each, then measures how often the target's top-1 and top-2 logits are exactly
equal in fp16. If fp32 is perfectly lossless while fp16 is not, and the fp16
divergences sit on exact ties, the cause is precision rather than the tree mask
or cache pruning.

Both precisions run on the **same** backend by default, so the comparison
isolates precision. Comparing across backends (CUDA vs MPS) is a separate
question and is answered by running this script once per machine and diffing the
two result files -- fp16 rounding is identical by IEEE-754, but reduction
*order* is a kernel implementation detail, so an exact tie can break differently
on different hardware. That is precisely why this must be re-run on CUDA.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from specheads.decode.speculative import speculative_generate
from specheads.decode.tree import TreeSpec
from specheads.decode.vanilla import generate
from specheads.model.medusa_drafter import MedusaDrafter
from specheads.model.medusa_heads import MedusaHeads
from specheads.model.target import encode_chat, load_target
from specheads.utils.env import capture_env, git_commit
from specheads.utils.seed import seed_everything


def default_device() -> str:
    """Prefer real CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str) -> None:
    """Release cached blocks on whichever backend is in use."""
    kind = torch.device(device).type
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif kind == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_heads(path: Path, hidden_size: int, device) -> MedusaHeads:
    state = torch.load(path, map_location=device)
    config = state.get("config", {})
    heads = MedusaHeads(
        hidden_size, num_heads=config.get("num_heads", 5),
        num_resblocks=config.get("num_resblocks", 1),
    ).to(device=device, dtype=torch.float32)
    heads.load_state_dict(state["heads"])
    heads.eval()
    return heads


def run_pass(device: str, dtype: str, heads_path: Path, prompts, max_new_tokens, spec):
    """Vanilla vs speculative on one precision. Returns divergences and tie stats."""
    target = load_target(device=device, dtype=dtype)
    eos = target.tokenizer.eos_token_id
    drafter = MedusaDrafter(load_heads(heads_path, target.hidden_size, target.device), target.lm_head)

    divergences = []
    tie_positions = 0
    scored_positions = 0

    for index, prompt in enumerate(prompts):
        ids = encode_chat(target.tokenizer, prompt, target.device)
        baseline, _, _ = generate(
            target.model, ids, max_new_tokens=max_new_tokens, eos_token_id=eos
        )
        result, _ = speculative_generate(
            target.model, ids, drafter, spec, max_new_tokens=max_new_tokens, eos_token_id=eos
        )

        if result.tokens != baseline.tokens:
            first = next(
                (i for i, (a, b) in enumerate(zip(result.tokens, baseline.tokens)) if a != b),
                None,
            )
            record = {"prompt_index": index, "first_diff": first}
            if first is not None:
                prefix = torch.cat(
                    [ids, torch.tensor([baseline.tokens[:first]], device=target.device)], dim=1
                )
                with torch.no_grad():
                    logits = target.model(
                        input_ids=prefix, use_cache=False, logits_to_keep=1
                    ).logits[0, -1].float()
                top = torch.topk(logits, 2)
                record.update(
                    {
                        "vanilla_token": baseline.tokens[first],
                        "speculative_token": result.tokens[first],
                        "top1_logit": float(top.values[0]),
                        "top2_logit": float(top.values[1]),
                        "gap": float(top.values[0] - top.values[1]),
                        "exact_tie": float(top.values[0] - top.values[1]) == 0.0,
                    }
                )
            divergences.append(record)

        # How often is the argmax an exact tie at this precision?
        prefix = torch.cat(
            [ids, torch.tensor([baseline.tokens], device=target.device)], dim=1
        )
        with torch.no_grad():
            all_logits = target.model(input_ids=prefix, use_cache=False).logits[0].float()
        top2 = torch.topk(all_logits, 2, dim=-1).values
        gaps = (top2[:, 0] - top2[:, 1])
        tie_positions += int((gaps == 0).sum())
        scored_positions += int(gaps.numel())

    model = target.model
    del target, drafter, model
    gc.collect()
    empty_cache(device)

    return {
        "device": device,
        "dtype": dtype,
        "n_prompts": len(prompts),
        "n_divergent_prompts": len(divergences),
        "exact_tie_positions": tie_positions,
        "scored_positions": scored_positions,
        "exact_tie_rate": tie_positions / scored_positions if scored_positions else 0.0,
        "divergences": divergences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=Path, default=Path("results/medusa_chat/heads.pt"))
    parser.add_argument("--domain", default="chat")
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--fp16-device", default=None,
        help="backend for the fp16 pass (default: cuda, else mps, else cpu)",
    )
    parser.add_argument(
        "--fp32-device", default=None,
        help="backend for the fp32 control (default: same as --fp16-device, so the "
             "only variable is precision)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/fp16_tie_investigation"))
    args = parser.parse_args()

    fp16_device = args.fp16_device or default_device()
    # Same backend by default: a CPU fp32 control would confound precision with
    # backend and make it impossible to say which one fixed the divergences.
    fp32_device = args.fp32_device or fp16_device

    seed_everything(args.seed)
    prompts = json.loads(Path(f"data/prompts/eval_{args.domain}.json").read_text())[: args.n_prompts]
    spec = TreeSpec.chain(2)

    print(f"=== fp16 ({fp16_device}) ===", flush=True)
    fp16 = run_pass(fp16_device, "float16", args.heads, prompts, args.max_new_tokens, spec)
    print(f"  divergent prompts: {fp16['n_divergent_prompts']}/{fp16['n_prompts']}")
    print(f"  exact-tie rate:    {fp16['exact_tie_rate']:.4%}")

    print(f"=== fp32 ({fp32_device}) ===", flush=True)
    fp32 = run_pass(fp32_device, "float32", args.heads, prompts, args.max_new_tokens, spec)
    print(f"  divergent prompts: {fp32['n_divergent_prompts']}/{fp32['n_prompts']}")
    print(f"  exact-tie rate:    {fp32['exact_tie_rate']:.4%}")

    ties = [d for d in fp16["divergences"] if d.get("exact_tie")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "domain": args.domain,
        "max_new_tokens": args.max_new_tokens,
        "backend": {
            "fp16_device": fp16_device,
            "fp32_device": fp32_device,
            "torch_version": torch.__version__,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cuda_capability": (
                ".".join(map(str, torch.cuda.get_device_capability(0)))
                if torch.cuda.is_available() else None
            ),
        },
        "fp16": fp16,
        "fp32": fp32,
        "conclusion": {
            "fp16_divergences_on_exact_ties": len(ties),
            "fp16_divergences_total": len(fp16["divergences"]),
            "fp32_divergences_total": len(fp32["divergences"]),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "investigation.json").write_text(json.dumps(payload, indent=2))
    print(f"\nfp16 divergences on exact ties: {len(ties)}/{len(fp16['divergences'])}")
    print(f"wrote {args.out / 'investigation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
