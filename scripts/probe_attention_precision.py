#!/usr/bin/env python3
"""Where do the fp16 argmax ties actually come from? Kernel, not just precision.

The earlier investigation showed fp16 diverges where fp32 does not, and
concluded "fp16 ties". That is incomplete. This probe holds precision fixed and
varies only the attention implementation, and the ties disappear -- so the ties
are produced by the *kernel's internal accumulation policy*, not by fp16 storage
as such.

The mechanism is in transformers' own source: `eager_attention_forward` calls
``softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)``, i.e. it
upcasts the softmax to fp32 and casts back. SDPA's fused kernel on MPS does not,
so the attention output is coarser and downstream logits collide.

This matters for portability. CUDA's SDPA dispatches to FlashAttention or the
memory-efficient kernel, which accumulate in fp32 even for fp16 inputs, so the
tie rate on a T4 may be far lower than on MPS -- possibly zero. Attributing the
divergences to "fp16" alone would have made that prediction impossible.
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


def probe(device, dtype, attn, heads_path, prompts, max_new_tokens, spec) -> dict:
    target = load_target(device=device, dtype=dtype, attn_implementation=attn)
    eos = target.tokenizer.eos_token_id
    drafter = MedusaDrafter(load_heads(heads_path, target.hidden_size, target.device),
                            target.lm_head)

    divergent = 0
    ties = scored = 0
    for prompt in prompts:
        ids = encode_chat(target.tokenizer, prompt, target.device)
        baseline, _, _ = generate(
            target.model, ids, max_new_tokens=max_new_tokens, eos_token_id=eos
        )
        result, _ = speculative_generate(
            target.model, ids, drafter, spec, max_new_tokens=max_new_tokens, eos_token_id=eos
        )
        if result.tokens != baseline.tokens:
            divergent += 1

        full = torch.cat([ids, torch.tensor([baseline.tokens], device=target.device)], dim=1)
        with torch.no_grad():
            logits = target.model(input_ids=full, use_cache=False).logits[0].float()
        top2 = torch.topk(logits, 2, dim=-1).values
        gaps = top2[:, 0] - top2[:, 1]
        ties += int((gaps == 0).sum())
        scored += int(gaps.numel())

    model = target.model
    del target, drafter, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "device": device,
        "dtype": dtype,
        "attn_implementation": attn,
        "n_prompts": len(prompts),
        "divergent_prompts": divergent,
        "exact_tie_positions": ties,
        "scored_positions": scored,
        "exact_tie_rate": ties / scored if scored else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--heads", type=Path, default=Path("results/medusa_chat/heads.pt"))
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/attention_precision_probe"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    prompts = json.loads(Path("data/prompts/eval_chat.json").read_text())[: args.n_prompts]
    spec = TreeSpec.chain(2)

    configs = [
        (device, "float16", "sdpa"),
        (device, "float16", "eager"),
        (device, "float32", "sdpa"),
    ]
    rows = []
    for dev, dtype, attn in configs:
        print(f"=== {dtype} / {attn} on {dev} ===", flush=True)
        row = probe(dev, dtype, attn, args.heads, prompts, args.max_new_tokens, spec)
        rows.append(row)
        print(f"  divergent {row['divergent_prompts']}/{row['n_prompts']} | "
              f"tie rate {row['exact_tie_rate']:.4%}", flush=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "env": capture_env().as_dict(),
        "settings": {
            "device": device, "n_prompts": args.n_prompts,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        },
        "mechanism": (
            "transformers' eager_attention_forward upcasts the softmax to fp32 "
            "(softmax(..., dtype=torch.float32).to(query.dtype)); the fused SDPA "
            "kernel on this backend does not. Holding precision fixed and swapping "
            "only the kernel therefore changes the tie rate."
        ),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
