#!/usr/bin/env python3
"""Analytic parameter and memory budget for the drafters on a 16GB T4.

These are *derived* numbers, not measurements: they come from the target
model's published config plus explicit arithmetic, so the Phase 0 plan can
argue about feasibility before any GPU time is spent. Every figure here is
superseded by the measured `torch.cuda.max_memory_allocated()` recorded in
Phase 3 and Phase 5 -- see docs/plan.md.

Run: python scripts/memory_budget.py [--json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class TargetConfig:
    """The frozen target's shape, from Qwen/Qwen2.5-1.5B-Instruct config.json."""

    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Width of the K (or V) projection under GQA."""
        return self.num_key_value_heads * self.head_dim


def medusa_params(cfg: TargetConfig, n_heads: int = 4, n_resblocks: int = 1) -> int:
    """Added parameters for K Medusa heads.

    Each head is `n_resblocks` x [Linear(d, d) + SiLU + residual], then the
    *shared, frozen* LM head. Because Qwen2.5 ties its embeddings, that LM head
    is the input embedding matrix -- 151936 x 1536 = 233M params. Duplicating it
    per head would add ~933M for K=4, nearly two-thirds of the target itself,
    which is exactly why the spec forbids it.
    """
    d = cfg.hidden_size
    per_resblock = d * d + d  # weight + bias
    return n_heads * n_resblocks * per_resblock


def eagle_params(cfg: TargetConfig) -> dict[str, int]:
    """Added parameters for the EAGLE drafter: fusion projection + one decoder layer."""
    d, kv, inter = cfg.hidden_size, cfg.kv_dim, cfg.intermediate_size

    # Input is concat(feature_t, embed(token_{t+1})) -> project back to d.
    fusion = (2 * d) * d + d

    # Qwen2 attention: q/k/v carry bias, o_proj does not.
    attn = (d * d + d) + (d * kv + kv) + (d * kv + kv) + (d * d)
    mlp = 3 * (d * inter)  # gate, up, down
    norms = 2 * d  # input_layernorm + post_attention_layernorm (RMSNorm, no bias)
    layer = attn + mlp + norms

    return {"fusion": fusion, "decoder_layer": layer, "total": fusion + layer}


def logit_bytes(seq_len: int, vocab: int, dtype_bytes: int, n_heads: int = 1) -> int:
    """Bytes for one set of logits -- the term that actually decides the budget.

    At vocab 151936 a single 1024-token sequence of fp32 logits is 622 MB. For
    Medusa that cost is multiplied by K, which is why the plan chunks the
    cross-entropy over the sequence instead of materialising all of it.
    """
    return seq_len * vocab * dtype_bytes * n_heads


def optimizer_bytes(n_params: int) -> int:
    """fp32 master weights + fp32 grads + Adam m and v = 16 bytes per parameter."""
    return n_params * 16


def kv_cache_bytes(cfg: TargetConfig, seq_len: int, dtype_bytes: int = 2) -> int:
    """KV cache for the frozen target. GQA (2 KV heads) keeps this small."""
    per_token = cfg.num_hidden_layers * 2 * cfg.kv_dim * dtype_bytes
    return per_token * seq_len


def build_report(seq_len: int = 1024, n_heads: int = 4, chunk: int = 256) -> dict:
    cfg = TargetConfig()
    target_params = 1_543_714_304  # published total for Qwen2.5-1.5B

    med = medusa_params(cfg, n_heads=n_heads)
    eag = eagle_params(cfg)

    target_fp16 = target_params * 2

    # Cross-entropy is computed in fp32; chunking caps how much is live at once.
    med_logits_full = logit_bytes(seq_len, cfg.vocab_size, 4, n_heads)
    med_logits_chunked = logit_bytes(chunk, cfg.vocab_size, 4, n_heads)
    eag_logits_full = logit_bytes(seq_len, cfg.vocab_size, 4, 1)
    eag_logits_chunked = logit_bytes(chunk, cfg.vocab_size, 4, 1)

    def gb(x: float) -> float:
        return round(x / BYTES_PER_GB, 3)

    return {
        "target": {
            "name": cfg.name,
            "params": target_params,
            "fp16_gb": gb(target_fp16),
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "lm_head_params_if_duplicated": cfg.vocab_size * cfg.hidden_size,
            "kv_cache_gb_at_2048": gb(kv_cache_bytes(cfg, 2048)),
            "kv_cache_kb_per_token": round(kv_cache_bytes(cfg, 1) / 1024, 2),
        },
        "medusa": {
            "n_heads": n_heads,
            "added_params": med,
            "added_params_pct_of_target": round(100 * med / target_params, 3),
            "optimizer_state_gb": gb(optimizer_bytes(med)),
            "logits_fp32_gb_unchunked": gb(med_logits_full),
            "logits_fp32_gb_chunked": gb(med_logits_chunked),
            "est_peak_gb": gb(
                target_fp16 + optimizer_bytes(med) + med_logits_chunked * 2 + 1.0 * BYTES_PER_GB
            ),
        },
        "eagle": {
            "added_params": eag["total"],
            "breakdown": eag,
            "added_params_pct_of_target": round(100 * eag["total"] / target_params, 3),
            "optimizer_state_gb": gb(optimizer_bytes(eag["total"])),
            "logits_fp32_gb_unchunked": gb(eag_logits_full),
            "logits_fp32_gb_chunked": gb(eag_logits_chunked),
            "est_peak_gb": gb(
                target_fp16
                + optimizer_bytes(eag["total"])
                + eag_logits_chunked * 2
                + 1.0 * BYTES_PER_GB
            ),
        },
        "assumptions": {
            "seq_len": seq_len,
            "ce_chunk": chunk,
            "batch_size": 1,
            "activation_slack_gb": 1.0,
            "note": (
                "Analytic only. 'activation_slack_gb' is a flat 1 GB allowance for "
                "transient forward/backward activations under SDPA, not a measurement. "
                "Replace all est_peak_gb with measured max_memory_allocated in Phase 3/5."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    report = build_report(args.seq_len, args.n_heads, args.chunk)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    t, m, e = report["target"], report["medusa"], report["eagle"]
    print(f"Target: {t['name']}  ({t['params']:,} params, {t['fp16_gb']} GB fp16)")
    print(f"  tied embeddings: {t['tie_word_embeddings']}  "
          f"(LM head would cost {t['lm_head_params_if_duplicated']:,} params per copy)")
    print(f"  KV cache: {t['kv_cache_kb_per_token']} KB/token, {t['kv_cache_gb_at_2048']} GB @ 2048\n")
    print(f"Medusa x{m['n_heads']}: +{m['added_params']:,} params "
          f"({m['added_params_pct_of_target']}% of target)")
    print(f"  optimizer state {m['optimizer_state_gb']} GB | "
          f"logits fp32 {m['logits_fp32_gb_unchunked']} GB unchunked -> "
          f"{m['logits_fp32_gb_chunked']} GB chunked")
    print(f"  estimated peak: {m['est_peak_gb']} GB\n")
    print(f"EAGLE: +{e['added_params']:,} params ({e['added_params_pct_of_target']}% of target)")
    print(f"  optimizer state {e['optimizer_state_gb']} GB | "
          f"logits fp32 {e['logits_fp32_gb_unchunked']} GB unchunked -> "
          f"{e['logits_fp32_gb_chunked']} GB chunked")
    print(f"  estimated peak: {e['est_peak_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
