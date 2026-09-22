"""The budget drives a GPU-time decision, so its arithmetic is pinned."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from memory_budget import (  # noqa: E402
    TargetConfig,
    build_report,
    eagle_params,
    kv_cache_bytes,
    medusa_params,
)


def test_target_config_matches_published_qwen_shape():
    cfg = TargetConfig()
    assert cfg.hidden_size == 1536
    assert cfg.vocab_size == 151936
    assert cfg.head_dim == 128
    # GQA: 2 KV heads x 128, not 12 x 128. This is why the KV cache is small.
    assert cfg.kv_dim == 256


def test_medusa_params_hand_computed():
    cfg = TargetConfig()
    # One resblock per head: 1536*1536 + 1536 = 2,360,832 params.
    assert medusa_params(cfg, n_heads=1) == 2_360_832
    assert medusa_params(cfg, n_heads=4) == 4 * 2_360_832


def test_duplicating_the_lm_head_would_dwarf_the_heads():
    """The reason the spec shares one frozen LM head across all Medusa heads."""
    cfg = TargetConfig()
    lm_head = cfg.vocab_size * cfg.hidden_size
    # ~24.7x the entire K=4 head stack, and that is for a *single* copy;
    # one per head would add ~933M params to a 1.54B model.
    assert lm_head > 20 * medusa_params(cfg, n_heads=4)
    assert 4 * lm_head > 0.5 * 1_543_714_304


def test_eagle_params_breakdown_sums():
    parts = eagle_params(TargetConfig())
    assert parts["fusion"] + parts["decoder_layer"] == parts["total"]
    # Fusion takes concat(feature, embedding) = 2d -> d.
    assert parts["fusion"] == (2 * 1536) * 1536 + 1536


def test_kv_cache_is_linear_in_sequence_length():
    cfg = TargetConfig()
    assert kv_cache_bytes(cfg, 2048) == 2 * kv_cache_bytes(cfg, 1024)
    # 28 layers x 2 (K,V) x 256 x 2 bytes = 28,672 bytes per token.
    assert kv_cache_bytes(cfg, 1) == 28_672


def test_chunking_cuts_logit_memory_by_the_chunk_ratio():
    report = build_report(seq_len=1024, n_heads=4, chunk=256)
    medusa = report["medusa"]
    ratio = medusa["logits_fp32_gb_unchunked"] / medusa["logits_fp32_gb_chunked"]
    assert abs(ratio - 4.0) < 0.05


def test_both_drafters_fit_in_a_t4():
    report = build_report()
    assert report["medusa"]["est_peak_gb"] < 16.0
    assert report["eagle"]["est_peak_gb"] < 16.0


def test_report_flags_its_own_estimates_as_estimates():
    """Rule 1: nothing derived may masquerade as a measurement."""
    note = build_report()["assumptions"]["note"]
    assert "Analytic only" in note and "not a measurement" in note
