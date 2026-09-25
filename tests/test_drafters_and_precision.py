"""Tests for the drafter architectures and the fp16 tie classification.

The ULP test matters more than it looks: the entire losslessness conclusion is
"every divergence sits within one fp16 ULP", so if `fp16_ulp` is wrong that
conclusion is unfounded.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import TREES, fp16_ulp  # noqa: E402

from specheads.decode.speculative import DraftContext  # noqa: E402
from specheads.decode.tree import TreeSpec  # noqa: E402
from specheads.model.medusa_drafter import MedusaDrafter  # noqa: E402
from specheads.model.medusa_heads import MedusaHeads, head_targets  # noqa: E402


# --- fp16 numerics -----------------------------------------------------------


def test_fp16_ulp_matches_the_observed_divergence_gap():
    """The real divergence at logit magnitude 19.27 had gap exactly 0.015625."""
    assert fp16_ulp(19.265625) == pytest.approx(0.015625)
    assert fp16_ulp(19.265625) == pytest.approx(2.0**-6)


def test_fp16_ulp_matches_torch_spacing():
    """Cross-check against torch's own fp16 representation rather than our algebra."""
    for magnitude in (1.0, 3.7, 19.27, 40.0, 100.0):
        value = torch.tensor(magnitude, dtype=torch.float16)
        following = torch.nextafter(value, torch.tensor(float("inf"), dtype=torch.float16))
        assert fp16_ulp(magnitude) == pytest.approx(float(following - value), rel=1e-6)


def test_fp16_ulp_grows_with_magnitude():
    assert fp16_ulp(1.0) < fp16_ulp(19.0) < fp16_ulp(1000.0)


def test_fp16_ulp_handles_zero_without_dividing_by_log_zero():
    assert fp16_ulp(0.0) > 0


def test_values_closer_than_one_ulp_are_indistinguishable_in_fp16():
    """The mechanism behind the divergences, asserted rather than argued."""
    a = torch.tensor(19.272718, dtype=torch.float32)
    b = torch.tensor(19.268789, dtype=torch.float32)
    assert float(a - b) < fp16_ulp(19.27)
    assert a.half() == b.half()  # same fp16 value -> argmax order decides


# --- tree names --------------------------------------------------------------


def test_tree_names_contain_no_commas():
    """--trees is comma-separated; a comma in a name silently drops that config."""
    for name in TREES:
        assert "," not in name


# --- Medusa ------------------------------------------------------------------


def test_medusa_param_count_matches_the_plan():
    heads = MedusaHeads(1536, num_heads=5, num_resblocks=1)
    assert heads.num_parameters == 11_804_160


def test_medusa_heads_start_as_identity():
    """Zero-init means each head initially reproduces the base model's output."""
    heads = MedusaHeads(32, num_heads=3)
    x = torch.randn(4, 32)
    out = heads(x)
    assert out.shape == (3, 4, 32)
    for k in range(3):
        assert torch.allclose(out[k], x)


def test_head_targets_align_to_t_plus_k_plus_2():
    """Head k predicts t+k+2; an off-by-one trains the wrong horizon silently."""
    slices = head_targets(torch.zeros(1, 10), num_heads=3)
    src, dst = slices[0]
    assert (src.start, src.stop, dst.start, dst.stop) == (0, 8, 2, 10)
    src, dst = slices[2]
    assert (src.start, src.stop, dst.start, dst.stop) == (0, 6, 4, 10)


def test_head_targets_degenerate_for_short_sequences():
    for src, dst in head_targets(torch.zeros(1, 2), num_heads=3):
        assert src.stop - src.start == 0


def test_medusa_drafter_produces_distinct_siblings():
    """accept_path requires it: duplicate siblings make the accepted path ambiguous."""
    torch.manual_seed(0)
    hidden = 32
    heads = MedusaHeads(hidden, num_heads=3)
    # Perturb away from identity so the heads differ from one another.
    for head in heads.heads:
        torch.nn.init.normal_(head[0].linear.weight, std=0.5)
    lm_head = torch.nn.Linear(hidden, 50, bias=False)

    drafter = MedusaDrafter(heads, lm_head)
    spec = TreeSpec.from_widths((3, 2))
    tokens = drafter.draft(
        spec, DraftContext(hidden=torch.randn(hidden), logits=torch.randn(50), pending_token=1)
    )
    assert tokens.shape == (spec.size,)

    by_parent: dict[int, list[int]] = {}
    for node, parent in enumerate(spec.parents):
        by_parent.setdefault(parent, []).append(int(tokens[node]))
    for siblings in by_parent.values():
        assert len(siblings) == len(set(siblings))


def test_medusa_drafter_rejects_a_tree_deeper_than_its_heads():
    heads = MedusaHeads(16, num_heads=2)
    drafter = MedusaDrafter(heads, torch.nn.Linear(16, 20, bias=False))
    with pytest.raises(ValueError, match="exceeds"):
        drafter.draft(
            TreeSpec.chain(4),
            DraftContext(hidden=torch.randn(16), logits=torch.randn(20), pending_token=0),
        )
