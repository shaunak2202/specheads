"""Correctness for temperature > 0.

Greedy speculation is checked by token-identity. Sampling cannot be: two correct
samplers disagree on nearly every token. The criterion instead is
*distributional* -- the speculative sampler must draw from the same distribution
the target would have -- and that is checked here with a chi-square test rather
than asserted.

The tests deliberately use a **bad** drafter. A good drafter is accepted most of
the time, so the residual path -- the step where bias would actually creep in --
would barely execute.
"""

import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from specheads.decode.sampling import (
    residual_distribution,
    sample_from,
    softmax_with_temperature,
    typical_threshold,
    verify_chain_rejection,
    verify_chain_typical,
)

VOCAB = 12
SAMPLES = 60_000


def fixed_distributions(seed: int = 0):
    """A target p and a deliberately mismatched draft q."""
    generator = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(VOCAB, generator=generator), dim=-1)
    q = torch.softmax(torch.randn(VOCAB, generator=generator) * 2.0, dim=-1)
    return p, q


def expected_counts(probs: torch.Tensor, total: int) -> np.ndarray:
    """Expected counts rescaled to sum exactly to `total`.

    scipy's chisquare rejects inputs whose sums differ, and `p * total` drifts
    by a few ULP.
    """
    values = probs.numpy().astype(np.float64)
    values = values / values.sum()
    return values * total


def empirical_counts(sampler, n: int) -> np.ndarray:
    counts = np.zeros(VOCAB, dtype=np.int64)
    for _ in range(n):
        counts[sampler()] += 1
    return counts


# --- building blocks ---------------------------------------------------------


def test_residual_is_the_positive_part_normalised():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.1, 0.8, 0.1])
    residual = residual_distribution(p, q)
    # (0.4, 0, 0.1) -> normalised
    assert residual.tolist() == pytest.approx([0.8, 0.0, 0.2], abs=1e-6)
    assert float(residual.sum()) == pytest.approx(1.0)


def test_residual_falls_back_to_target_when_draft_dominates():
    p = torch.tensor([0.2, 0.8])
    q = torch.tensor([0.5, 0.5])
    residual = residual_distribution(p, torch.maximum(q, p))
    assert float(residual.sum()) == pytest.approx(1.0)
    assert torch.all(residual >= 0)


def test_softmax_temperature_sharpens_and_flattens():
    logits = torch.tensor([2.0, 1.0, 0.0])
    cold = softmax_with_temperature(logits, 0.1)
    hot = softmax_with_temperature(logits, 10.0)
    assert float(cold.max()) > float(hot.max())
    assert float(cold.sum()) == pytest.approx(1.0)


def test_softmax_rejects_zero_temperature():
    with pytest.raises(ValueError, match="temperature"):
        softmax_with_temperature(torch.zeros(3), 0.0)


def test_typical_threshold_rises_with_target_confidence():
    peaked = typical_threshold(torch.tensor([0.98, 0.01, 0.01]), 0.3, 0.09)
    flat = typical_threshold(torch.tensor([0.34, 0.33, 0.33]), 0.3, 0.09)
    assert peaked > flat


# --- the distributional claim ------------------------------------------------


def test_rejection_sampling_preserves_the_target_distribution():
    """The correctness criterion for sampling, tested rather than assumed.

    Chain of length 1 with a mismatched drafter: the emitted token must be
    distributed exactly as the target p, whatever q is.
    """
    p, q = fixed_distributions()
    generator = torch.Generator().manual_seed(7)

    def draw():
        # The draft token must be SAMPLED from q. The proof of correctness
        # assumes x ~ q; feeding a fixed argmax(q) breaks it (see the test below).
        draft_token = sample_from(q, generator)
        return verify_chain_rejection(
            [draft_token], q.unsqueeze(0), torch.stack([p, p]), generator
        ).tokens[0]

    counts = empirical_counts(draw, SAMPLES)
    statistic, pvalue = chisquare(counts, expected_counts(p, int(counts.sum())))
    assert pvalue > 0.01, f"speculative sampling deviates from target (p={pvalue:.2e})"


def test_rejection_sampling_matches_direct_sampling():
    """Same claim from the other side: against an empirical vanilla baseline."""
    p, q = fixed_distributions(seed=3)

    spec_gen = torch.Generator().manual_seed(11)
    direct_gen = torch.Generator().manual_seed(29)

    spec = empirical_counts(
        lambda: verify_chain_rejection(
            [sample_from(q, spec_gen)], q.unsqueeze(0), torch.stack([p, p]), spec_gen
        ).tokens[0],
        SAMPLES,
    )
    direct = empirical_counts(lambda: sample_from(p, direct_gen), SAMPLES)

    # Compare the two empirical distributions against each other.
    expected = (direct / direct.sum()) * spec.sum()
    mask = expected > 5  # chi-square validity
    observed = spec[mask].astype(np.float64)
    reference = expected[mask] * (observed.sum() / expected[mask].sum())
    statistic, pvalue = chisquare(observed, reference)
    assert pvalue > 0.01, f"speculative and direct sampling differ (p={pvalue:.2e})"


def test_typical_acceptance_is_measurably_biased():
    """The honest counterpart: typical acceptance does NOT preserve p.

    This is asserted so the distinction can never quietly rot into "both modes
    are lossless". If this test ever starts failing, the claim in the README
    about typical acceptance needs revisiting, not the test.
    """
    p, q = fixed_distributions(seed=5)
    generator = torch.Generator().manual_seed(13)
    draft_token = int(torch.argmax(q))

    counts = empirical_counts(
        lambda: verify_chain_typical(
            [draft_token], torch.stack([p, p]), generator=generator
        ).tokens[0],
        SAMPLES,
    )
    statistic, pvalue = chisquare(counts, expected_counts(p, int(counts.sum())))
    assert pvalue < 1e-6, (
        "typical acceptance matched the target distribution, which contradicts "
        f"the documented trade-off (p={pvalue:.3g})"
    )


# --- mechanics ---------------------------------------------------------------


def test_full_acceptance_emits_chain_plus_bonus():
    """A perfect drafter: q == p and the drafted token is accepted throughout."""
    p = torch.zeros(VOCAB)
    p[3] = 1.0
    generator = torch.Generator().manual_seed(0)
    out = verify_chain_rejection([3, 3], torch.stack([p, p]), torch.stack([p, p, p]), generator)
    assert out.n_accepted == 2
    assert out.emitted == 3  # two accepted plus the bonus
    assert out.rejected_at is None


def test_rejection_stops_immediately_and_still_emits_one():
    """A rejected step must still advance by one, exactly as in the greedy case."""
    p = torch.zeros(VOCAB)
    p[1] = 1.0
    q = torch.zeros(VOCAB)
    q[9] = 1.0  # drafts a token the target assigns zero mass
    generator = torch.Generator().manual_seed(0)
    out = verify_chain_rejection([9, 9], torch.stack([q, q]), torch.stack([p, p, p]), generator)
    assert out.n_accepted == 0
    assert out.rejected_at == 0
    assert out.tokens == (1,)


def test_target_probs_length_is_validated():
    p = torch.ones(VOCAB) / VOCAB
    with pytest.raises(ValueError, match="target distributions"):
        verify_chain_rejection([0, 1], torch.stack([p, p]), p.unsqueeze(0))


def test_argmax_drafts_break_the_distribution_guarantee():
    """Why the greedy drafter cannot be reused verbatim for sampling.

    Speculative sampling preserves p only when the draft is drawn from q. A
    top-k/argmax drafter -- which is exactly what `MedusaDrafter.draft` returns
    for greedy decoding -- violates that precondition, and the output is
    measurably biased. Pinned so the requirement cannot be quietly dropped.
    """
    p, q = fixed_distributions(seed=9)
    generator = torch.Generator().manual_seed(17)
    fixed_draft = int(torch.argmax(q))

    counts = empirical_counts(
        lambda: verify_chain_rejection(
            [fixed_draft], q.unsqueeze(0), torch.stack([p, p]), generator
        ).tokens[0],
        SAMPLES,
    )
    statistic, pvalue = chisquare(counts, expected_counts(p, int(counts.sum())))
    assert pvalue < 1e-6, (
        "argmax drafting matched the target distribution; the documented "
        f"precondition (x ~ q) would then be unnecessary (p={pvalue:.3g})"
    )
