"""Hard Rule 2: speculative greedy output must be token-identical to vanilla greedy.

These run on CPU against a tiny randomly-initialised Qwen2, so the whole
correctness story is checkable without a GPU or a trained drafter. That is the
point of testing with two synthetic drafters that bracket the acceptance range:

* `RandomDrafter` has everything rejected, so the loop degrades to one token per
  forward. It proves the fallback path is exact.
* `OracleDrafter` proposes the true continuation, so every step accepts to full
  depth. **This is the one that matters**, because with a random drafter the
  accepted path is almost always empty and the rejected-branch cache pruning --
  the most bug-prone step in the project -- never actually runs.
"""

import pytest
import torch

from specheads.decode.drafters import OracleDrafter, RandomDrafter
from specheads.decode.speculative import speculative_generate
from specheads.decode.tree import TreeSpec
from specheads.decode.vanilla import generate
from specheads.model.target import tiny_target

VOCAB = 256

TREES = {
    "chain-1": TreeSpec.chain(1),
    "chain-2": TreeSpec.chain(2),
    "chain-4": TreeSpec.chain(4),
    "chain-5": TreeSpec.chain(5),
    "tree(2,2)": TreeSpec.from_widths((2, 2)),
    "tree(3,2)": TreeSpec.from_widths((3, 2)),
    "tree(4,2,2)": TreeSpec.from_widths((4, 2, 2)),
}


@pytest.fixture(scope="module")
def target():
    return tiny_target(vocab_size=VOCAB, seed=17)


def prompt(seed: int, length: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, length), generator=generator)


def vanilla_tokens(model, ids, max_new_tokens=24):
    result, _, _ = generate(model, ids, max_new_tokens=max_new_tokens)
    return result.tokens


@pytest.mark.parametrize("name", list(TREES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lossless_with_a_useless_drafter(target, name, seed):
    """Random drafts must still produce exactly vanilla's tokens."""
    ids = prompt(seed)
    expected = vanilla_tokens(target.model, ids)
    result, _ = speculative_generate(
        target.model, ids, RandomDrafter(VOCAB, seed=seed), TREES[name], max_new_tokens=24
    )
    assert result.tokens == expected


@pytest.mark.parametrize("name", list(TREES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lossless_with_a_perfect_drafter(target, name, seed):
    """Full-depth acceptance every step -- this is what exercises cache pruning."""
    ids = prompt(seed)
    expected = vanilla_tokens(target.model, ids)
    result, _ = speculative_generate(
        target.model,
        ids,
        OracleDrafter(expected, VOCAB),
        TREES[name],
        max_new_tokens=24,
    )
    assert result.tokens == expected


def test_perfect_drafter_actually_accepts_full_depth(target):
    """Guards the guard: if acceptance were 0, the test above would prove nothing."""
    ids = prompt(1)
    expected = vanilla_tokens(target.model, ids, max_new_tokens=30)
    spec = TreeSpec.chain(5)
    result, stats = speculative_generate(
        target.model, ids, OracleDrafter(expected, VOCAB), spec, max_new_tokens=30
    )
    assert result.tokens == expected
    assert stats.mean_accepted_length == pytest.approx(5.0)
    # 6 tokens per forward: 5 accepted drafts plus the bonus token.
    assert stats.mean_emitted_per_step == pytest.approx(6.0)
    assert result.forward_passes < 10  # vanilla needs 30


def test_random_drafter_falls_back_to_one_token_per_step(target):
    """The no-acceptance path must cost the same as vanilla, never more tokens."""
    ids = prompt(0)
    _, stats = speculative_generate(
        target.model, ids, RandomDrafter(VOCAB, seed=0), TreeSpec.chain(4), max_new_tokens=20
    )
    assert stats.mean_accepted_length == pytest.approx(0.0)
    assert stats.mean_emitted_per_step == pytest.approx(1.0)


def test_deeper_tree_never_changes_the_output(target):
    """Tree shape is a speed knob and must not be observable in the tokens."""
    ids = prompt(2)
    expected = vanilla_tokens(target.model, ids)
    outputs = []
    for spec in TREES.values():
        result, _ = speculative_generate(
            target.model, ids, OracleDrafter(expected, VOCAB), spec, max_new_tokens=24
        )
        outputs.append(result.tokens)
    assert all(tokens == expected for tokens in outputs)


def test_respects_max_new_tokens_exactly(target):
    """A step can accept several tokens at once and must not overshoot the cap."""
    ids = prompt(3)
    expected = vanilla_tokens(target.model, ids, max_new_tokens=30)
    for cap in (1, 3, 7, 13):
        result, _ = speculative_generate(
            target.model, ids, OracleDrafter(expected, VOCAB), TreeSpec.chain(5), max_new_tokens=cap
        )
        assert len(result.tokens) == cap
        assert result.tokens == expected[:cap]


def test_eos_stops_generation(target):
    """EOS must terminate even when it arrives mid-accepted-path."""
    ids = prompt(1)
    expected = vanilla_tokens(target.model, ids, max_new_tokens=30)
    eos = expected[6]
    result, _ = speculative_generate(
        target.model,
        ids,
        OracleDrafter(expected, VOCAB),
        TreeSpec.chain(5),
        max_new_tokens=30,
        eos_token_id=eos,
    )
    assert result.finished
    assert result.tokens[-1] == eos
    assert result.tokens == expected[: result.tokens.index(eos) + 1]
