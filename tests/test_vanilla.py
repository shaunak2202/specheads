"""Phase 1 gate: our decode loop must match `model.generate` exactly."""

import pytest
import torch

from specheads.decode.vanilla import generate
from specheads.model.target import tiny_target

VOCAB = 256


@pytest.fixture(scope="module")
def target():
    return tiny_target(vocab_size=VOCAB, seed=3)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_greedy_matches_hf_generate(target, seed):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (1, 6), generator=generator)

    ours, _, _ = generate(target.model, ids, max_new_tokens=20)
    theirs = target.model.generate(
        ids, max_new_tokens=20, do_sample=False, pad_token_id=0, use_cache=True
    )
    assert ours.tokens == theirs[0, ids.shape[1] :].tolist()


def test_vanilla_uses_one_forward_per_token(target):
    """The baseline the speculative methods are measured against."""
    ids = torch.tensor([[1, 2, 3]])
    result, _, _ = generate(target.model, ids, max_new_tokens=15)
    # One prefill plus one forward per token after the first.
    assert result.forward_passes == result.num_tokens
    assert result.num_tokens == 15


def test_cache_grows_with_generation(target):
    ids = torch.tensor([[4, 5, 6, 7]])
    result, cache, _ = generate(target.model, ids, max_new_tokens=10)
    # Prompt plus every token that was fed back in (the last one never is).
    assert cache.get_seq_length() == ids.shape[1] + result.num_tokens - 1


def test_eos_halts_generation(target):
    ids = torch.tensor([[1, 2, 3]])
    baseline, _, _ = generate(target.model, ids, max_new_tokens=12)
    eos = baseline.tokens[4]
    result, _, _ = generate(target.model, ids, max_new_tokens=12, eos_token_id=eos)
    assert result.finished
    assert result.tokens[-1] == eos


def test_sampling_is_reproducible_and_respects_temperature(target):
    ids = torch.tensor([[2, 4, 8]])
    first, _, _ = generate(
        target.model, ids, max_new_tokens=12, greedy=False, temperature=0.9,
        generator=torch.Generator().manual_seed(42),
    )
    second, _, _ = generate(
        target.model, ids, max_new_tokens=12, greedy=False, temperature=0.9,
        generator=torch.Generator().manual_seed(42),
    )
    assert first.tokens == second.tokens


def test_low_temperature_sampling_converges_to_greedy(target):
    ids = torch.tensor([[3, 6, 9]])
    greedy, _, _ = generate(target.model, ids, max_new_tokens=10)
    sampled, _, _ = generate(
        target.model, ids, max_new_tokens=10, greedy=False, temperature=1e-4,
        generator=torch.Generator().manual_seed(0),
    )
    assert sampled.tokens == greedy.tokens


def test_top_p_never_selects_an_empty_set(target):
    """A tiny top_p must still keep the single most likely token."""
    ids = torch.tensor([[1, 1, 1]])
    result, _, _ = generate(
        target.model, ids, max_new_tokens=5, greedy=False, temperature=1.0, top_p=1e-6,
        generator=torch.Generator().manual_seed(0),
    )
    assert len(result.tokens) == 5


def test_sampling_rejects_zero_temperature(target):
    with pytest.raises(ValueError, match="temperature"):
        generate(target.model, torch.tensor([[1, 2]]), max_new_tokens=2,
                 greedy=False, temperature=0.0)


def test_rejects_batched_input(target):
    with pytest.raises(ValueError, match=r"\[1, seq\]"):
        generate(target.model, torch.ones(2, 3, dtype=torch.long), max_new_tokens=2)
