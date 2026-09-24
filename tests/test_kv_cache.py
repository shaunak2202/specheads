import pytest
import torch
from transformers import DynamicCache

from specheads.decode.kv_cache import cache_length, prune_to_indices, rollback_to
from specheads.model.target import tiny_target


def filled_cache(model, length: int) -> DynamicCache:
    cache = DynamicCache(config=model.config)
    ids = torch.arange(length).unsqueeze(0) % model.config.vocab_size
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache


@pytest.fixture(scope="module")
def model():
    return tiny_target(vocab_size=128, seed=5).model


def test_cache_length_tracks_forwards(model):
    assert cache_length(None) == 0
    cache = filled_cache(model, 7)
    assert cache_length(cache) == 7


def test_prune_keeps_only_requested_positions(model):
    cache = filled_cache(model, 10)
    before = [layer.keys.clone() for layer in cache.layers]

    keep = [0, 1, 2, 5, 9]
    prune_to_indices(cache, keep)

    assert cache_length(cache) == len(keep)
    for layer, original in zip(cache.layers, before):
        assert torch.equal(layer.keys, original.index_select(-2, torch.tensor(keep)))


def test_prune_preserves_the_given_order(model):
    """Order matters: positions must stay in sequence order after pruning."""
    cache = filled_cache(model, 6)
    original = cache.layers[0].keys.clone()
    prune_to_indices(cache, [4, 1])
    assert torch.equal(cache.layers[0].keys[..., 0, :], original[..., 4, :])
    assert torch.equal(cache.layers[0].keys[..., 1, :], original[..., 1, :])


def test_prune_touches_every_layer(model):
    cache = filled_cache(model, 8)
    prune_to_indices(cache, [0, 3])
    for layer in cache.layers:
        assert layer.keys.shape[-2] == 2
        assert layer.values.shape[-2] == 2


def test_rollback_wraps_crop_in_absolute_terms(model):
    """`crop` takes tokens-to-remove; rollback_to takes a target length."""
    cache = filled_cache(model, 9)
    rollback_to(cache, 4)
    assert cache_length(cache) == 4


def test_rollback_rejects_impossible_lengths(model):
    cache = filled_cache(model, 3)
    with pytest.raises(ValueError, match="cannot roll back"):
        rollback_to(cache, 5)
    with pytest.raises(ValueError, match="non-negative"):
        rollback_to(cache, -1)
