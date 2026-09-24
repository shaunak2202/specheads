"""Manual KV cache manipulation.

Everything here is written against the *installed* transformers, which is 5.x --
the cache API differs from the 4.x one most references describe. Two facts were
read out of `transformers.cache_utils` rather than assumed, and both matter:

* A `DynamicLayer` stores `.keys` / `.values` of shape
  ``[batch, num_kv_heads, seq, head_dim]``, appended along ``dim=-2``.
* `DynamicCache.crop(n)` takes the number of tokens **to remove**, not a target
  length. Passing a target length there silently truncates to the wrong place.

Tree verification needs something `crop` cannot express: keep the prefix and a
scattered subset of the candidate positions, dropping the rejected branches from
the middle. That is `prune_to_indices`.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers.cache_utils import Cache


def cache_length(cache: Cache | None) -> int:
    """Number of cached positions, or 0 for an empty/absent cache."""
    if cache is None:
        return 0
    return int(cache.get_seq_length())


def _layers(cache: Cache) -> list:
    """The per-layer cache objects, which hold the actual key/value tensors."""
    layers = getattr(cache, "layers", None)
    if layers is None:  # pragma: no cover - guards against a future API change
        raise AttributeError(
            "cache has no `.layers`; the installed transformers cache API differs "
            "from the one this module was written against (5.x). Re-read "
            "transformers.cache_utils before changing this."
        )
    return layers


def prune_to_indices(cache: Cache, indices: Sequence[int] | torch.Tensor) -> None:
    """Keep only `indices` along the sequence axis, in the order given, in place.

    This is the step that makes tree verification correct. After a tree forward
    the cache holds every candidate, including the rejected branches; leaving
    them there would let the next step attend to tokens the model never actually
    emitted, which corrupts generation without raising anything.

    Args:
        cache: the cache to prune.
        indices: absolute positions to keep. Order is preserved, so the caller
            is responsible for passing them in sequence order.
    """
    layers = _layers(cache)
    if not layers:
        return

    index_tensor: torch.Tensor | None = None
    for layer in layers:
        keys = getattr(layer, "keys", None)
        if keys is None:
            continue  # layer never initialized (no forward yet)
        if index_tensor is None or index_tensor.device != keys.device:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=keys.device)
        layer.keys = keys.index_select(-2, index_tensor).contiguous()
        layer.values = layer.values.index_select(-2, index_tensor).contiguous()


def rollback_to(cache: Cache, length: int) -> None:
    """Truncate the cache to the first `length` positions.

    Wraps `Cache.crop`, which is mid-deprecation and whose sign is load-bearing:
    a **negative** argument removes that many tokens, while a **positive** one
    takes the legacy meaning of "final absolute size", logs a deprecation
    warning, and is scheduled for removal in transformers 5.18. Passing
    ``current - length`` therefore truncates to the wrong place *and* breaks on
    upgrade. We pass the negative form so the meaning is unambiguous now and
    survives 5.18.
    """
    current = cache_length(cache)
    if length > current:
        raise ValueError(f"cannot roll back to {length}: cache only holds {current}")
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    to_remove = current - length
    if to_remove:
        cache.crop(-to_remove)
