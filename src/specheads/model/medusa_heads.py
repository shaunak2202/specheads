"""Medusa-style heads: K extra decoding heads on the target's final hidden state.

Head *k* predicts the token at ``t + k + 2`` while the base LM head predicts
``t + 1``, so K heads let a step propose K+1 tokens. Each head is a residual
block followed by the **shared, frozen** LM head.

Sharing that LM head is a correctness requirement, not only a memory
optimisation. Qwen2.5 sets `tie_word_embeddings: True`, so the output projection
*is* the input embedding matrix: a gradient arriving there would alter the
target's own embeddings and change the very model losslessness is defined
against. It is also 233M parameters -- 24.7x the entire K=4 head stack -- so
duplicating it per head would add ~933M to a 1.54B model.

The final linear of each residual block is zero-initialised, so every head starts
as the identity and therefore starts by reproducing the base model's own
distribution. Training moves them away from that, which is a far better starting
point than random projections of the hidden state.
"""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    """``x + SiLU(Linear(x))``, with the linear zero-initialised."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        # Identity at init: SiLU(0) == 0, so the block returns x unchanged and
        # the head's first predictions equal the base model's.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHeads(nn.Module):
    """K independent heads reading the same final hidden state.

    Independence is what makes verification cheap: all K predictions come from
    one hidden state in parallel, so drafting costs a single small forward rather
    than K sequential ones.
    """

    def __init__(self, hidden_size: int, num_heads: int = 5, num_resblocks: int = 1) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_resblocks = num_resblocks
        self.heads = nn.ModuleList(
            nn.Sequential(*[ResBlock(hidden_size) for _ in range(num_resblocks)])
            for _ in range(num_heads)
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden states to per-head features.

        Args:
            hidden: ``[..., hidden_size]``.

        Returns:
            ``[num_heads, ..., hidden_size]`` -- features, not logits. The caller
            applies the frozen LM head, so this module never holds a reference to
            it and cannot accidentally train it.
        """
        return torch.stack([head(hidden) for head in self.heads], dim=0)


def head_targets(input_ids: torch.Tensor, num_heads: int) -> list[tuple[slice, slice]]:
    """Index slices aligning each head's outputs with the tokens it predicts.

    With the base head predicting ``t+1`` from position ``t``, Medusa head ``k``
    (0-indexed) predicts ``t + k + 2``. So head k's prediction at position i is
    scored against ``input_ids[i + k + 2]``, and the last ``k + 2`` positions of
    the sequence have no target and must be dropped -- getting this off by one
    silently trains every head on the wrong horizon while the loss still falls.
    """
    length = input_ids.shape[-1]
    slices = []
    for k in range(num_heads):
        shift = k + 2
        if length <= shift:
            slices.append((slice(0, 0), slice(0, 0)))
        else:
            slices.append((slice(0, length - shift), slice(shift, length)))
    return slices


@torch.no_grad()
def topk_per_head(
    features: torch.Tensor, lm_head: nn.Module, k: int
) -> torch.Tensor:
    """Top-k token ids per head from head features.

    Args:
        features: ``[num_heads, hidden_size]`` for a single position.
        lm_head: the frozen shared output projection.
        k: candidates per head.

    Returns:
        ``[num_heads, k]`` token ids, ranked.
    """
    logits = lm_head(features)
    return torch.topk(logits, k=k, dim=-1).indices
