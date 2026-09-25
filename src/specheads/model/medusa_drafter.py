"""Turning trained Medusa heads into a `Drafter` for the speculative loop.

A tree node at depth *d* with rank *r* takes the *r*-th ranked token from head
``d - 1``. Because the heads are independent predictions from one hidden state,
every node in the tree is available from a single small forward -- which is the
property that makes Medusa cheap relative to an autoregressive drafter.

Siblings are distinct by construction: they share a depth, so they read the same
head, and they differ in rank, so they take different entries of the same top-k.
That is exactly the invariant `accept_path` requires.
"""

from __future__ import annotations

import torch
from torch import nn

from ..decode.speculative import DraftContext
from ..decode.tree import TreeSpec
from .medusa_heads import MedusaHeads


class MedusaDrafter:
    """Drafts a candidate tree from trained Medusa heads."""

    def __init__(self, heads: MedusaHeads, lm_head: nn.Module) -> None:
        self.heads = heads
        self.lm_head = lm_head

    @torch.no_grad()
    def draft(self, spec: TreeSpec, context: DraftContext) -> torch.Tensor:
        if spec.depth > self.heads.num_heads:
            raise ValueError(
                f"tree depth {spec.depth} exceeds {self.heads.num_heads} trained heads"
            )

        max_rank = max((path[-1] for path in spec.ordered), default=0)
        # Heads hold fp32 master weights; the frozen LM head is the target's
        # dtype. Cast at the boundary rather than forcing either side to match.
        head_dtype = next(self.heads.parameters()).dtype
        weight_dtype = next(self.lm_head.parameters()).dtype
        features = self.heads(context.hidden.to(head_dtype).unsqueeze(0))  # [K, 1, hidden]
        logits = self.lm_head(features.squeeze(1).to(weight_dtype))        # [K, vocab]
        ranked = torch.topk(logits.float(), k=max_rank + 1, dim=-1).indices

        tokens = torch.zeros(spec.size, dtype=torch.long)
        for node, path in enumerate(spec.ordered):
            depth, rank = len(path), path[-1]
            tokens[node] = ranked[depth - 1, rank]
        return tokens

    @torch.no_grad()
    def draft_chain_sampled(
        self,
        depth: int,
        context: "DraftContext",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], torch.Tensor]:
        """Draft a chain by **sampling** each head, returning tokens and their q.

        Separate from `draft` on purpose. `draft` takes top-k, which is right for
        greedy verification and *wrong* for rejection sampling: that algorithm
        preserves the target distribution only when the draft is drawn from q.
        Reusing the greedy path here would bias the output while still looking
        plausible -- see `tests/test_sampling.py`.

        Returns:
            ``(tokens, probs)`` where ``probs`` is ``[depth, vocab]``, head k's
            full distribution at temperature.
        """
        from ..decode.sampling import sample_from, softmax_with_temperature

        if depth > self.heads.num_heads:
            raise ValueError(f"depth {depth} exceeds {self.heads.num_heads} trained heads")

        head_dtype = next(self.heads.parameters()).dtype
        weight_dtype = next(self.lm_head.parameters()).dtype
        features = self.heads(context.hidden.to(head_dtype).unsqueeze(0))
        logits = self.lm_head(features.squeeze(1).to(weight_dtype))

        probs = torch.stack(
            [softmax_with_temperature(logits[k], temperature) for k in range(depth)]
        )
        tokens = [sample_from(probs[k], generator) for k in range(depth)]
        return tokens, probs
