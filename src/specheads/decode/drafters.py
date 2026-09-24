"""Reference drafters used to test the decode machinery before anything is trained.

Neither of these is a contribution; they exist so that tree construction, the
attention mask, verification and cache pruning can be tested on CPU against a
tiny model, with no GPU and no trained weights.

They bracket the acceptance range on purpose:

* `RandomDrafter` has essentially everything rejected, so the loop degrades to
  one token per forward. It proves losslessness survives a useless drafter.
* `OracleDrafter` proposes the true greedy continuation, so every step accepts to
  full depth. This is the one that actually exercises cache pruning -- with a
  random drafter the accepted path is almost always empty and the interesting
  branch of `prune_to_indices` never runs.

The trained Medusa and EAGLE drafters land in `specheads.model` at Phases 3 and 5.
"""

from __future__ import annotations

import torch

from .speculative import DraftContext
from .tree import TreeSpec


class RandomDrafter:
    """Uniformly random tokens, distinct among siblings.

    Sibling distinctness is required by `accept_path`, not cosmetic: two siblings
    carrying the same token would make the accepted path ambiguous.
    """

    def __init__(self, vocab_size: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self._generator = torch.Generator().manual_seed(seed)

    def draft(self, spec: TreeSpec, context: DraftContext) -> torch.Tensor:
        del context
        tokens = torch.zeros(spec.size, dtype=torch.long)
        groups: dict[int, list[int]] = {}
        for node, parent in enumerate(spec.parents):
            groups.setdefault(parent, []).append(node)

        for siblings in groups.values():
            drawn = torch.randperm(self.vocab_size, generator=self._generator)[: len(siblings)]
            for node, token in zip(siblings, drawn):
                tokens[node] = token
        return tokens


class OracleDrafter:
    """Proposes the known-correct continuation along the rank-0 path.

    Constructed from the vanilla greedy output for the same prompt. At each step
    it looks up where generation has reached and fills the all-rank-0 chain with
    the true next tokens; every other node gets a deliberately wrong token so
    that only the intended path can be accepted.

    This drives acceptance to full tree depth, which is the only way the
    rejected-branch pruning gets meaningfully exercised.
    """

    def __init__(self, expected_tokens: list[int], vocab_size: int) -> None:
        self.expected = list(expected_tokens)
        self.vocab_size = vocab_size
        self.position = 0  # index in `expected` of the next token to be emitted

    def draft(self, spec: TreeSpec, context: DraftContext) -> torch.Tensor:
        del context
        # A node on the rank-0 chain at depth d predicts `position + d`.
        tokens = torch.zeros(spec.size, dtype=torch.long)
        for node, path in enumerate(spec.ordered):
            depth = len(path)
            if all(rank == 0 for rank in path):
                index = self.position + depth
                tokens[node] = (
                    self.expected[index] if index < len(self.expected) else self.vocab_size - 1
                )
            else:
                # Off-chain nodes must never accidentally match: offset by the
                # rank so siblings stay distinct from each other and from the
                # true token.
                base = self.position + depth
                truth = self.expected[base] if base < len(self.expected) else 0
                tokens[node] = (truth + 1 + path[-1]) % self.vocab_size
        return tokens

    def advance(self, emitted: int) -> None:
        """Tell the drafter how many tokens the last step actually emitted."""
        self.position += emitted
