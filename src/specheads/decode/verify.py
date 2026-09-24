"""Accepting a path out of a verified candidate tree.

Given the target's logits over every tree node, walk down from the root taking
the child whose token matches the target's own greedy choice, and stop at the
first depth where no child matches. The accepted path plus one bonus token --
the target's greedy continuation from the last accepted node -- is what the step
emits.

The bonus token is what makes speculative decoding a strict win rather than a
gamble: even when the drafter proposes nothing usable, the verification forward
has already produced the target's next token, so a fully-rejected step still
advances by one and costs the same as vanilla.

Only one child can ever match at a given depth, because a parent has exactly one
greedy argmax and the tree's candidates at a depth are distinct tokens. The walk
is therefore unambiguous, and `accept_path` asserts it rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tree import TreeSpec


@dataclass(frozen=True)
class Acceptance:
    """Outcome of verifying one tree.

    Attributes:
        path: node indices accepted, in order; empty when nothing matched.
        tokens: the accepted tokens followed by the bonus token. Always at least
            length 1.
        bonus_token: the target's greedy continuation from the last accepted node.
        accepted_length: number of drafted tokens accepted, excluding the bonus.
    """

    path: tuple[int, ...]
    tokens: tuple[int, ...]
    bonus_token: int

    @property
    def accepted_length(self) -> int:
        return len(self.path)

    @property
    def emitted(self) -> int:
        """Tokens this step produced -- accepted drafts plus the bonus."""
        return len(self.tokens)


def accept_path(
    spec: TreeSpec,
    candidate_tokens: torch.Tensor,
    node_logits: torch.Tensor,
    root_logits: torch.Tensor,
) -> Acceptance:
    """Walk the tree, accepting while candidates match the target's greedy choice.

    Args:
        spec: the tree that was drafted.
        candidate_tokens: ``[size]`` token id proposed at each node, in
            `spec.ordered` order.
        node_logits: ``[size, vocab]`` target logits *at* each node -- that is,
            the distribution over the token that would follow it.
        root_logits: ``[vocab]`` target logits at the last already-accepted
            token, which decide the depth-1 nodes. These come free from the
            previous step's forward pass.

    Returns:
        The accepted path, its tokens, and the bonus token.
    """
    if candidate_tokens.shape[0] != spec.size:
        raise ValueError(
            f"expected {spec.size} candidate tokens, got {candidate_tokens.shape[0]}"
        )
    if node_logits.shape[0] != spec.size:
        raise ValueError(f"expected {spec.size} rows of logits, got {node_logits.shape[0]}")

    tokens = [int(t) for t in candidate_tokens.tolist()]
    path: list[int] = []

    # The token the target would emit next, from wherever the walk currently is.
    expected = int(torch.argmax(root_logits).item())
    candidates = spec.root_children

    while True:
        matches = [node for node in candidates if tokens[node] == expected]
        if not matches:
            break
        if len(matches) > 1:
            raise ValueError(
                f"tree is malformed: nodes {matches} at the same depth share token "
                f"{expected}; candidates within a depth must be distinct"
            )
        node = matches[0]
        path.append(node)
        expected = int(torch.argmax(node_logits[node]).item())
        candidates = spec.children[node]
        if not candidates:
            break

    # `expected` is now the target's greedy continuation from the last accepted
    # node (or from the root, if nothing was accepted) -- the bonus token.
    accepted_tokens = tuple(tokens[node] for node in path) + (expected,)
    return Acceptance(path=tuple(path), tokens=accepted_tokens, bonus_token=expected)


def cache_indices_after_acceptance(
    prefix_len: int, spec: TreeSpec, path: tuple[int, ...]
) -> list[int]:
    """Absolute cache positions to keep after accepting `path`.

    The verification forward appended every node to the cache. Only the prefix
    and the accepted path may survive; everything else is a branch the model
    never emitted and must not be attendable on the next step.
    """
    del spec  # kept in the signature so the contract reads explicitly
    return list(range(prefix_len)) + [prefix_len + node for node in path]
