"""Candidate trees and the attention mask that verifies them in one pass.

A draft tree is described by a `TreeSpec`: a list of paths, each path a tuple of
per-depth ranks. ``(0,)`` is "the top-1 candidate from the depth-1 drafter";
``(0, 2)`` is "top-1 at depth 1, then third-choice at depth 2". The empty tuple
is the implicit root -- the last already-accepted token -- and is never listed.

Why paths rather than a branching factor: the useful trees are not uniform. The
top-1 branch deserves more children than the fourth, because acceptance falls off
sharply with rank, and a spec of explicit paths says exactly that without a
special case.

The mask is the whole point. All candidates go through the target in a single
forward, so each one must see the prefix and its own ancestors and nothing else.
Get it wrong in the permissive direction and a candidate attends to a sibling
branch that was never emitted -- which still produces fluent text, still passes a
smoke test, and silently breaks losslessness.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

Path = tuple[int, ...]


@dataclass(frozen=True)
class TreeSpec:
    """A static draft-tree shape.

    Attributes:
        paths: every node as a path of per-depth ranks, excluding the root.
    """

    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("tree must contain at least one node")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("duplicate paths in tree spec")
        if any(len(p) == 0 for p in self.paths):
            raise ValueError("the empty path is the implicit root and must not be listed")
        if any(r < 0 for p in self.paths for r in p):
            raise ValueError("ranks must be non-negative")

        # Prefix closure: a node whose parent is missing can never be verified,
        # because verification walks down from the root one accepted edge at a
        # time and would have no way to reach it.
        known = set(self.paths)
        for path in self.paths:
            if len(path) > 1 and path[:-1] not in known:
                raise ValueError(f"node {path} is missing its parent {path[:-1]}")

    @staticmethod
    def chain(depth: int) -> "TreeSpec":
        """The degenerate tree: top-1 at every depth, no branching.

        Phase 4 starts here. A chain exercises the mask, the position ids and the
        cache pruning while keeping the tree itself trivially checkable.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")
        return TreeSpec(tuple(tuple(0 for _ in range(d + 1)) for d in range(depth)))

    @staticmethod
    def from_widths(widths: tuple[int, ...]) -> "TreeSpec":
        """A tree that gives rank-0 nodes `widths[d]` children at each depth.

        Only the top-1 path branches; deeper ranks continue as chains. This is
        the cheap, standard shape and a reasonable default for the Phase 6 sweep.
        """
        if not widths:
            raise ValueError("widths must be non-empty")
        paths: list[Path] = []
        frontier: list[Path] = [()]
        for width in widths:
            next_frontier: list[Path] = []
            for parent in frontier:
                for rank in range(width):
                    node = parent + (rank,)
                    paths.append(node)
                    if rank == 0:
                        next_frontier.append(node)
            frontier = next_frontier
        return TreeSpec(tuple(paths))

    @cached_property
    def ordered(self) -> tuple[Path, ...]:
        """Nodes sorted by (depth, path) so a parent always precedes its children.

        The forward pass and the mask are both indexed by this order, and the
        guarantee that parents come first is what lets the mask be built with a
        single pass instead of a topological sort.
        """
        return tuple(sorted(self.paths, key=lambda p: (len(p), p)))

    @cached_property
    def size(self) -> int:
        return len(self.ordered)

    @cached_property
    def depth(self) -> int:
        return max(len(p) for p in self.ordered)

    @cached_property
    def depths(self) -> tuple[int, ...]:
        """1-based depth of each node, in `ordered` order."""
        return tuple(len(p) for p in self.ordered)

    @cached_property
    def parents(self) -> tuple[int, ...]:
        """Index of each node's parent in `ordered`, or -1 when the parent is the root."""
        index = {path: i for i, path in enumerate(self.ordered)}
        return tuple(-1 if len(p) == 1 else index[p[:-1]] for p in self.ordered)

    @cached_property
    def ancestors(self) -> tuple[tuple[int, ...], ...]:
        """Indices of each node's proper ancestors (excluding the root)."""
        result: list[tuple[int, ...]] = []
        for parent in self.parents:
            if parent < 0:
                result.append(())
            else:
                result.append(result[parent] + (parent,))
        return tuple(result)

    @cached_property
    def children(self) -> tuple[tuple[int, ...], ...]:
        """Children of the root (as index -1) plus children of each node."""
        buckets: list[list[int]] = [[] for _ in range(self.size)]
        roots: list[int] = []
        for i, parent in enumerate(self.parents):
            (roots if parent < 0 else buckets[parent]).append(i)
        self.__dict__["root_children"] = tuple(roots)
        return tuple(tuple(b) for b in buckets)

    @cached_property
    def root_children(self) -> tuple[int, ...]:
        """Depth-1 nodes, whose parent is the last already-accepted token."""
        _ = self.children  # populates root_children as a side effect
        return self.__dict__["root_children"]

    def ranks_at(self, node: int) -> int:
        """The rank this node took from its drafter (its last path element)."""
        return self.ordered[node][-1]

    def position_ids(self, prefix_len: int, device=None) -> torch.Tensor:
        """Absolute position ids for the candidates, shape ``[1, size]``.

        A node at depth d sits at ``prefix_len + d - 1``: depth 1 occupies the
        slot immediately after the prefix. Siblings share a position, which is
        correct -- they are competing hypotheses for the same slot, and RoPE must
        treat them identically or the branch a candidate sits in would change its
        own embedding.
        """
        offsets = torch.tensor(self.depths, dtype=torch.long, device=device) - 1
        return (offsets + prefix_len).unsqueeze(0)

    def attention_mask(
        self, prefix_len: int, device=None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Tree attention mask of shape ``[1, 1, size, prefix_len + size]``.

        Every node attends to the whole prefix, to its proper ancestors, and to
        itself -- never to a sibling or to any other branch.

        Returns a boolean mask (``True`` = attend) by default, which SDPA takes
        directly. Pass a float `dtype` to get the additive ``0 / -inf`` form the
        eager attention path expects.
        """
        size = self.size
        total = prefix_len + size
        mask = torch.zeros((size, total), dtype=torch.bool, device=device)

        if prefix_len:
            mask[:, :prefix_len] = True
        for i, ancestors in enumerate(self.ancestors):
            mask[i, prefix_len + i] = True  # self
            for ancestor in ancestors:
                mask[i, prefix_len + ancestor] = True

        mask = mask.unsqueeze(0).unsqueeze(0)
        if dtype is None or dtype == torch.bool:
            return mask
        # Additive form: 0 where attending, -inf where masked.
        return torch.where(
            mask,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
        )

    def path_to(self, node: int) -> tuple[int, ...]:
        """Node indices from the first depth-1 ancestor down to `node` inclusive."""
        return self.ancestors[node] + (node,)
