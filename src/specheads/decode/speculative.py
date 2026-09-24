"""The speculative decoding step: draft, verify in one pass, prune, repeat.

**One target forward per step.** The cache holds every emitted token except the
most recent, which is carried as `pending_token`; each step feeds
``[pending_token] + candidates`` through the target exactly once. The root row
produces the logits that decide the depth-1 candidates, so verification needs no
extra pass.

That constraint is what forces the drafter's interface. Candidates must exist
*before* the forward runs, so they cannot be conditioned on the root's own
logits -- they are drafted from the hidden state of the position that produced
the root, which the previous step already computed. This is exactly how Medusa
works: heads sitting on hidden state at position ``p`` predict tokens ``p+2``
onwards, while the base LM head predicts ``p+1``. Getting this wrong by running
the root separately costs a second forward per step and erases the speedup.

After verification the cache is pruned to prefix + root + accepted path. This is
the step that has to be right: a rejected branch left in the cache lets the next
step attend to a token the model never emitted, which yields perfectly fluent
output that is silently not what greedy decoding would have produced.

Losslessness does not depend on draft quality. A drafter proposing random tokens
has everything rejected and degrades to one token per forward -- the same tokens
vanilla emits, with no speedup. That is what lets the correctness tests run on
CPU with a random drafter, before any head is trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache

from .kv_cache import cache_length, prune_to_indices
from .tree import TreeSpec
from .vanilla import GenerationResult
from .verify import accept_path


@dataclass(frozen=True)
class DraftContext:
    """What a drafter sees before the verification forward runs.

    Attributes:
        hidden: ``[hidden_size]`` final hidden state at the position that
            produced `pending_token`. Medusa heads and the EAGLE drafter both
            read this.
        logits: ``[vocab]`` target logits at that same position -- the
            distribution `pending_token` was drawn from.
        pending_token: the token the tree's root will carry.
    """

    hidden: torch.Tensor
    logits: torch.Tensor
    pending_token: int


class Drafter(Protocol):
    """Proposes a token for every node of a tree.

    Implementations must guarantee that **siblings carry distinct tokens**.
    Verification descends by matching the target's single greedy argmax, so two
    siblings proposing the same token would make the accepted path ambiguous;
    `accept_path` raises rather than silently picking one.
    """

    def draft(self, spec: TreeSpec, context: DraftContext) -> torch.Tensor:
        """Return ``[spec.size]`` token ids, in `spec.ordered` order."""

    def advance(self, emitted: int) -> None:  # optional
        """Told how many tokens the step actually emitted, if implemented.

        Stateful drafters need this. The EAGLE drafter runs autoregressively and
        keeps its own KV cache, so it has to know how far generation moved to
        stay aligned with the target after a partial rejection.
        """


@dataclass
class SpeculativeStats:
    """Per-step acceptance, for the mean-accepted-length and per-depth metrics."""

    accepted_lengths: list[int] = field(default_factory=list)
    emitted_per_step: list[int] = field(default_factory=list)
    depth_hits: dict[int, int] = field(default_factory=dict)
    depth_attempts: dict[int, int] = field(default_factory=dict)

    def record(self, spec: TreeSpec, path: tuple[int, ...], emitted: int) -> None:
        self.accepted_lengths.append(len(path))
        self.emitted_per_step.append(emitted)
        for depth in range(1, spec.depth + 1):
            self.depth_attempts[depth] = self.depth_attempts.get(depth, 0) + 1
            if len(path) >= depth:
                self.depth_hits[depth] = self.depth_hits.get(depth, 0) + 1

    @property
    def mean_accepted_length(self) -> float:
        """Drafted tokens accepted per step, excluding the bonus token."""
        if not self.accepted_lengths:
            return 0.0
        return sum(self.accepted_lengths) / len(self.accepted_lengths)

    @property
    def mean_emitted_per_step(self) -> float:
        """Tokens emitted per target forward pass -- the speedup ceiling."""
        if not self.emitted_per_step:
            return 0.0
        return sum(self.emitted_per_step) / len(self.emitted_per_step)

    def acceptance_by_depth(self) -> dict[int, float]:
        """Fraction of steps that accepted at least to each depth."""
        return {
            depth: self.depth_hits.get(depth, 0) / attempts
            for depth, attempts in sorted(self.depth_attempts.items())
            if attempts
        }


def build_step_mask(
    spec: TreeSpec, prefix_len: int, device=None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Mask for ``[root] + candidates``: ``[1, 1, 1 + size, prefix_len + 1 + size]``.

    The root attends to the prefix and itself. Candidates take prefix+root as
    their prefix, since every candidate descends from the root.

    Boolean (``True`` = attend) by default, which SDPA accepts directly. Passing
    a float `dtype` gives the additive ``0 / -inf`` form the eager path wants.
    """
    total = prefix_len + 1 + spec.size
    candidate_mask = spec.attention_mask(prefix_len + 1, device=device, dtype=torch.bool)

    root_row = torch.zeros((1, 1, 1, total), dtype=torch.bool, device=device)
    root_row[..., : prefix_len + 1] = True

    mask = torch.cat([root_row, candidate_mask], dim=2)
    if dtype is None or dtype == torch.bool:
        return mask
    return torch.where(
        mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
    )


def build_step_positions(spec: TreeSpec, prefix_len: int, device=None) -> torch.Tensor:
    """Position ids for ``[root] + candidates``: ``[1, 1 + size]``."""
    root = torch.tensor([[prefix_len]], dtype=torch.long, device=device)
    return torch.cat([root, spec.position_ids(prefix_len + 1, device=device)], dim=1)


def _forward(model, input_ids, position_ids, cache, attention_mask=None):
    """Run the base model and project to logits, returning both.

    Calls `model.model` rather than the causal-LM wrapper because the drafters
    need the hidden states, and asking the wrapper for them returns every layer's
    output when only the last is wanted.
    """
    kwargs = {}
    if attention_mask is not None:
        # A dict here bypasses transformers' causal-mask construction entirely
        # and uses ours verbatim (transformers 5.x, Qwen2Model.forward).
        kwargs["attention_mask"] = {"full_attention": attention_mask}
    out = model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        **kwargs,
    )
    hidden = out.last_hidden_state[0]
    return hidden, model.lm_head(hidden)


@torch.no_grad()
def speculative_generate(
    model,
    input_ids: torch.Tensor,
    drafter: Drafter,
    spec: TreeSpec,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    cache: Cache | None = None,
) -> tuple[GenerationResult, SpeculativeStats]:
    """Greedy speculative decoding, token-identical to vanilla greedy.

    Args:
        model: the frozen target.
        input_ids: ``[1, prompt_len]``, batch size 1.
        drafter: proposes candidates; its quality affects speed, never output.
        spec: the static candidate tree.
        max_new_tokens: cap on emitted tokens. A step can accept several at once,
            so the result is truncated to the cap.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids of shape [1, seq], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one token")

    device = input_ids.device
    if cache is None:
        cache = DynamicCache(config=model.config)

    result = GenerationResult()
    stats = SpeculativeStats()

    # Prefill the whole prompt. Its last position gives both the first emitted
    # token and the hidden state the first draft is conditioned on.
    hidden, logits = _forward(
        model,
        input_ids,
        torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
        cache,
    )
    result.forward_passes += 1

    root_hidden = hidden[-1]
    root_logits = logits[-1]
    pending_token = int(torch.argmax(root_logits).item())
    result.tokens.append(pending_token)

    if eos_token_id is not None and pending_token == eos_token_id:
        result.finished = True
        return result, stats

    while result.num_tokens < max_new_tokens:
        prefix_len = cache_length(cache)

        candidates = drafter.draft(
            spec, DraftContext(hidden=root_hidden, logits=root_logits, pending_token=pending_token)
        ).to(device)
        if candidates.shape != (spec.size,):
            raise ValueError(
                f"drafter returned {tuple(candidates.shape)}, expected ({spec.size},)"
            )

        block = torch.cat(
            [torch.tensor([pending_token], device=device, dtype=torch.long), candidates]
        ).unsqueeze(0)

        hidden, logits = _forward(
            model,
            block,
            build_step_positions(spec, prefix_len, device=device),
            cache,
            attention_mask=build_step_mask(spec, prefix_len, device=device),
        )
        result.forward_passes += 1

        acceptance = accept_path(spec, candidates, logits[1:], logits[0])
        stats.record(spec, acceptance.path, acceptance.emitted)

        advance = getattr(drafter, "advance", None)
        if callable(advance):
            advance(acceptance.emitted)

        # Keep prefix + root + accepted path; drop every rejected branch.
        root_index = prefix_len
        keep = list(range(prefix_len + 1)) + [
            root_index + 1 + node for node in acceptance.path
        ]
        prune_to_indices(cache, keep)

        # The next draft is conditioned on the last surviving position, whose
        # base-head argmax is exactly the bonus token now becoming the root.
        last_block_index = 0 if not acceptance.path else 1 + acceptance.path[-1]
        root_hidden = hidden[last_block_index]
        root_logits = logits[last_block_index]

        hit_eos = False
        for token in acceptance.tokens:
            if result.num_tokens >= max_new_tokens:
                break
            result.tokens.append(token)
            if eos_token_id is not None and token == eos_token_id:
                hit_eos = True
                result.finished = True
                break

        if hit_eos:
            break
        pending_token = acceptance.bonus_token

    del result.tokens[max_new_tokens:]
    return result, stats
