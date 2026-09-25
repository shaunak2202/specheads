"""Speculative decoding at temperature > 0.

Greedy speculation has a clean correctness criterion: the output must be
*token-identical* to greedy vanilla. Sampling has no such criterion -- two
correct samplers will disagree on almost every token -- so correctness here means
something different and weaker-sounding but equally checkable: the speculative
sampler must draw from the **same distribution** the target would have.

Two verification rules, and the difference between them matters:

* `rejection` -- the standard speculative-sampling rule (Leviathan et al. 2023;
  Chen et al. 2023). Accept draft token ``x`` with probability
  ``min(1, p(x)/q(x))``; on rejection, draw from the normalised residual
  ``(p - q)_+`` and stop. This **provably preserves the target distribution** and
  is implemented for chains only, where the argument holds.
* `typical` -- Medusa's typical acceptance: keep ``x`` when ``p(x)`` clears
  ``min(epsilon, delta * exp(-H(p)))``. It accepts far more aggressively and works
  on trees, but it **does not preserve the target distribution** and is not
  "lossless" in any sense comparable to the greedy case. It is reported as a
  speed/fidelity trade, never as a correctness-preserving mode.

`tests/test_sampling.py` checks the first claim empirically rather than taking it
on faith: it runs both samplers many times against a tiny model and applies a
chi-square test to the resulting token distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

VerificationMode = Literal["rejection", "typical"]


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled probabilities, computed in fp32.

    fp32 regardless of model dtype: an fp16 softmax over a 151936-way vocabulary
    loses enough of the tail that the acceptance ratio ``p(x)/q(x)`` -- which is
    a ratio of two small numbers -- picks up meaningful error.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """Draw one index from a probability vector.

    `torch.multinomial` requires the generator to live on the same device as the
    tensor -- a CPU generator against an MPS or CUDA tensor raises. Rather than
    forcing callers to build a device-specific generator, a mismatched generator
    samples on CPU. That costs one small device-to-host copy per draw and buys
    something worth more here: with a CPU generator the same seed yields the same
    tokens on **any** backend, which is exactly the cross-device reproducibility
    this project keeps needing.
    """
    if generator is not None and generator.device.type != probs.device.type:
        return int(torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item())
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def residual_distribution(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """Normalised ``(p - q)_+``, the distribution to draw from after a rejection.

    This is what makes rejection sampling exact: the mass the draft
    over-allocated is removed, and what remains is precisely the target mass the
    accept step could not account for. Falling back to ``p`` here instead -- an
    easy and tempting simplification -- silently biases the output.
    """
    residual = torch.clamp(target - draft, min=0.0)
    total = residual.sum()
    if total <= 0:
        # Degenerate: q dominates p everywhere. Fall back to the target itself,
        # which is the correct limit as the residual vanishes.
        return target / target.sum()
    return residual / total


@dataclass(frozen=True)
class SamplingAcceptance:
    """Outcome of verifying one drafted chain under sampling."""

    tokens: tuple[int, ...]
    n_accepted: int
    rejected_at: int | None

    @property
    def emitted(self) -> int:
        return len(self.tokens)


def verify_chain_rejection(
    draft_tokens: list[int],
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> SamplingAcceptance:
    """Exact speculative sampling over a drafted chain.

    **Precondition: each draft token must be SAMPLED from its row of
    `draft_probs`, not taken as the argmax.** The distribution-preserving proof
    assumes ``x ~ q``; with a top-k draft the accept probability ``p(x)/q(x)``
    is evaluated at a point that was never drawn from ``q`` and the output is
    biased. This is not a technicality -- the greedy `MedusaDrafter.draft`
    returns top-k tokens and therefore **cannot be reused here unmodified**.
    `tests/test_sampling.py` pins both directions of this.

    Args:
        draft_tokens: ``k`` proposed tokens.
        draft_probs: ``[k, vocab]``, the drafter's distribution at each depth.
        target_probs: ``[k + 1, vocab]``, the target's distribution at the parent
            of each draft position, plus one extra for the bonus token.
        generator: RNG, for reproducibility.

    Returns:
        Accepted tokens followed by exactly one more -- either the residual draw
        after a rejection, or the bonus token if the whole chain was accepted.
    """
    if target_probs.shape[0] != len(draft_tokens) + 1:
        raise ValueError(
            f"expected {len(draft_tokens) + 1} target distributions, "
            f"got {target_probs.shape[0]}"
        )

    accepted: list[int] = []
    for index, token in enumerate(draft_tokens):
        p = float(target_probs[index, token])
        q = float(draft_probs[index, token])

        # q == 0 means the drafter proposed something it assigned no mass to,
        # which should be impossible; accepting unconditionally would break the
        # distribution, so treat it as an automatic rejection.
        ratio = 1.0 if q <= 0 else min(1.0, p / q)
        draw = float(torch.rand(1, generator=generator).item())

        if draw < ratio:
            accepted.append(token)
            continue

        residual = residual_distribution(target_probs[index], draft_probs[index])
        return SamplingAcceptance(
            tokens=tuple(accepted) + (sample_from(residual, generator),),
            n_accepted=len(accepted),
            rejected_at=index,
        )

    bonus = sample_from(target_probs[-1], generator)
    return SamplingAcceptance(
        tokens=tuple(accepted) + (bonus,), n_accepted=len(accepted), rejected_at=None
    )


def typical_threshold(probs: torch.Tensor, epsilon: float, delta: float) -> float:
    """Medusa's typical-acceptance threshold: ``min(epsilon, delta * exp(-H(p)))``.

    The entropy term is what makes it adaptive: where the target is confident
    (low entropy) the bar rises, and where it is uncertain the bar drops and more
    candidates pass.
    """
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-10))).sum())
    return min(epsilon, delta * float(torch.exp(torch.tensor(-entropy))))


def verify_chain_typical(
    draft_tokens: list[int],
    target_probs: torch.Tensor,
    epsilon: float = 0.3,
    delta: float = 0.09,
    generator: torch.Generator | None = None,
) -> SamplingAcceptance:
    """Medusa typical acceptance. Faster, and **not** distribution-preserving.

    Accepts any drafted token the target considers plausible enough, which is a
    deliberate fidelity trade -- the emitted text is not a sample from the target
    distribution, and this mode must never be reported as lossless.
    """
    accepted: list[int] = []
    for index, token in enumerate(draft_tokens):
        probs = target_probs[index]
        if float(probs[token]) > typical_threshold(probs, epsilon, delta):
            accepted.append(token)
            continue
        return SamplingAcceptance(
            tokens=tuple(accepted) + (sample_from(probs, generator),),
            n_accepted=len(accepted),
            rejected_at=index,
        )

    return SamplingAcceptance(
        tokens=tuple(accepted) + (sample_from(target_probs[-1], generator),),
        n_accepted=len(accepted),
        rejected_at=None,
    )


@torch.no_grad()
def speculative_sample_generate(
    model,
    input_ids: torch.Tensor,
    drafter,
    depth: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    mode: VerificationMode = "rejection",
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
):
    """Speculative decoding at temperature > 0, over a drafted chain.

    Chains only. The rejection rule's correctness argument is stated for a linear
    sequence of draft positions; extending it to a branching tree needs the
    multi-round SpecInfer construction, which is not implemented here. Claiming
    tree + rejection without that would be claiming a guarantee we have not
    earned, so it raises instead.

    Returns:
        ``(tokens, stats)`` where stats carries mean accepted length.
    """
    from transformers import DynamicCache

    from .kv_cache import cache_length, prune_to_indices
    from .speculative import DraftContext, build_step_mask, build_step_positions, _forward
    from .tree import TreeSpec
    from .vanilla import GenerationResult

    if mode not in ("rejection", "typical"):
        raise ValueError(f"unknown verification mode {mode!r}")

    device = input_ids.device
    cache = DynamicCache(config=model.config)
    spec = TreeSpec.chain(depth)
    result = GenerationResult()
    accepted_lengths: list[int] = []

    hidden, logits = _forward(
        model, input_ids, torch.arange(input_ids.shape[1], device=device).unsqueeze(0), cache
    )
    result.forward_passes += 1
    root_hidden, root_logits = hidden[-1], logits[-1]
    pending = sample_from(softmax_with_temperature(root_logits, temperature), generator)
    result.tokens.append(pending)
    if eos_token_id is not None and pending == eos_token_id:
        result.finished = True
        return result, {"mean_accepted_length": 0.0, "steps": 0}

    while result.num_tokens < max_new_tokens:
        prefix_len = cache_length(cache)
        draft_tokens, draft_probs = drafter.draft_chain_sampled(
            depth,
            DraftContext(hidden=root_hidden, logits=root_logits, pending_token=pending),
            temperature=temperature,
            generator=generator,
        )

        block = torch.cat(
            [torch.tensor([pending], device=device, dtype=torch.long),
             torch.tensor(draft_tokens, device=device, dtype=torch.long)]
        ).unsqueeze(0)

        hidden, logits = _forward(
            model,
            block,
            build_step_positions(spec, prefix_len, device=device),
            cache,
            attention_mask=build_step_mask(spec, prefix_len, device=device),
        )
        result.forward_passes += 1

        target_probs = torch.stack(
            [softmax_with_temperature(logits[i], temperature) for i in range(depth + 1)]
        )
        if mode == "rejection":
            acceptance = verify_chain_rejection(
                draft_tokens, draft_probs.to(target_probs.device), target_probs, generator
            )
        else:
            acceptance = verify_chain_typical(draft_tokens, target_probs, generator=generator)

        accepted_lengths.append(acceptance.n_accepted)
        keep = list(range(prefix_len + 1)) + [
            prefix_len + 1 + i for i in range(acceptance.n_accepted)
        ]
        prune_to_indices(cache, keep)

        last = acceptance.n_accepted  # index into the fed block
        root_hidden, root_logits = hidden[last], logits[last]

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
        pending = acceptance.tokens[-1]

    del result.tokens[max_new_tokens:]
    steps = len(accepted_lengths)
    return result, {
        "mean_accepted_length": sum(accepted_lengths) / steps if steps else 0.0,
        "steps": steps,
        "mode": mode,
        "temperature": temperature,
    }
