"""Our own decode loop with explicit KV cache management.

This is the reference every speculative result is measured against, so it is
written to be obviously correct rather than clever: prefill once, then one token
per forward, carrying the cache by hand. Phase 1's gate is that greedy output
here is token-identical to `model.generate`.

The cache is threaded explicitly rather than left to `generate` because Phase 4
needs to prune it mid-stream, and a loop that hides the cache cannot do that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache

from .kv_cache import cache_length


@dataclass
class GenerationResult:
    """Tokens produced by a decode loop, plus what the next step would need.

    Attributes:
        tokens: generated token ids, excluding the prompt.
        forward_passes: target forward passes used. For vanilla this equals
            `len(tokens)`; speculative methods drive it below that, and the ratio
            is the mean accepted length.
        finished: whether generation stopped on an EOS rather than the length cap.
    """

    tokens: list[int] = field(default_factory=list)
    forward_passes: int = 0
    finished: bool = False

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


def _next_token_greedy(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def _next_token_sampled(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    """Temperature + nucleus sampling over the final-position logits."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling; use greedy instead")

    # float32 for the softmax regardless of model dtype: fp16 softmax over a
    # 151936-way vocabulary loses enough precision in the tail to change which
    # tokens survive a top-p cut.
    scaled = (logits.float() / temperature).softmax(dim=-1)

    if 0.0 < top_p < 1.0:
        ordered, indices = torch.sort(scaled, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # Keep the first token that crosses the threshold, so top_p can never
        # select an empty set.
        keep = cumulative - ordered <= top_p
        ordered = torch.where(keep, ordered, torch.zeros_like(ordered))
        ordered = ordered / ordered.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(ordered, num_samples=1, generator=generator)
        return int(indices.gather(-1, choice).item())

    return int(torch.multinomial(scaled, num_samples=1, generator=generator).item())


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    greedy: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
    cache: Cache | None = None,
) -> tuple[GenerationResult, Cache, torch.Tensor]:
    """Decode up to `max_new_tokens` from `input_ids`.

    Args:
        model: a causal LM.
        input_ids: ``[1, prompt_len]``. Batch size 1 throughout -- the benchmark
            protocol fixes it, and speculative decoding at batch 1 is the regime
            this project is about.
        cache: an existing cache to continue from; a fresh one is made if None.

    Returns:
        The result, the cache, and the final-position logits. The logits are
        returned because speculative decoding needs them to verify its first
        depth without paying for another forward.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids of shape [1, seq], got {tuple(input_ids.shape)}")

    if cache is None:
        cache = DynamicCache(config=model.config)

    result = GenerationResult()
    device = input_ids.device
    prefix = cache_length(cache)

    # Prefill. `logits_to_keep=1` avoids materialising logits for the whole
    # prompt, which at this vocabulary is the single largest allocation in the
    # forward pass.
    position_ids = torch.arange(prefix, prefix + input_ids.shape[1], device=device).unsqueeze(0)
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    logits = outputs.logits[0, -1]
    result.forward_passes += 1

    for _ in range(max_new_tokens):
        token = (
            _next_token_greedy(logits)
            if greedy
            else _next_token_sampled(logits, temperature, top_p, generator)
        )
        result.tokens.append(token)

        if eos_token_id is not None and token == eos_token_id:
            result.finished = True
            break
        if result.num_tokens >= max_new_tokens:
            break

        position = cache_length(cache)
        outputs = model(
            input_ids=torch.tensor([[token]], device=device),
            position_ids=torch.tensor([[position]], device=device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[0, -1]
        result.forward_passes += 1

    return result, cache, logits
