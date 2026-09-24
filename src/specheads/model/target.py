"""Loading the frozen target model.

The dtype choice is load-bearing rather than incidental. Qwen2.5-1.5B is
published as bfloat16, and a Kaggle T4 is Turing (sm_75) with no native bf16 --
so on a T4 it runs in fp16, which is a real numerics change from the weights as
released. Losslessness is only meaningful between matched precisions, so the
vanilla baseline and every speculative run must load through this one function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class LoadedTarget:
    """A frozen target model with the metadata the drafters need."""

    model: Any
    tokenizer: Any
    dtype: torch.dtype
    device: torch.device

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def lm_head(self) -> torch.nn.Module:
        """The output projection.

        Qwen2.5 sets `tie_word_embeddings: True`, so this module's weight *is*
        the input embedding matrix. Both drafters read it and neither may train
        it: a gradient arriving here would alter the target's own embeddings and
        silently change the model that losslessness is defined against.
        """
        return self.model.get_output_embeddings()


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[name]


def supports_bfloat16(device: torch.device) -> bool:
    """Whether this device has native bf16. False on Turing, which is the T4."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def load_target(
    name: str = DEFAULT_TARGET,
    dtype: str = "float16",
    device: str | None = None,
    attn_implementation: str = "sdpa",
) -> LoadedTarget:
    """Load and freeze the target model."""
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch_dtype = resolve_dtype(dtype)

    if torch_dtype is torch.bfloat16 and not supports_bfloat16(resolved_device):
        raise ValueError(
            f"bfloat16 requested but {resolved_device} has no native support "
            "(Turing/T4 is sm_75). Use float16 -- and load the baseline the same way."
        )

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch_dtype, attn_implementation=attn_implementation
    )
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)

    return LoadedTarget(
        model=model, tokenizer=tokenizer, dtype=torch_dtype, device=resolved_device
    )


def tiny_target(vocab_size: int = 512, seed: int = 0) -> LoadedTarget:
    """A randomly initialised, CPU-sized Qwen2 for tests.

    Small enough to run the full decode and tree-verification paths on a laptop,
    which is what lets losslessness and mask correctness be tested without
    spending a minute of the T4 budget.
    """
    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(
        DEFAULT_TARGET,
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    # `layer_types` is materialised on the pretrained config as a 28-element
    # list and does NOT shrink when num_hidden_layers is overridden -- DynamicCache
    # reads it and would allocate 28 layer slots for a 2-layer model, leaving 26
    # of them permanently uninitialised.
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model.requires_grad_(False)
    return LoadedTarget(
        model=model, tokenizer=None, dtype=torch.float32, device=torch.device("cpu")
    )
