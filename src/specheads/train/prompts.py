"""Prompt sets: training mixtures, fixed eval sets, and decontamination.

Two invariants this module exists to enforce.

**Train and eval must be disjoint.** Code instruction datasets are known to carry
HumanEval problems, and a drafter that has memorised an eval problem posts high
acceptance for reasons that have nothing to do with speculative decoding -- which
would corrupt the domain-shift result specifically, since code is one of the
shifted domains. `decontaminate` drops training prompts overlapping any eval
prompt, and the number dropped is recorded rather than discarded.

**The chat-only and mixed sets must match on token count, not prompt count.**
Otherwise the domain comparison confounds mixture with data volume, and a
difference in acceptance could just mean one drafter saw more tokens.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Dataset ids and licenses, verified against the HF API (docs/plan.md section 2).
SOURCES: dict[str, dict[str, Any]] = {
    "chat": {"id": "HuggingFaceH4/ultrachat_200k", "license": "mit", "split": "train_sft"},
    "code": {"id": "glaiveai/glaive-code-assistant", "license": "apache-2.0", "split": "train"},
    "math": {"id": "openai/gsm8k", "license": "mit", "split": "train", "config": "main"},
}

EVAL_SOURCES: dict[str, dict[str, Any]] = {
    "chat": {"id": "HuggingFaceH4/mt_bench_prompts", "license": "apache-2.0", "split": "train"},
    "code": {"id": "openai/openai_humaneval", "license": "mit", "split": "test"},
    "math": {"id": "openai/gsm8k", "license": "mit", "split": "test", "config": "main"},
}

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class PromptSet:
    """A named list of prompts with provenance."""

    domain: str
    source: str
    license: str
    prompts: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.prompts)

    @property
    def content_hash(self) -> str:
        """SHA-256 over the materialised prompts, so a rerun can prove identity."""
        digest = hashlib.sha256()
        for prompt in self.prompts:
            digest.update(prompt.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()


def normalise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, for overlap detection only."""
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def decontaminate(
    training: list[str], evaluation: Iterable[str], n: int = 13
) -> tuple[list[str], int]:
    """Drop training prompts sharing an n-gram with any eval prompt.

    13-gram overlap on normalised text: long enough that ordinary phrasing does
    not collide, short enough to catch a reworded eval problem.

    Returns the surviving prompts and how many were dropped.
    """
    banned: set[tuple[str, ...]] = set()
    for prompt in evaluation:
        banned |= ngrams(normalise(prompt), n)

    if not banned:
        return list(training), 0

    kept = [p for p in training if not (ngrams(normalise(p), n) & banned)]
    return kept, len(training) - len(kept)


def balance_by_tokens(
    sets: dict[str, list[str]], token_counts: dict[str, list[int]], budget: int
) -> dict[str, list[str]]:
    """Take prompts from each domain until each reaches `budget` tokens.

    Equalising on tokens rather than prompts is what keeps the domain-shift
    comparison honest: code answers run far longer than chat answers, so equal
    prompt counts would hand the code-containing mixture substantially more
    training signal.
    """
    out: dict[str, list[str]] = {}
    for domain, prompts in sets.items():
        counts = token_counts[domain]
        running = 0
        taken: list[str] = []
        for prompt, count in zip(prompts, counts):
            if running >= budget:
                break
            taken.append(prompt)
            running += count
        out[domain] = taken
    return out


def extract_prompt(domain: str, row: dict[str, Any]) -> str | None:
    """Pull the user-facing prompt out of a dataset row.

    Each source stores it differently, and getting this wrong silently trains on
    the wrong text, so the mapping is explicit per dataset rather than a
    best-effort field search.
    """
    if domain == "chat":
        # UltraChat: `messages` alternating user/assistant; MT-Bench: `prompt` list.
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            for message in messages:
                if message.get("role") == "user":
                    return message.get("content")
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt:
            return prompt[0]
        if isinstance(prompt, str):
            return prompt
        return None
    if domain == "code":
        # Glaive: `question`/`answer`. HumanEval: `prompt` is the function stub.
        return row.get("question") or row.get("prompt")
    if domain == "math":
        return row.get("question")
    raise ValueError(f"unknown domain {domain!r}")
