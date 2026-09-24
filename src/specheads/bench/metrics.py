"""Metric aggregation, including which metrics survive a change of hardware.

The distinction matters enough to encode it rather than leave it to a README
sentence. Wall-clock throughput and peak memory describe a machine; acceptance
length, per-head accuracy and forwards-per-token describe a drafter. A result
file carries both, and `HARDWARE_DEPENDENT` marks the ones that may not be
compared across devices -- which is what keeps a laptop number from being read
later as if it came from the T4.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable, Sequence

#: Metrics that describe the machine, not the method.
HARDWARE_DEPENDENT = frozenset(
    {
        "median_tokens_per_second",
        "p90_tokens_per_second",
        "speedup_vs_vanilla",
        "peak_memory_gb",
        "seconds",
    }
)

#: Metrics that are properties of the model and drafter and transfer across devices.
HARDWARE_INDEPENDENT = frozenset(
    {
        "mean_accepted_length",
        "mean_tokens_per_forward",
        "acceptance_by_depth",
        "head_top1_accuracy",
        "head_top5_accuracy",
        "lossless",
        "forward_passes",
    }
)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over prompts.

    Prompt-level variation dominates here -- a long arithmetic chain and a
    one-line chat reply have very different acceptance -- so the interval is
    resampled over prompts rather than assumed normal.
    """
    values = [float(v) for v in values]
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choices(values, k=n)))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, int(math.floor(tail * samples)) - 1)]
    high = means[min(samples - 1, int(math.ceil((1.0 - tail) * samples)) - 1)]
    return (low, high)


def mean_or_nan(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return statistics.mean(values) if values else float("nan")


def speedup(method_tps: float, baseline_tps: float) -> float:
    """Throughput ratio against the baseline from the *same session*.

    Shared GPUs drift, so a baseline measured in another session is not a
    baseline. The harness re-runs vanilla alongside every method for this reason.
    """
    if baseline_tps <= 0:
        return float("nan")
    return method_tps / baseline_tps


def summarize_acceptance(accepted_lengths: Sequence[int]) -> dict[str, float]:
    """Mean accepted length with a bootstrap interval."""
    if not accepted_lengths:
        return {"mean_accepted_length": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = bootstrap_ci(accepted_lengths)
    return {
        "mean_accepted_length": statistics.mean(accepted_lengths),
        "ci_low": low,
        "ci_high": high,
    }
