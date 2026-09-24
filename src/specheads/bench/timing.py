"""Wall-clock timing that is honest about what it measured.

Two things make a naive `time.time()` around a generate call wrong on an
accelerator. Kernel launches are asynchronous, so the clock must be stopped only
after the device has actually finished; and the first call pays for allocator
warmup, autotuning and lazy module init, which is not what we want to report.
`timed_runs` handles both, and records the device it ran on so a number can
never be silently read as if it came from different hardware.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch


def synchronize(device: torch.device | str | None = None) -> None:
    """Block until the device has finished all queued work.

    A no-op on CPU. Dispatches per backend because `torch.cuda.synchronize()`
    does nothing for MPS, and timing MPS without its own barrier measures how
    fast Python can enqueue kernels rather than how fast they run.
    """
    if device is None:
        return
    kind = torch.device(device).type
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass
class RunTiming:
    """One timed repeat."""

    seconds: float
    tokens: int
    forward_passes: int

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Tokens emitted per target forward pass -- the speedup ceiling.

        Unlike tokens/sec this is hardware-independent: it is a property of the
        drafter and the model, so it transfers across machines in a way a
        wall-clock number never does.
        """
        return self.tokens / self.forward_passes if self.forward_passes else 0.0


@dataclass
class TimingSummary:
    """Aggregated timings for one method on one prompt set."""

    device: str
    runs: list[RunTiming] = field(default_factory=list)

    def add(self, seconds: float, tokens: int, forward_passes: int) -> None:
        self.runs.append(RunTiming(seconds, tokens, forward_passes))

    @property
    def tokens_per_second(self) -> list[float]:
        return [r.tokens_per_second for r in self.runs]

    def median_tps(self) -> float:
        return statistics.median(self.tokens_per_second) if self.runs else 0.0

    def p90_tps(self) -> float:
        """90th percentile, using the nearest-rank method.

        Deliberately not interpolated: with only a handful of repeats,
        interpolating invents a value between two measurements that is itself
        not a measurement.
        """
        values = sorted(self.tokens_per_second)
        if not values:
            return 0.0
        rank = max(1, int(round(0.9 * len(values))))
        return values[min(rank, len(values)) - 1]

    def mean_tokens_per_forward(self) -> float:
        if not self.runs:
            return 0.0
        return statistics.mean(r.tokens_per_forward for r in self.runs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "median_tokens_per_second": self.median_tps(),
            "p90_tokens_per_second": self.p90_tps(),
            "mean_tokens_per_forward": self.mean_tokens_per_forward(),
            "n_runs": len(self.runs),
            "runs": [asdict(r) for r in self.runs],
        }


def timed_runs(
    fn: Callable[[], tuple[int, int]],
    device: torch.device | str | None,
    repeats: int = 3,
    warmup: int = 1,
) -> TimingSummary:
    """Run `fn` `warmup` + `repeats` times, timing only the repeats.

    Args:
        fn: performs one generation and returns ``(tokens, forward_passes)``.
        device: what to synchronize on.
        repeats: timed runs.
        warmup: discarded runs, to pay allocator and autotune costs up front.
    """
    for _ in range(warmup):
        fn()
    synchronize(device)

    summary = TimingSummary(device=str(device))
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        tokens, forwards = fn()
        synchronize(device)
        summary.add(time.perf_counter() - start, tokens, forwards)
    return summary
