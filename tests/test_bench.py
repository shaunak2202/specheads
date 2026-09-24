import math

import pytest

from specheads.bench.metrics import (
    HARDWARE_DEPENDENT,
    HARDWARE_INDEPENDENT,
    bootstrap_ci,
    speedup,
    summarize_acceptance,
)
from specheads.bench.timing import TimingSummary, synchronize, timed_runs


def test_tokens_per_forward_is_the_speedup_ceiling():
    s = TimingSummary(device="cpu")
    s.add(seconds=2.0, tokens=100, forward_passes=25)
    assert s.runs[0].tokens_per_forward == pytest.approx(4.0)
    assert s.runs[0].tokens_per_second == pytest.approx(50.0)


def test_p90_is_nearest_rank_not_interpolated():
    """With 3 repeats an interpolated p90 would report a value never measured."""
    s = TimingSummary(device="cpu")
    for seconds in (1.0, 2.0, 4.0):  # 100, 50, 25 tps
        s.add(seconds=seconds, tokens=100, forward_passes=100)
    assert s.p90_tps() in {25.0, 50.0, 100.0}
    assert s.p90_tps() == 100.0
    assert s.median_tps() == pytest.approx(50.0)


def test_empty_summary_is_zero_not_error():
    s = TimingSummary(device="cpu")
    assert s.median_tps() == 0.0 and s.p90_tps() == 0.0


def test_timed_runs_records_exactly_the_requested_repeats():
    calls = []

    def fn():
        calls.append(1)
        return 10, 5

    summary = timed_runs(fn, device=None, repeats=3, warmup=2)
    assert len(calls) == 5          # warmups happened
    assert len(summary.runs) == 3   # but were not timed
    assert all(r.seconds > 0 for r in summary.runs)


def test_synchronize_is_a_noop_without_a_device():
    synchronize(None)
    synchronize("cpu")


def test_speedup_against_a_dead_baseline_is_nan():
    assert math.isnan(speedup(10.0, 0.0))
    assert speedup(20.0, 10.0) == pytest.approx(2.0)


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = bootstrap_ci(values, samples=2000, seed=1)
    assert low < 3.0 < high


def test_bootstrap_ci_is_deterministic_under_a_seed():
    values = [1.0, 4.0, 2.0, 8.0]
    assert bootstrap_ci(values, samples=500, seed=7) == bootstrap_ci(values, samples=500, seed=7)


def test_bootstrap_ci_of_a_single_value_is_degenerate_not_nan():
    assert bootstrap_ci([2.5]) == (2.5, 2.5)


def test_summarize_acceptance_reports_an_interval():
    out = summarize_acceptance([1, 2, 3, 2, 1])
    assert out["mean_accepted_length"] == pytest.approx(1.8)
    assert out["ci_low"] <= 1.8 <= out["ci_high"]


def test_hardware_classes_are_disjoint():
    """The guard that keeps a laptop timing from being read as a T4 timing."""
    assert not (HARDWARE_DEPENDENT & HARDWARE_INDEPENDENT)
    assert "median_tokens_per_second" in HARDWARE_DEPENDENT
    assert "mean_accepted_length" in HARDWARE_INDEPENDENT
