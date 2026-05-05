import os
import statistics
import time

import numpy as np
import pytest

from isttc.acfunc import acf_sttc, acf_sttc_fast
from isttc.scripts.calculate_acf import (
    sttc_calculate_p,
    sttc_calculate_p_fast,
    sttc_calculate_t,
    sttc_calculate_t_fast,
)
from isttc.spike_utils import simulate_hawkes_thinning


N_LAGS = 20
LAG_SHIFT_MS = 50
DT_MS = 25
T_MS = 60_000
BENCHMARK_T_MS = 600_000
BENCHMARK_N_LAGS = 30
BENCHMARK_SETTINGS = tuple(
    zip(
        np.geomspace(0.01, 10.0, num=7),
        np.linspace(50.0, 300.0, num=7),
        np.linspace(0.1, 0.9, num=7),
    )
)


def _synthetic_spikes(seed=0, duration_ms=T_MS, fr_hz=3.5, tau_ms=100.0, alpha=0.3):
    return simulate_hawkes_thinning(
        fr_hz_=fr_hz,
        tau_ms_=tau_ms,
        alpha_=alpha,
        duration_ms_=duration_ms,
        seed_=seed,
    )


def _reference_acf(spikes, n_lags=N_LAGS, lag_shift_ms=LAG_SHIFT_MS, dt_ms=DT_MS, t_ms=T_MS):
    return np.asarray(
        acf_sttc(
            spikes,
            n_lags,
            lag_shift_=lag_shift_ms,
            sttc_dt_=dt_ms,
            signal_length_=t_ms,
            verbose_=False,
        ),
        dtype=np.float32,
    )


def _time_call(func, *args):
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    ("spikes", "dt_ms", "t_start_ms", "t_stop_ms"),
    [
        (np.array([], dtype=np.float64), 3, 0, 100),
        (np.array([12.0], dtype=np.float64), 3, 0, 100),
        (np.array([1.0, 8.0], dtype=np.float64), 3, 0, 20),
        (np.array([1.0, 4.0, 8.0, 18.0], dtype=np.float64), 5, 0, 20),
    ],
)
def test_sttc_calculate_t_fast_matches_reference(spikes, dt_ms, t_start_ms, t_stop_ms):
    actual = sttc_calculate_t_fast(spikes, len(spikes), dt_ms, t_start_ms, t_stop_ms)
    expected = sttc_calculate_t(spikes, len(spikes), dt_ms, t_start_ms, t_stop_ms, verbose_=False)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("spikes_a", "spikes_b", "dt_ms"),
    [
        (np.array([], dtype=np.float64), np.array([1.0, 2.0], dtype=np.float64), 3),
        (np.array([12.0], dtype=np.float64), np.array([], dtype=np.float64), 3),
        (np.array([1.0, 8.0], dtype=np.float64), np.array([2.0, 20.0], dtype=np.float64), 3),
        (np.array([1.0, 10.0, 25.0], dtype=np.float64), np.array([5.0, 13.0, 30.0], dtype=np.float64), 5),
    ],
)
def test_sttc_calculate_p_fast_matches_reference(spikes_a, spikes_b, dt_ms):
    actual = sttc_calculate_p_fast(spikes_a, spikes_b, len(spikes_a), len(spikes_b), dt_ms)
    expected = sttc_calculate_p(spikes_a, spikes_b, len(spikes_a), len(spikes_b), dt_ms)

    assert actual == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_acf_sttc_fast_matches_reference_on_synthetic_hawkes_spike_trains(seed):
    spikes = _synthetic_spikes(seed=seed)

    actual = acf_sttc_fast(spikes, N_LAGS, LAG_SHIFT_MS, DT_MS, T_MS)
    expected = _reference_acf(spikes)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("spikes", "n_lags", "lag_shift_ms", "dt_ms", "t_ms"),
    [
        (np.array([], dtype=np.float64), 5, 10, 3, 100),
        (np.array([12.0], dtype=np.float64), 5, 10, 3, 100),
        (np.array([1.0, 8.0], dtype=np.float64), 5, 10, 3, 20),
    ],
)
def test_acf_sttc_fast_edge_cases_match_reference(spikes, n_lags, lag_shift_ms, dt_ms, t_ms):
    actual = acf_sttc_fast(spikes, n_lags, lag_shift_ms, dt_ms, t_ms)
    expected = _reference_acf(spikes, n_lags, lag_shift_ms, dt_ms, t_ms)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)


@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("ISTTC_RUN_BENCHMARKS") != "1",
    reason="set ISTTC_RUN_BENCHMARKS=1 to run timing-sensitive benchmark tests",
)
def test_acf_sttc_fast_benchmark_across_synthetic_parameter_range():
    speedups = []
    nontrivial_speedups = []
    benchmark_rows = []

    for seed, (fr_hz, tau_ms, alpha) in enumerate(BENCHMARK_SETTINGS, start=123):
        spikes = _synthetic_spikes(
            seed=seed,
            duration_ms=BENCHMARK_T_MS,
            fr_hz=fr_hz,
            tau_ms=tau_ms,
            alpha=alpha,
        )

        expected = _reference_acf(spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)
        actual = acf_sttc_fast(spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9)

        # Warm both paths before measuring.
        acf_sttc_fast(spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)
        _reference_acf(spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)

        optimized_times = [
            _time_call(acf_sttc_fast, spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)
            for _ in range(3)
        ]
        reference_times = [
            _time_call(_reference_acf, spikes, BENCHMARK_N_LAGS, LAG_SHIFT_MS, DT_MS, BENCHMARK_T_MS)
            for _ in range(3)
        ]
        speedup = statistics.median(reference_times) / statistics.median(optimized_times)
        speedups.append(speedup)
        benchmark_rows.append((len(spikes), fr_hz, tau_ms, alpha, speedup))

        if len(spikes) >= 100:
            nontrivial_speedups.append(speedup)

    summary = "; ".join(
        f"n={n_spikes}, fr={fr_hz:.3g}, tau={tau_ms:.1f}, alpha={alpha:.3g}, speedup={speedup:.2f}x"
        for n_spikes, fr_hz, tau_ms, alpha, speedup in benchmark_rows
    )
    assert nontrivial_speedups, summary
    assert min(nontrivial_speedups) >= 10.0, summary
    assert statistics.median(nontrivial_speedups) >= 15.0, summary
