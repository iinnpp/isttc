"""Benchmark acf_sttc_fast against the reference acf_sttc implementation."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from isttc.acfunc import acf_sttc, acf_sttc_fast  # noqa: E402
from isttc.spike_utils import simulate_hawkes_thinning  # noqa: E402


LAG_SHIFT_MS = 50
DT_MS = 25


def reference_acf(spikes, n_lags, lag_shift_ms, dt_ms, duration_ms):
    return np.asarray(
        acf_sttc(
            spikes,
            n_lags,
            lag_shift_=lag_shift_ms,
            sttc_dt_=dt_ms,
            signal_length_=duration_ms,
            verbose_=False,
        ),
        dtype=np.float32,
    )


def time_call(func, *args):
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


def median_runtime(func, args, repeats):
    return statistics.median(time_call(func, *args) for _ in range(repeats))


def build_parameter_settings(num_settings):
    return tuple(
        zip(
            np.geomspace(0.01, 10.0, num=num_settings),
            np.linspace(50.0, 300.0, num=num_settings),
            np.linspace(0.1, 0.9, num=num_settings),
        )
    )


def run_benchmark(args):
    rows = []
    settings = build_parameter_settings(args.num_settings)

    for seed, (fr_hz, tau_ms, alpha) in enumerate(settings, start=args.seed):
        spikes = simulate_hawkes_thinning(
            fr_hz_=fr_hz,
            tau_ms_=tau_ms,
            alpha_=alpha,
            duration_ms_=args.duration_ms,
            seed_=seed,
        )

        expected = reference_acf(spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms)
        actual = acf_sttc_fast(spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms)
        max_abs_error = float(np.nanmax(np.abs(actual - expected)))

        acf_sttc_fast(spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms)
        reference_acf(spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms)

        fast_s = median_runtime(
            acf_sttc_fast,
            (spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms),
            args.repeats,
        )
        reference_s = median_runtime(
            reference_acf,
            (spikes, args.n_lags, LAG_SHIFT_MS, DT_MS, args.duration_ms),
            args.repeats,
        )
        speedup = reference_s / fast_s

        rows.append(
            {
                "fr_hz": fr_hz,
                "tau_ms": tau_ms,
                "alpha": alpha,
                "n_spikes": len(spikes),
                "reference_s": reference_s,
                "fast_s": fast_s,
                "speedup": speedup,
                "max_abs_error": max_abs_error,
            }
        )

    return rows


def print_report(rows):
    print(
        "fr_hz    tau_ms  alpha  n_spikes  reference_s  fast_s      speedup   max_abs_error"
    )
    print("-" * 88)
    for row in rows:
        print(
            f"{row['fr_hz']:>7.3g}  "
            f"{row['tau_ms']:>6.1f}  "
            f"{row['alpha']:>5.2f}  "
            f"{row['n_spikes']:>8d}  "
            f"{row['reference_s']:>11.6f}  "
            f"{row['fast_s']:>9.6f}  "
            f"{row['speedup']:>7.2f}x  "
            f"{row['max_abs_error']:>13.3g}"
        )

    speedups = [row["speedup"] for row in rows]
    nontrivial_speedups = [row["speedup"] for row in rows if row["n_spikes"] >= 100]

    print()
    print(f"Average speedup: {statistics.mean(speedups):.2f}x")
    print(f"Median speedup:  {statistics.median(speedups):.2f}x")
    if nontrivial_speedups:
        print(f"Average speedup for trains with >=100 spikes: {statistics.mean(nontrivial_speedups):.2f}x")
        print(f"Median speedup for trains with >=100 spikes:  {statistics.median(nontrivial_speedups):.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark acf_sttc_fast against acf_sttc over synthetic Hawkes spike trains."
    )
    parser.add_argument("--duration-ms", type=float, default=600_000)
    parser.add_argument("--n-lags", type=int, default=30)
    parser.add_argument("--num-settings", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    print_report(run_benchmark(parse_args()))
