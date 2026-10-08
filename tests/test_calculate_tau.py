import numpy as np

from isttc.acfunc import acf_sttc_fast
from isttc.scripts import calculate_tau
from isttc.spike_utils import simulate_hawkes_thinning


def test_fit_single_exp_ci_uses_only_fitted_observations(monkeypatch):
    ydata = np.linspace(1.0, 0.1, 11)
    popt = np.array([1.0, 2.0, 0.0])
    pcov = np.eye(3)
    captured = {}

    def fake_curve_fit(*args, **kwargs):
        return popt, pcov

    def fake_student_t_tau_ci(tau, covariance, n_observations, n_params):
        captured.update(
            tau=tau,
            covariance=covariance,
            n_observations=n_observations,
            n_params=n_params,
        )
        return 1.0, 3.0

    monkeypatch.setattr(calculate_tau, "curve_fit", fake_curve_fit)
    monkeypatch.setattr(calculate_tau, "_student_t_tau_ci", fake_student_t_tau_ci)

    fit = calculate_tau.fit_single_exp(ydata, start_idx_=1)

    assert captured["n_observations"] == 10
    assert captured["n_params"] == 3
    assert captured["tau"] == popt[calculate_tau.TAU_PARAM_INDEX]
    np.testing.assert_array_equal(captured["covariance"], pcov)
    assert fit[3] == (1.0, 3.0)


def test_func_multi_exp_sums_all_components():
    x = np.array([0.0, 10.0, 20.0])
    actual = calculate_tau.func_multi_exp(x, 0.7, 10.0, 0.3, 100.0, 0.05)
    expected = 0.7 * np.exp(-x / 10.0) + 0.3 * np.exp(-x / 100.0) + 0.05

    np.testing.assert_allclose(actual, expected)


def test_one_component_multi_exp_is_nested_with_single_exp():
    lag_steps = np.arange(101, dtype=float)
    acf = 0.4 * (np.exp(-lag_steps / 12.0) + 0.075)

    single_fit = calculate_tau.fit_single_exp(acf, start_idx_=1)
    multi_fit = calculate_tau.fit_multi_exponential(
        acf,
        lag_times_=lag_steps,
        start_idx_=1,
        max_components_=1,
    )

    single_amplitude, single_tau, single_scaled_offset = single_fit[0]
    np.testing.assert_allclose(multi_fit["coefficients"], [single_amplitude], rtol=1e-6)
    np.testing.assert_allclose(multi_fit["taus"], [single_tau], rtol=1e-6)
    np.testing.assert_allclose(
        multi_fit["baseline"],
        single_amplitude * single_scaled_offset,
        rtol=1e-6,
    )
    assert multi_fit["candidate_fits"][0]["n_params"] == 3


def test_fit_multi_exponential_selects_two_components_and_returns_all_taus():
    lag_times = np.arange(0.0, 505.0, 5.0)
    rng = np.random.default_rng(42)
    acf = (
        0.7 * np.exp(-lag_times / 25.0)
        + 0.3 * np.exp(-lag_times / 180.0)
        + 0.04
        + rng.normal(0.0, 0.001, lag_times.size)
    )

    fit = calculate_tau.fit_multi_exponential(
        acf,
        lag_times_=lag_times,
        max_components_=3,
    )

    assert fit["log_message"] == "ok"
    assert fit["n_components"] == 2
    assert fit["taus"].shape == (2,)
    assert np.all(np.diff(fit["taus"]) > 0)
    np.testing.assert_allclose(fit["taus"], [25.0, 180.0], rtol=0.15)
    np.testing.assert_allclose(fit["baseline"], 0.04, atol=0.005)
    assert len(fit["candidate_fits"]) == 3


def test_multi_exp_baseline_prevents_upper_bound_timescale_artifact():
    lag_times = np.arange(0.0, 1001.0, 10.0)
    acf = 0.3 * np.exp(-lag_times / 100.0) + 0.03
    fits = [
        calculate_tau.fit_multi_exponential(
            acf,
            lag_times_=lag_times,
            max_components_=2,
            max_tau_=max_tau,
        )
        for max_tau in (500.0, 1000.0, 5000.0)
    ]

    assert all(fit["n_components"] == 1 for fit in fits)
    np.testing.assert_allclose([fit["tau_eff"] for fit in fits], 100.0, rtol=1e-6)
    np.testing.assert_allclose([fit["baseline"] for fit in fits], 0.03, rtol=1e-6)


def test_fit_multi_exponential_does_not_reject_low_r_squared():
    lag_times = np.arange(0.0, 105.0, 5.0)
    acf = np.array([
        1.0, 0.3, 0.8, 0.2, 0.7, 0.1, 0.6, 0.0, 0.5, -0.1, 0.4,
        -0.2, 0.3, -0.3, 0.2, -0.4, 0.1, -0.5, 0.0, -0.6, -0.1,
    ])

    fit = calculate_tau.fit_multi_exponential(
        acf,
        lag_times_=lag_times,
        max_components_=1,
    )

    assert fit["log_message"] == "ok"
    assert fit["n_components"] == 1
    assert fit["fit_r_squared"] < 0.5


def test_fit_multi_exponential_rejects_component_below_one_percent():
    lag_times = np.arange(0.0, 505.0, 5.0)
    acf = (
        0.995 * np.exp(-lag_times / 50.0)
        + 0.005 * np.exp(-lag_times / 250.0)
        + 0.02
    )

    fit = calculate_tau.fit_multi_exponential(
        acf,
        lag_times_=lag_times,
        max_components_=2,
        min_component_fraction_=0.01,
    )
    one_component, two_component = fit["candidate_fits"]

    # BIC alone prefers the exact two-component fit, but its 0.5% component
    # is below the minimum contribution and therefore makes it ineligible.
    assert two_component["bic"] < one_component["bic"]
    assert np.min(two_component["component_fractions"]) < 0.01
    assert not two_component["passes_contribution"]
    assert fit["n_components"] == 1
    assert fit["passes_contribution"]


def _recover_hawkes_timescale(target_tau_ms, seed):
    duration_ms = 600_000
    lag_shift_ms = 10.0
    n_lags = 100
    spikes = simulate_hawkes_thinning(
        fr_hz_=5.0,
        tau_ms_=target_tau_ms,
        alpha_=0.5,
        duration_ms_=duration_ms,
        seed_=seed,
    )
    acf = acf_sttc_fast(
        spikes,
        n_lags_=n_lags,
        lag_shift_=lag_shift_ms,
        sttc_dt_=5.0,
        signal_length_=duration_ms,
    )

    single_fit = calculate_tau.fit_single_exp(acf, start_idx_=1)
    single_tau_ms = single_fit[2] * lag_shift_ms

    lag_times_ms = np.arange(n_lags + 1) * lag_shift_ms
    multi_fit = calculate_tau.fit_multi_exponential(
        acf,
        lag_times_=lag_times_ms,
        start_idx_=1,
    )

    assert single_fit[6] == "ok"
    assert multi_fit["log_message"] == "ok"
    return single_tau_ms, multi_fit["tau_eff"], multi_fit["n_components"]


def test_multi_exponential_recovers_hawkes_timescales_across_replicates():
    settings = [
        (target_tau_ms, 100 + replicate + int(target_tau_ms))
        for target_tau_ms in (50.0, 100.0, 200.0)
        for replicate in range(4)
    ]
    single_errors = []
    multi_errors = []
    selected_components = []

    for target_tau_ms, seed in settings:
        single_tau_ms, multi_tau_ms, n_components = _recover_hawkes_timescale(
            target_tau_ms, seed
        )
        single_errors.append(abs(single_tau_ms - target_tau_ms) / target_tau_ms)
        multi_errors.append(abs(multi_tau_ms - target_tau_ms) / target_tau_ms)
        selected_components.append(n_components)

    single_errors = np.asarray(single_errors)
    multi_errors = np.asarray(multi_errors)
    diagnostics = (
        f"single errors={single_errors.tolist()}, "
        f"multi errors={multi_errors.tolist()}, "
        f"selected components={selected_components}"
    )

    # Both methods should recover each generating timescale reasonably well.
    assert np.max(single_errors) < 0.25, diagnostics
    assert np.max(multi_errors) < 0.25, diagnostics

    # A selected one-component model is equivalent to fit_single_exp, apart
    # from numerical optimization details. Additional components may improve
    # recovery, but should not materially degrade it across these replicates.
    one_component = np.asarray(selected_components) == 1
    np.testing.assert_allclose(
        multi_errors[one_component],
        single_errors[one_component],
        atol=1e-5,
    )
    assert np.mean(multi_errors) < np.mean(single_errors), diagnostics
    assert np.median(multi_errors) < np.median(single_errors), diagnostics
    assert np.max(multi_errors - single_errors) < 0.01, diagnostics
    assert np.count_nonzero(multi_errors < single_errors - 0.01) >= 2, diagnostics
