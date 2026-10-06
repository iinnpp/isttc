import numpy as np

from isttc.scripts import calculate_tau


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
