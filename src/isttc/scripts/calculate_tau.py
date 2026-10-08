"""Estimate intrinsic timescale (tau) from autocorrelation functions."""

import numpy as np
import warnings

from scipy import stats
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import explained_variance_score, r2_score


MAX_FIT_EVALUATIONS = 1_000_000_000
TAU_PARAM_INDEX = 1
CONFIDENCE_LEVEL = 0.975
BOUNDED_TAU_PARAMS = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])


def func_single_exp(x, a, tau, c):
    """Exponential decay plus a scaled constant offset.

    The model is ``a * (exp(-x / tau) + c)``, so its amplitude is ``a``
    and its additive baseline is ``a * c``.

    :param x: 1d array, independent variable
    :param a: float, amplitude parameter
    :param tau: float, time constant parameter
    :param c: float, offset scaled relative to the amplitude
    :return: computed exponential function values
    """
    return a * (np.exp(-x / tau) + c)


def func_multi_exp(x, *params):
    """Constant baseline plus a sum of exponential decays.

    Parameters are ``(c_1, tau_1, c_2, tau_2, ..., baseline)``. The
    baseline is additive and is not treated as a timescale component.
    """
    if len(params) < 3 or len(params) % 2 == 0:
        raise ValueError(
            "Multi-exponential parameters must be (c, tau) pairs followed by a baseline"
        )

    x = np.asarray(x, dtype=float)
    coefficients = np.asarray(params[:-1:2], dtype=float)
    taus = np.asarray(params[1:-1:2], dtype=float)
    baseline = float(params[-1])
    return baseline + np.sum(
        coefficients[:, np.newaxis] * np.exp(-x[np.newaxis, :] / taus[:, np.newaxis]),
        axis=0,
    )


def _nan_ci():
    return np.nan, np.nan


def _format_fit_error(error):
    print(f'{type(error).__name__}: {error}')
    if isinstance(error, ValueError):
        print('Possible reason: acf contains NaNs, low spike count')
    return type(error).__name__


def _student_t_tau_ci(tau, covariance, n_observations, n_params):
    dof = max(n_observations - n_params, 1)
    t_score = stats.t.ppf(CONFIDENCE_LEVEL, dof)
    tau_std_err = np.sqrt(covariance[TAU_PARAM_INDEX, TAU_PARAM_INDEX])
    return tau - t_score * tau_std_err, tau + t_score * tau_std_err


def _normal_tau_ci(tau, covariance):
    tau_variance = covariance[TAU_PARAM_INDEX, TAU_PARAM_INDEX]
    if np.isnan(tau_variance):
        return _nan_ci(), np.nan

    tau_std_err = np.sqrt(tau_variance)
    z_score = stats.norm.ppf(CONFIDENCE_LEVEL)
    return (tau - z_score * tau_std_err, tau + z_score * tau_std_err), tau_variance


def _fit_quality(y_true, y_pred):
    return r2_score(y_true, y_pred), explained_variance_score(y_true, y_pred)


def _multi_exp_initial_parameters(x, y, n_components, min_tau, max_tau, offset):
    """Construct initial component pairs and an additive baseline."""
    fractions = np.arange(1, n_components + 1, dtype=float) / (n_components + 1)
    fractions = np.clip(fractions + offset, 0.02, 0.98)
    log_taus = np.log(min_tau) + fractions * (np.log(max_tau) - np.log(min_tau))
    taus = np.exp(log_taus)

    # Jointly estimate starting amplitudes and the constant baseline for the
    # chosen taus, then project only the amplitudes onto their non-negative
    # parameter space. The baseline remains unbounded.
    design = np.column_stack([
        np.exp(-x[:, np.newaxis] / taus[np.newaxis, :]),
        np.ones(x.size),
    ])
    linear_params, *_ = np.linalg.lstsq(design, y, rcond=None)
    coefficients = linear_params[:-1]
    baseline = linear_params[-1]
    scale = max(float(np.nanmax(np.abs(y))), np.finfo(float).eps)
    coefficients = np.maximum(coefficients, scale * 1e-6)

    initial = np.empty(2 * n_components + 1, dtype=float)
    initial[0:-1:2] = coefficients
    initial[1:-1:2] = taus
    initial[-1] = baseline
    return initial


def _sort_multi_exp_fit(popt, pcov):
    """Order components by increasing tau and leave the baseline last."""
    component_order = np.argsort(popt[1:-1:2])
    parameter_order = [
        index
        for component in component_order
        for index in (2 * component, 2 * component + 1)
    ]
    parameter_order.append(len(popt) - 1)
    parameter_order = np.asarray(parameter_order)
    return popt[parameter_order], pcov[np.ix_(parameter_order, parameter_order)]


def fit_multi_exponential(
    ydata_to_fit_,
    lag_times_=None,
    start_idx_=1,
    max_components_=4,
    min_component_fraction_=0.01,
    min_tau_=None,
    max_tau_=None,
    n_initializations_=5,
):
    """Fit and select 1--4 exponential components with an additive baseline.

    The fitted model is ``AC(t) = baseline + sum_i c_i * exp(-t / tau_i)``.
    Including an additive baseline in every candidate makes the one-component
    model equivalent to ``func_single_exp`` up to parameterization. Candidate
    models are ranked by ``BIC = n * log(RSS / n) + k * log(n)``, where
    ``k = 2 * n_components + 1``. A candidate is eligible for selection only
    when every exponential component contributes at least
    ``min_component_fraction_`` of the total exponential amplitude. The
    baseline is excluded from both this contribution calculation and the
    amplitude-weighted effective timescale. No R-squared threshold is applied;
    R-squared and explained variance are returned only as diagnostics.

    :param ydata_to_fit_: 1D autocorrelation values.
    :param lag_times_: Optional 1D lag values in physical units. If omitted,
        sample indices are used and returned taus are therefore in lag steps.
    :param start_idx_: Index of the first lag included in fitting.
    :param max_components_: Maximum number of exponential components (1--4).
    :param min_component_fraction_: Minimum fractional amplitude per component.
    :param min_tau_: Optional lower timescale bound, in ``lag_times_`` units.
    :param max_tau_: Optional upper timescale bound, in ``lag_times_`` units.
    :param n_initializations_: Number of deterministic initializations per model.
    :return: Dictionary containing the selected baseline, all selected taus
        and coefficients, the effective timescale of the baseline-subtracted
        decay, fit diagnostics, and results for every candidate.
    """
    ydata = np.asarray(ydata_to_fit_, dtype=float)
    if ydata.ndim != 1:
        raise ValueError("ydata_to_fit_ must be one-dimensional")
    if lag_times_ is None:
        lag_times = np.arange(ydata.size, dtype=float)
    else:
        lag_times = np.asarray(lag_times_, dtype=float)
        if lag_times.ndim != 1 or lag_times.shape != ydata.shape:
            raise ValueError("lag_times_ must be one-dimensional and match the ACF shape")
    if not 0 <= start_idx_ < ydata.size:
        raise ValueError("start_idx_ must index an element of the ACF")
    if not 1 <= max_components_ <= 4:
        raise ValueError("max_components_ must be between 1 and 4")
    if not 0 <= min_component_fraction_ < 1:
        raise ValueError("min_component_fraction_ must be in [0, 1)")
    if n_initializations_ < 1:
        raise ValueError("n_initializations_ must be at least 1")

    x_fit = lag_times[start_idx_:]
    y_fit = ydata[start_idx_:]
    finite = np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit = x_fit[finite]
    y_fit = y_fit[finite]
    if x_fit.size < 3:
        raise ValueError("At least three finite ACF observations are required")
    if np.any(np.diff(x_fit) <= 0):
        raise ValueError("Fitted lag times must be strictly increasing")
    if np.any(x_fit < 0):
        raise ValueError("Fitted lag times must be non-negative")

    positive_steps = np.diff(np.unique(lag_times[np.isfinite(lag_times)]))
    positive_steps = positive_steps[positive_steps > 0]
    inferred_min_tau = float(np.min(positive_steps)) if positive_steps.size else np.nan
    min_tau = inferred_min_tau if min_tau_ is None else float(min_tau_)
    max_tau = float(np.max(x_fit)) if max_tau_ is None else float(max_tau_)
    if not np.isfinite(min_tau) or not np.isfinite(max_tau) or min_tau <= 0:
        raise ValueError("Timescale bounds must be finite and min_tau_ must be positive")
    if max_tau <= min_tau:
        raise ValueError("max_tau_ must be greater than min_tau_")

    candidate_fits = []
    initialization_offsets = (
        np.asarray([0.0])
        if n_initializations_ == 1
        else np.linspace(-0.2, 0.2, n_initializations_)
    )

    for n_components in range(1, max_components_ + 1):
        n_params = 2 * n_components + 1
        if x_fit.size <= n_params:
            candidate_fits.append({
                "n_components": n_components,
                "log_message": "insufficient observations",
            })
            continue

        lower_bounds = np.concatenate([
            np.tile([0.0, min_tau], n_components),
            [-np.inf],
        ])
        upper_bounds = np.concatenate([
            np.tile([np.inf, max_tau], n_components),
            [np.inf],
        ])
        best_attempt = None

        for offset in initialization_offsets:
            p0 = _multi_exp_initial_parameters(
                x_fit, y_fit, n_components, min_tau, max_tau, offset
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    popt, pcov = curve_fit(
                        func_multi_exp,
                        x_fit,
                        y_fit,
                        p0=p0,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=MAX_FIT_EVALUATIONS,
                    )
                y_pred = func_multi_exp(x_fit, *popt)
                rss = float(np.sum((y_fit - y_pred) ** 2))
                if best_attempt is None or rss < best_attempt[0]:
                    best_attempt = rss, popt, pcov, y_pred
            except (RuntimeError, OptimizeWarning, RuntimeWarning, ValueError):
                continue

        if best_attempt is None:
            candidate_fits.append({
                "n_components": n_components,
                "log_message": "fit failed",
            })
            continue

        rss, popt, pcov, y_pred = best_attempt
        popt, pcov = _sort_multi_exp_fit(popt, pcov)
        y_pred = func_multi_exp(x_fit, *popt)
        coefficients = popt[:-1:2]
        taus = popt[1:-1:2]
        baseline = float(popt[-1])
        coefficient_sum = float(np.sum(coefficients))
        fractions = (
            coefficients / coefficient_sum
            if coefficient_sum > 0
            else np.zeros_like(coefficients)
        )
        passes_contribution = bool(np.all(fractions >= min_component_fraction_))
        safe_rss = max(rss, np.finfo(float).tiny)
        bic = float(
            x_fit.size * np.log(safe_rss / x_fit.size)
            + n_params * np.log(x_fit.size)
        )
        fit_r_squared, explained_var = _fit_quality(y_fit, y_pred)

        candidate_fits.append({
            "n_components": n_components,
            "n_params": n_params,
            "coefficients": coefficients,
            "taus": taus,
            "baseline": baseline,
            "popt": popt,
            "pcov": pcov,
            "rss": rss,
            "bic": bic,
            "y_pred": y_pred,
            "fit_r_squared": fit_r_squared,
            "explained_var": explained_var,
            "component_fractions": fractions,
            "passes_contribution": passes_contribution,
            "log_message": "ok",
        })

    eligible = [
        candidate for candidate in candidate_fits
        if candidate.get("log_message") == "ok" and candidate["passes_contribution"]
    ]
    if not eligible:
        return {
            "n_components": 0,
            "coefficients": np.asarray([], dtype=float),
            "taus": np.asarray([], dtype=float),
            "baseline": np.nan,
            "tau_eff": np.nan,
            "popt": np.nan,
            "pcov": np.nan,
            "bic": np.nan,
            "rss": np.nan,
            "fit_r_squared": np.nan,
            "explained_var": np.nan,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "candidate_fits": candidate_fits,
            "log_message": "no eligible model",
        }

    selected = min(eligible, key=lambda candidate: candidate["bic"])
    tau_eff = float(
        np.sum(selected["coefficients"] * selected["taus"])
        / np.sum(selected["coefficients"])
    )
    return {
        **selected,
        "tau_eff": tau_eff,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "candidate_fits": candidate_fits,
        "log_message": "ok",
    }


def fit_single_exp(ydata_to_fit_, start_idx_=1, exp_fun_=func_single_exp):
    """Fit an exponential function to one ACF using non-linear least squares.

    Confidence interval is estimated using Student's t-distribution.

    :param ydata_to_fit_: 1D array, dependent data to fit
    :param start_idx_: int, index to start fitting from (default: 1)
    :param exp_fun_: function, the exponential function to fit
    :return: fit_popt, fit_pcov, tau, tau_CI, fit_r_squared, explained_var, log_message
    """
    t = np.arange(len(ydata_to_fit_))
    x_fit = t[start_idx_:]
    y_fit = ydata_to_fit_[start_idx_:]

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            popt, pcov = curve_fit(
                exp_fun_,
                x_fit,
                y_fit,
                maxfev=MAX_FIT_EVALUATIONS,
                bounds=BOUNDED_TAU_PARAMS,
            )
            fit_popt = popt
            fit_pcov = pcov
            tau = fit_popt[TAU_PARAM_INDEX]
            tau_ci = _student_t_tau_ci(tau, fit_pcov, len(y_fit), len(fit_popt))
            y_pred = exp_fun_(x_fit, *popt)
            fit_r_squared, explained_var = _fit_quality(y_fit, y_pred)
            log_message = "ok"
        except (RuntimeError, OptimizeWarning, RuntimeWarning, ValueError) as error:
            fit_popt = fit_pcov = tau = fit_r_squared = explained_var = np.nan
            tau_ci = _nan_ci()
            log_message = _format_fit_error(error)

    return fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message


def fit_single_exp_2d(ydata_to_fit_2d_, start_idx_=1, exp_fun_=func_single_exp):
    """Fit an exponential function to stacked trial ACF values.

    Confidence interval is estimated using normal distribution.

    :param exp_fun_:
    :param ydata_to_fit_2d_: 1d array, the dependant data to fit
    :param start_idx_: int, index to start fitting from
    :return: fit_popt, fit_pcov, tau, fit_r_squared, log_message
    """
    t = np.arange(start_idx_, ydata_to_fit_2d_.shape[1])
    acf_1d = np.hstack(ydata_to_fit_2d_[:, start_idx_:])
    t_1d = np.tile(t, reps=ydata_to_fit_2d_.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            popt, pcov = curve_fit(exp_fun_, t_1d, acf_1d, maxfev=MAX_FIT_EVALUATIONS)
            fit_popt = popt
            fit_pcov = pcov
            tau = fit_popt[TAU_PARAM_INDEX]
            tau_ci, tau_variance = _normal_tau_ci(tau, fit_pcov)
            y_pred = exp_fun_(t_1d, *popt)
            fit_r_squared, fit_explained_var = _fit_quality(acf_1d, y_pred)
            log_message = 'ok'
        except (RuntimeError, OptimizeWarning, RuntimeWarning, ValueError) as error:
            fit_popt = fit_pcov = tau = tau_variance = fit_r_squared = fit_explained_var = np.nan
            tau_ci = _nan_ci()
            log_message = _format_fit_error(error)

    return fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, fit_explained_var, log_message
