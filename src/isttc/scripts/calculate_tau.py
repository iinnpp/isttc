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
    """Exponential decay with tau fitted directly.

    :param x: 1d array, independent variable
    :param a: float, amplitude parameter
    :param tau: float, time constant parameter
    :param c: float, offset parameter
    :return: computed exponential function values
    """
    return a * (np.exp(-x / tau) + c)


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
            tau_ci = _student_t_tau_ci(tau, fit_pcov, len(ydata_to_fit_), len(fit_popt))
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
