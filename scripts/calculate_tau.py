"""
Functions that are used to estimate intrinsic timescale (time constant tau) based on ACF.
"""

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score, explained_variance_score
from scipy import stats

import warnings


def deprecated_func_single_exp(x, a, b, c):
    """
    Exponential function to fit the data.
    :param x: 1d array, independent variable
    :param a: float, parameter to fit
    :param b: float, parameter to fit
    :param c: float, parameter to fit
    :return: callable
    """
    return a * np.exp(-b * x) + c


def func_single_exp(x, a, tau, c):
    """
    Exponential function where the decay time constant tau is fitted directly.

    :param x: 1d array, independent variable
    :param a: float, amplitude parameter
    :param tau: float, time constant parameter
    :param c: float, offset parameter
    :return: computed exponential function values
    """
    return a * np.exp(-x / tau) + c


def deprecated_func_single_exp_monkey(x, a, b, c):
    """
    Exponential function to fit the data.
    :param x: 1d array, independent variable
    :param a: float, parameter to fit
    :param b: float, parameter to fit
    :param c: float, parameter to fit
    :return: callable
    """
    return a * (np.exp(-b * x) + c)


def func_single_exp_monkey(x, a, tau, c):
    """
    Exponential function where the decay time constant tau is fitted directly.

    :param x: 1d array, independent variable
    :param a: float, amplitude parameter
    :param tau: float, time constant parameter
    :param c: float, offset parameter
    :return: computed exponential function values
    """
    return a * (np.exp(-x / tau) + c)


def deprecated_fit_single_exp(ydata_to_fit_, start_idx_=1, exp_fun_=func_single_exp):
    """
    Fit function func_exp to data using non-linear least square.

    :param exp_fun_:
    :param ydata_to_fit_: 1d array, the dependant data to fit
    :param start_idx_: int, index to start fitting from
    :return: fit_popt, fit_pcov, tau, fit_r_squared, log_message
    """
    t = np.linspace(0, len(ydata_to_fit_) - 1, len(ydata_to_fit_)).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # maxfev - I used 5000, now it is like in Siegle
            popt, pcov = curve_fit(exp_fun_, t[start_idx_:], ydata_to_fit_[start_idx_:], maxfev=1000000000)
            fit_popt = popt
            fit_pcov = pcov
            tau = 1 / fit_popt[1]
            # fit r-squared
            y_pred = exp_fun_(t[start_idx_:], *popt)
            fit_r_squared = r2_score(ydata_to_fit_[start_idx_:], y_pred)
            log_message = 'ok'
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'RuntimeError'
        except OptimizeWarning as o:
            print('OptimizeWarning: {}'.format(o))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'OptimizeWarning'
        except RuntimeWarning as re:
            print('RuntimeWarning: {}'.format(re))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'RuntimeWarning'
        except ValueError as ve:
            print('ValueError: {}'.format(ve))
            print('Possible reason: acf contains NaNs, low spike count')
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'ValueError'

    return fit_popt, fit_pcov, tau, fit_r_squared, log_message


def fit_single_exp(ydata_to_fit_, start_idx_=1, exp_fun_=func_single_exp):
    """
    Fit an exponential function to data using non-linear least squares.

    :param ydata_to_fit_: 1D array, dependent data to fit
    :param start_idx_: int, index to start fitting from (default: 1)
    :param exp_fun_: function, the exponential function to fit
    :return: fit_popt, fit_pcov, tau, tau_CI, fit_r_squared, explained_var, log_message
    """

    t = np.arange(len(ydata_to_fit_))  # Time indices

    with warnings.catch_warnings():
        warnings.filterwarnings('error')  # Convert warnings to exceptions

        try:
            # Perform curve fitting with parameter bounds, maxfev - I used 5000, now it is like in Siegle
            popt, pcov = curve_fit(
                exp_fun_, t[start_idx_:], ydata_to_fit_[start_idx_:],
                maxfev=1000000000,
                bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])  # Ensuring tau > 0
            )
            fit_popt = popt
            fit_pcov = pcov
            tau = fit_popt[1]

            # Compute confidence interval for tau using Student's t-distribution
            dof = max(len(ydata_to_fit_) - len(fit_popt), 1)  # Degrees of freedom
            t_score = stats.t.ppf(0.975, dof)  # 95% confidence level
            tau_std_err = np.sqrt(fit_pcov[1, 1])  # Standard error for tau
            tau_ci = (tau - t_score * tau_std_err, tau + t_score * tau_std_err)

            # Compute R-squared score
            y_pred = exp_fun_(t[start_idx_:], *popt)
            fit_r_squared = r2_score(ydata_to_fit_[start_idx_:], y_pred)

            # Compute Explained Variance Score
            explained_var = explained_variance_score(ydata_to_fit_[start_idx_:], y_pred)

            log_message = "Fit successful"
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
            fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var = np.nan, np.nan, np.nan, (
                np.nan, np.nan), np.nan, np.nan
            log_message = 'RuntimeError'
        except OptimizeWarning as o:
            print('OptimizeWarning: {}'.format(o))
            fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var = np.nan, np.nan, np.nan, (
                np.nan, np.nan), np.nan, np.nan
            log_message = 'OptimizeWarning'
        except RuntimeWarning as re:
            print('RuntimeWarning: {}'.format(re))
            fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var = np.nan, np.nan, np.nan, (
                np.nan, np.nan), np.nan, np.nan
            log_message = 'RuntimeWarning'
        except ValueError as ve:
            print('ValueError: {}'.format(ve))
            print('Possible reason: acf contains NaNs, low spike count')
            fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var = np.nan, np.nan, np.nan, (
                np.nan, np.nan), np.nan, np.nan
            log_message = 'ValueError'

    return fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message


def fit_single_exp_2d(ydata_to_fit_2d_, start_idx_=1, exp_fun_=func_single_exp):
    """
    Fit function func_exp to data using non-linear least square.

    :param exp_fun_:
    :param ydata_to_fit_2d_: 1d array, the dependant data to fit
    :param start_idx_: int, index to start fitting from
    :return: fit_popt, fit_pcov, tau, fit_r_squared, log_message
    """

    t = np.linspace(start_idx_, ydata_to_fit_2d_.shape[1] - 1, ydata_to_fit_2d_.shape[1] - start_idx_).astype(int)
    # make 1d for curve_fit
    acf_1d = np.hstack(ydata_to_fit_2d_[:, start_idx_:])
    t_1d = np.tile(t, reps=ydata_to_fit_2d_.shape[0])

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # maxfev - I used 5000, now it is like in Siegle
            popt, pcov = curve_fit(exp_fun_, t_1d, acf_1d, maxfev=1000000000)
            fit_popt = popt
            fit_pcov = pcov
            tau = 1 / fit_popt[1]
            # fit r-squared
            y_pred = exp_fun_(t_1d, *popt)
            fit_r_squared = r2_score(acf_1d, y_pred)
            log_message = 'ok'
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'RuntimeError'
        except OptimizeWarning as o:
            print('OptimizeWarning: {}'.format(o))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'OptimizeWarning'
        except RuntimeWarning as re:
            print('RuntimeWarning: {}'.format(re))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'RuntimeWarning'
        except ValueError as ve:
            print('ValueError: {}'.format(ve))
            print('Possible reason: acf contains NaNs, low spike count')
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
            log_message = 'ValueError'

    return fit_popt, fit_pcov, tau, fit_r_squared, log_message
