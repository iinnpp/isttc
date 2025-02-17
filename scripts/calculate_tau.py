"""
Functions that are used to estimate intrinsic timescale (time constant tau) based on ACF.
"""

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score

import warnings


def func_single_exp(x, a, b, c):
    """
    Exponential function to fit the data.
    :param x: 1d array, independent variable
    :param a: float, parameter to fit
    :param b: float, parameter to fit
    :param c: float, parameter to fit
    :return: callable
    """
    return a * np.exp(-b * x) + c


def func_single_exp_monkey(x, a, b, c):
    """
    Exponential function to fit the data.
    :param x: 1d array, independent variable
    :param a: float, parameter to fit
    :param b: float, parameter to fit
    :param c: float, parameter to fit
    :return: callable
    """
    return a * (np.exp(-b * x) + c)


def fit_single_exp(ydata_to_fit_, start_idx_=1, exp_fun_=func_single_exp):
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
