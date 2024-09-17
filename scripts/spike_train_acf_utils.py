"""
Functions that are used to calculate and fit ACF on spiking data.
"""

import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient
import neo

import warnings


def bin_spike_train(spike_train_int_l_, bin_length_ms_, fs_, verbose_=False):
    """
    Bin spike train.

    :param spike_train_int_l_: list, list of spike times (int), sampling frequency fs_
    :param bin_length_ms_: int, bin length in ms
    :param fs_: int, sampling frequency in Hz
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, binned spike train, each bin contains spike count
    """
    bin_length_fs = int(fs_ / 1000 * bin_length_ms_)
    n_bin_edges = int(np.ceil(spike_train_int_l_[-1] / bin_length_fs))  # using ceil to include the last spike
    bins_ = np.linspace(0, bin_length_fs * n_bin_edges, n_bin_edges).astype(int)
    binned_spike_train, _ = np.histogram(spike_train_int_l_, bins_)

    if verbose_:
        print('Binning spike train: bin_length_ms {}, bin_length_fs {}'.format(bin_length_ms_, bin_length_fs))
        print('n bins {}, spike bin count: number of spikes in bin - number of bins {}'.format(binned_spike_train.shape,
                                                                                               np.unique(
                                                                                                   binned_spike_train,
                                                                                                   return_counts=True)))
    return binned_spike_train


def calculate_acf_pearson(binned_spike_train_, n_lags_):
    """
    Calculate ACF using Pearson autocorrelation.
    todo: if binned_spike_train is too short for given n_lags?

    :param binned_spike_train_:
    :param n_lags_: int, number of time lags to calculate ACF on
    :return:
    """
    acf_binned = acf(binned_spike_train_, nlags=n_lags_)
    return acf_binned


def calculate_acf_isttc(spike_train_ms_int_l_, n_lags_, resolution_ms_):
    """
    Calculate ACF using iSTTC autocorrelation.

    :param spike_train_ms_int_l_:
    :param n_lags_: int, number of time lags to calculate ACF on
    :param resolution_ms_: int, shift step
    :return:
    """
    shift_ms_l = np.linspace(resolution_ms_+1, resolution_ms_ * n_lags_ + 1, n_lags_).astype(int)
    spike_train_bin = np.zeros(spike_train_ms_int_l_[-1] + 1)
    spike_train_bin[spike_train_ms_int_l_] = 1

    sttc_self_l = []

    # correlate with itself
    spike_train_neo = neo.SpikeTrain(spike_train_ms_int_l_, units='ms', t_start=0, t_stop=len(spike_train_bin))
    sttc_no_shift = spike_time_tiling_coefficient(spike_train_neo, spike_train_neo, dt=resolution_ms_ * pq.ms)
    sttc_self_l.append(sttc_no_shift)

    # correlated shifted signal
    for shift_ms in shift_ms_l:
        spike_train_bin1 = spike_train_bin[:-1 - shift_ms + 1]
        spike_train_bin2 = spike_train_bin[shift_ms:]

        spike_train_bin1_idx = np.nonzero(spike_train_bin1)[0]
        spike_train_bin2_idx = np.nonzero(spike_train_bin2)[0]

        spike_train_neo_1 = neo.SpikeTrain(spike_train_bin1_idx, units='ms', t_start=0, t_stop=len(spike_train_bin1))
        spike_train_neo_2 = neo.SpikeTrain(spike_train_bin2_idx, units='ms', t_start=0, t_stop=len(spike_train_bin2))

        sttc_self = spike_time_tiling_coefficient(spike_train_neo_1, spike_train_neo_2, dt=resolution_ms_ * pq.ms)
        sttc_self_l.append(sttc_self)

    acf_not_binned = np.asarray(sttc_self_l)
    return acf_not_binned


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


def fit_single_exp(ydata_to_fit_, start_idx_=1):
    """
    Fit function func_exp to data using non-linear least square.

    todo check that - important point: Fit is done from the first ACF value (acf[0] is skipped, it is done like this
    in the papers, still not sure)

    :param ydata_to_fit_: 1d array, the dependant data to fit
    :param start_idx_: int, index to start fitting from
    :return: fit_popt, fit_pcov, tau, fit_r_squared
    """
    t = np.linspace(start_idx_, len(ydata_to_fit_), len(ydata_to_fit_)).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            popt, pcov = curve_fit(func_single_exp, t, ydata_to_fit_[start_idx_:], maxfev=5000)
            fit_popt = popt
            fit_pcov = pcov
            tau = 1 / fit_popt[1]
            # fit r-squared
            y_pred = func_single_exp(t, *popt)
            fit_r_squared = r2_score(ydata_to_fit_[start_idx_:], y_pred)
        except RuntimeError as e:
            print('RuntimeError: {}'. format(e))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
        except OptimizeWarning as o:
            print('OptimizeWarning: {}'. format(o))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
        except RuntimeWarning as re:
            print('RuntimeWarning: {}'. format(re))
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan
        except ValueError as ve:
            print('ValueError: {}'. format(ve))
            print('Possible reason: acf contains NaNs, low spike count')
            fit_popt, fit_pcov, tau, fit_r_squared = np.nan, np.nan, np.nan, np.nan

    return fit_popt, fit_pcov, tau, fit_r_squared

