"""
Functions to calculate autocorrelation function:
* using ACF equation,
* using Pearson equation,
* using iSTTC,
* trail average ACF using Pearson correlation (see monkey papers),
* trial average ACF using STTC.
"""
import warnings

import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning
from statsmodels.tsa.stattools import acf


def autocorr_pearsonr(signal_, lag_=1, verbose_=True):
    """
    Correlation between lag_ and lag_-1.
    :param signal_:
    :param lag_:
    :param verbose_:
    :return:
    """
    if verbose_:
        print('Calc for lag {}, input length {}'.format(lag_, signal_.shape))
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            r, p = pearsonr(signal_[lag_:], signal_[:-lag_])
        except ConstantInputWarning as e:
            print('ERROR: Pearson r can not be calculated for lag {}, error {}'.format(lag_, e))
            r = np.nan
    if verbose_:
        print('Pearson correlation for lag {}: {}'.format(lag_, r))
    return r


def acf_pearsonr(signal_, n_lags_=2, verbose_=True):
    """
    Autocorrelation using Pearson correlation.
    :param signal_:
    :param n_lags_:
    :param verbose_:
    :return:
    """
    acf = [1]
    for i in range(1, n_lags_):
        acf.append(autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf)


def acf_pearsonr_trial_avg(trials_time_series_2d, n_lags_, verbose_=True):
    """
    Trial average autocorrelation using Pearson coefficient.
    :param trials_time_series_2d:
    :param n_lags_:
    :param verbose_:
    :return:
    """
    time_series_a = trials_time_series_2d[:, :n_lags_]
    n_bins = time_series_a.shape[1]
    if verbose_:
        print('n_bins to use {}'.format(n_bins))

    acf_matrix = np.zeros((n_bins, n_bins))
    for i in np.arange(n_bins - 1):
        for j in np.arange(i + 1, n_bins):  # filling i-th row
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    r, p = pearsonr(time_series_a[:, i], time_series_a[:, j])
                    acf_matrix[i, j] = r
                except ConstantInputWarning as e:
                    print('ERROR: Pearson r can not be calculated for i={}, j={}, error {}'.format(i, j, e))
                    acf_matrix[i, j] = np.nan
    np.fill_diagonal(acf_matrix, 1)

    acf_average = np.zeros((n_bins,))
    for i in range(n_bins):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                acf_average[i] = np.nanmean(np.diag(acf_matrix, k=i))
            except RuntimeWarning as e:
                print('ERROR: ACF average (mean) can not be calculated for lag={}'.format(i), e)
                acf_average[i] = np.nan

    return acf_matrix, acf_average


def run_t(spiketrain_, n_spikes_, dt_, t_start_, t_stop_):
    """
    Calculate the proportion of the total recording time 'tiled' by spikes.
    """
    time_abs = 2 * n_spikes_ * dt_  # maximum possible time

    if n_spikes_ == 1:  # for just one spike in train
        if spiketrain_[0] - t_start_ < dt_:
            time_abs = time_abs - dt_ + spiketrain_[0] - t_start_
        elif spiketrain_[0] + dt_ > t_stop_:
            time_abs = time_abs - dt_ - spiketrain_[0] + t_stop_

    else:  # if more than one spike in train
        diff = np.diff(spiketrain_)
        idx = np.where(diff < (2 * dt_))[0]
        idx_len = len(idx)
        time_abs = time_abs - 2 * idx_len * dt_ + diff[idx].sum()

        # todo this part and the same part above
        if (spiketrain_[0] - t_start_) < dt_:
            time_abs = time_abs + spiketrain_[0] - dt_ - t_start_
        if (t_stop_ - spiketrain_[n_spikes_ - 1]) < dt_:
            time_abs = time_abs - spiketrain_[-1] - dt_ + t_stop_

    time_prop = (time_abs / (t_stop_ - t_start_))  # .item()

    return time_abs, time_prop


def run_p(spiketrain_1_, spiketrain_2_, n_spikes_1_, n_spikes_2_, dt_):
    """
    Check every spike in train 1 to see if there's a spike in train 2 within dt
    """
    n_tiled_spikes = 0
    j = 0
    for i in range(n_spikes_1_):
        k = 0
        while j < n_spikes_2_:  # don't need to search all j each iteration
            if np.abs(spiketrain_1_[i] - spiketrain_2_[j]) <= dt_:
                n_tiled_spikes = n_tiled_spikes + 1
                k += 1
                break
            elif spiketrain_2_[j] > spiketrain_1_[i]:
                break
            else:
                j = j + 1
    return n_tiled_spikes


def calc_sttc(spiketrain_1_l_, spiketrain_2_l_, t_start_, t_stop_, dt_, verbose_=True):
    n_a = len(spiketrain_1_l_)
    n_b = len(spiketrain_2_l_)

    if n_a == 0 or n_b == 0:
        sttc = 0
    else:
        time_a, t_a = run_t(spiketrain_1_l_, n_a, dt_, t_start_, t_stop_)
        # print('time_a {}, t_a {}'.format(time_a, t_a))

        time_b, t_b = run_t(spiketrain_2_l_, n_b, dt_, t_start_, t_stop_)
        # print('time_b {}, t_b {}'.format(time_b, t_b))

        p_a_count = run_p(spiketrain_1_l_, spiketrain_2_l_, n_a, n_b, dt_)
        p_a = p_a_count / float(n_a)
        # print('p_a_count {}, p_a {}'.format(p_a_count, p_a))

        p_b_count = run_p(spiketrain_2_l_, spiketrain_1_l_, n_b, n_a, dt_)
        p_b = p_b_count / float(n_b)
        # print('p_b_count {}, p_b {}'.format(p_b_count, p_b))

        sttc = 0.5 * (p_a - t_b) / (1 - p_a * t_b) + 0.5 * (p_b - t_a) / (1 - p_b * t_a)
        if verbose_:
            print('STTC : {}'.format(sttc))
    return sttc


def get_lag_arrays(spike_train_l_, lag_1_idx_, lag_2_idx_, lag_shift_, zero_padding_len_):
    """

    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trails
    :param lag_1_idx_:
    :param lag_2_idx_:
    :param lag_shift_:
    :param zero_padding_len_:
    :return:
    """
    # select spikes for lags
    first_lag_l = []
    second_lag_l = []
    for i in range(len(spike_train_l_)):  # for all trials
        spike_train = spike_train_l_[i]
        first_lag = spike_train[
            (spike_train[:] > lag_1_idx_ * lag_shift_) & (spike_train[:] <= lag_1_idx_ * lag_shift_ + lag_shift_)]
        first_lag_l.append(first_lag)

        second_lag = spike_train[
            (spike_train[:] > lag_2_idx_ * lag_shift_) & (spike_train[:] <= lag_2_idx_ * lag_shift_ + lag_shift_)]
        second_lag_l.append(second_lag)

    # add padding zeros
    first_lag_l_spaced = []
    for i in range(0, len(spike_train_l_)):
        first_lag_l_spaced.append(first_lag_l[i] + i * zero_padding_len_)

    second_lag_l_spaced = []
    for i in range(len(spike_train_l_)):
        second_lag_l_spaced.append(second_lag_l[i] + i * zero_padding_len_)

    # reshape in 1d arrays of spike times
    lag1_l = []
    for i in range(0, len(spike_train_l_)):
        if len(first_lag_l_spaced[i]) > 0:
            # print(first_lag_l_spaced[i])
            if len(first_lag_l_spaced[i]) == 1:
                lag1_l.append(np.squeeze(first_lag_l_spaced[i]).tolist())
            else:
                n_spikes = len(first_lag_l_spaced[i])
                for j in range(n_spikes):
                    lag1_l.append(np.squeeze(first_lag_l_spaced[i][j]).tolist())

    lag2_l = []
    for i in range(len(spike_train_l_)):
        if len(second_lag_l_spaced[i]) > 0:
            # print(second_lag_l_spaced[i])
            if len(second_lag_l_spaced[i]) == 1:
                lag2_l.append(np.squeeze(second_lag_l_spaced[i]).tolist())
            else:
                n_spikes = len(second_lag_l_spaced[i])
                for j in range(n_spikes):
                    lag2_l.append(np.squeeze(second_lag_l_spaced[i][j]).tolist())

    return lag1_l, lag2_l


def acf_sttc_trial_avg(spike_train_l_, lag_shift_=50, zero_padding_len_=150, fs_=1000, sttc_dt_=25, verbose_=True):
    """
    Trial average autocorrelation using STTC.
    :param sttc_dt_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trails
    :param lag_shift_:
    :param zero_padding_len_:
    :param fs_:
    :param verbose_:
    :return:
    """
    #lag_shift = 50
    #padding = 150

    n_bins = int(fs_ / lag_shift_)
    if verbose_:
        print('n_bins to use {}'.format(n_bins))
    acf_matrix = np.zeros((n_bins, n_bins))

    t_start = 0
    t_stop = (len(spike_train_l_) - 1) * zero_padding_len_ + lag_shift_
    if verbose_:
        print(t_start, t_stop, len(spike_train_l_))

    for i in np.arange(n_bins - 1):
        for j in np.arange(i + 1, n_bins):  # filling i-th row
            # print(i,j)
            lag_1_spikes_l, lag_2_spikes_l = get_lag_arrays(spike_train_l_, i, j,
                                                            lag_shift_=lag_shift_, zero_padding_len_=zero_padding_len_)
            # print(lag_1_spikes_l)
            # print(lag_2_spikes_l)
            l1_aligned = [spike - lag_shift_ * i for spike in lag_1_spikes_l]
            l2_aligned = [spike - lag_shift_ * j for spike in lag_2_spikes_l]
            # t_start = i*lag_shift
            # t_stop = (len(v)-1)*padding + (j+1)*lag_shift
            # print(t_start, t_stop, t_stop-t_start)
            sttc_lag = calc_sttc(l1_aligned, l2_aligned, t_start, t_stop, sttc_dt_)
            acf_matrix[i, j] = sttc_lag
    np.fill_diagonal(acf_matrix, 1)

    acf_average = np.zeros((n_bins,))
    for i in range(n_bins):
        acf_average[i] = np.nanmean(np.diag(acf_matrix, k=i))

    return acf_matrix, acf_average


# todo
def acf_sttc(signal_, n_lags_=2, verbose_=True):
    """
    Autocorrelation using STTC.
    :param signal_:
    :param n_lags_:
    :param verbose_:
    :return:
    """
    acf = [1]
#    for i in range(1, n_lags_):
        # acf.append(autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf)