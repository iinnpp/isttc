"""
Functions to calculate autocorrelation function (some tests).
"""
import warnings
import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning

from src.isttc.scripts.calculate_acf import sttc_calculate_t, sttc_fixed_2t


def autocorr_pearsonr(signal_, lag_=1, verbose_=False):
    """
    Correlation between lag_ and lag_-1 using Pearson correlation coefficient.
    :param signal_: numeric, 1d array, binned spike train.
    :param lag_: int, lag to calculate correlation.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise.
    :return: Pearson correlation coefficient, np.nan in case of error.
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


def acf_pearsonr(signal_, n_lags_=2, verbose_=False):
    """
    Autocorrelation function using Pearson correlation coefficient.
    If n_lags_ >= signal_length then n_lags_ = signal_length-2. For example, is the signal_len = 20 then for lag 20
    there are no values to correlate and for the lag 19 there is only 1 value per array. In this case Pearson
    is NaN because denominator is 0. So the last lag to correlate is set to 18.
    ACF len is = n_lags_ + 1 for signal_length > n_lags_ + 1 (as in acf from statsmodels.tsa.stattools) otherwise
    ACF len is = len(signal_) - 1 (if n_lags_ >= len(signal_) or n_lags_ == len(signal_) - 1).
    :param signal_: numeric, 1d array, binned spike train.
    :param n_lags_: int, number of lags.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise.
    :return: 1d array, numeric.
    """
    acf_result = [1]
    if n_lags_ >= len(signal_):
        n_lags = len(signal_) - 2
        if verbose_:
            print('n_lags ({}) is >= signal length ({}). Setting n_lags to {}...'
                  .format(n_lags_, len(signal_), n_lags))
    elif n_lags_ == len(signal_) - 1:
        # case with only 1 value per array
        n_lags = n_lags_ - 1
        if verbose_:
            print('n_lags ({}) is == signal length ({}) - 1. Setting n_lags to {}...'
                  .format(n_lags_, len(signal_), n_lags))
    else:
        n_lags = n_lags_

    for i in range(1, n_lags + 1):
        acf_result.append(autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf_result)


def my_autocorr_pearsonr(signal_, lag_=1, verbose_=False):
    """
    Correlation between lag_ and lag_-1 using Pearson correlation coefficient. Same as autocorr_pearsonr but without
    using pearsonr function. Just a sanity check.
    :param signal_: numeric, 1d array, binned spike train.
    :param lag_: int, lag to calculate correlation.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise.
    :return: Pearson correlation coefficient, np.nan in case of error.
    """
    if verbose_:
        print('Calc for lag {}, input length {}'.format(lag_, signal_.shape))
    denominator = np.sqrt(
        sum((signal_[lag_:] - np.mean(signal_[lag_:])) ** 2) * sum((signal_[:-lag_] - np.mean(signal_[:-lag_])) ** 2))
    numerator_p1 = signal_[lag_:] - np.mean(signal_[lag_:])
    numerator_p2 = signal_[:-lag_] - np.mean(signal_[:-lag_])
    if verbose_:
        print('shape numerator_p1 {}, numerator_p2 {}'.format(numerator_p1.shape, numerator_p2.shape))
    numerator = sum(numerator_p1 * numerator_p2)
    ac_lag = numerator / denominator
    if verbose_:
        print('acf_lag {}'.format(ac_lag))
    return ac_lag


def my_acf_pearsonr(signal_, n_lags_=2, verbose_=False):
    """
    Same as acf_pearsonr but without using pearsonr function. Just a sanity check.
    :param signal_: numeric, 1d array, binned spike train.
    :param n_lags_: int, number of lags.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise.
    :return: 1d array, numeric.
    """
    acf_result = [1]
    if n_lags_ >= len(signal_):
        n_lags = n_lags_ - 2
        if verbose_:
            print('n_lags ({}) is >= signal length ({}). Setting n_lags to {}...'
                  .format(n_lags_, len(signal_), n_lags))
    elif n_lags_ == len(signal_) - 1:
        # case with only 1 value per array
        n_lags = n_lags_ - 1
        if verbose_:
            print('n_lags ({}) is == signal length ({}) - 1. Setting n_lags to {}...'
                  .format(n_lags_, len(signal_), n_lags))
    else:
        n_lags = n_lags_

    for i in range(1, n_lags + 1):
        acf_result.append(my_autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf_result)


def acf_sttc_trial_concat_global(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int,
                          trial_len_: int, zero_padding_len_: int, verbose_: bool = True) -> np.ndarray:
    """
    Autocorrelation calculated on concatenated trials. Trials are concatenated with zero padding. For the time lags
    the trials are shifted and then concatenated again with zero padding.
    T term (absolute time) is calculated for trials, summed up and divided by the signal len(sum of trial len
    without zero padding). T is calculated per time lag (on shifted trials).

    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param n_lags_: int, number of lags
    :param lag_shift_: int, shift for a time lag (in time points)
    :param sttc_dt_: int, dt parameter for STTC calculation
    :param trial_len_: int, len of a trial (in time points). All trials have the same length.
    :param zero_padding_len_: int, len of zero padding (in time points).
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, autocorrelation function.
    """

    def concatenate_trials_with_padding(spike_train_l, padding_len):
        concatenated = np.asarray(spike_train_l[0])
        for idx_spike_train, spike_train in enumerate(spike_train_l[1:], start=1):
            concatenated = np.hstack((concatenated, np.asarray(spike_train) + idx_spike_train * padding_len))
        return concatenated

    def calculate_t_term(spike_train_l, trial_len, dt, verbose):
        abs_time_sum = sum(sttc_calculate_t(spike_train, len(spike_train), dt, 0, trial_len, verbose)[0]
                           for spike_train in spike_train_l)
        return abs_time_sum / (len(spike_train_l) * trial_len)

    if verbose_:
        print('Processing {} trials: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))

    # calculate for the non-shifted signal (lag 0)
    spike_train_concat = concatenate_trials_with_padding(spike_train_l_, zero_padding_len_)
    time_proc_sum = calculate_t_term(spike_train_l_, trial_len_, sttc_dt_, verbose_)
    sttc_no_shift = sttc_fixed_2t(spike_train_concat, spike_train_concat, dt_=sttc_dt_,
                                  t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
    acf_l = [sttc_no_shift]

    # generate signal shifts
    if lag_shift_ * n_lags_ == trial_len_:
        shifts_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shifts_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('Generated lag shifts: {}'.format(shifts_l))

    # calculated for shifted signal
    for shift in shifts_l:
        if verbose_:
            print('Calculating sttc for lag shift {}'.format(shift))
        # Get shifted spike trains
        spike_train_1_shifted_l, spike_train_2_shifted_l = [], []
        for idx, trial in enumerate(spike_train_l_, start=0):
            spike_train_1_shifted_l.append(list(trial[trial >= shift] - shift))
            spike_train_2_shifted_l.append(list(trial[trial < trial_len_ - shift]))
        # Concatenate shifted spike trains
        spike_trial_1_concat = concatenate_trials_with_padding(spike_train_1_shifted_l, zero_padding_len_)
        spike_trial_2_concat = concatenate_trials_with_padding(spike_train_2_shifted_l, zero_padding_len_)
        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_trial_1_concat.shape, spike_trial_2_concat.shape))
            print(spike_trial_1_concat)
            print(spike_trial_2_concat)
        # Calculate T term for sttc
        #time_prop_sum_shift_1 = calculate_t_term(spike_train_1_shifted_l, trial_len_ - shift, sttc_dt_, verbose_)
        #time_prop_sum_shift_2 = calculate_t_term(spike_train_2_shifted_l, trial_len_ - shift, sttc_dt_, verbose_)
        # Calculate sttc for the shifted signals
        isttc = sttc_fixed_2t(spike_trial_1_concat, spike_trial_2_concat, dt_=sttc_dt_,
                              t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


def acf_sttc_trial_concat_global_v2(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int,
                          trial_len_: int, zero_padding_len_: int, verbose_: bool = True) -> np.ndarray:
    """
    Autocorrelation calculated on concatenated trials. Trials are concatenated with zero padding. For the time lags
    the trials are concatenated with zero padding and then shifted.
    T term (absolute time) is calculated for trials, summed up and divided by the signal len(sum of trial len
    without zero padding). T is calculated once for original signal (global) and IS NOT recalculated
    per time lag (on shifted concatenated trials).

    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param n_lags_: int, number of lags
    :param lag_shift_: int, shift for a time lag (in time points)
    :param sttc_dt_: int, dt parameter for STTC calculation
    :param trial_len_: int, len of a trial (in time points). All trials have the same length.
    :param zero_padding_len_: int, len of zero padding (in time points).
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, autocorrelation function.
    """

    def concatenate_trials_with_padding(spike_train_l, padding_len):
        concatenated = np.asarray(spike_train_l[0])
        for idx_spike_train, spike_train in enumerate(spike_train_l[1:], start=1):
            concatenated = np.hstack((concatenated, np.asarray(spike_train) + idx_spike_train * padding_len))
        return concatenated

    def calculate_t_term(spike_train_l, trial_len, dt, verbose):
        abs_time_sum = sum(sttc_calculate_t(spike_train, len(spike_train), dt, 0, trial_len, verbose)[0]
                           for spike_train in spike_train_l)
        return abs_time_sum / (len(spike_train_l) * trial_len)

    if verbose_:
        print('Processing {} trials: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))

    # calculate for the non-shifted signal (lag 0)
    spike_train_concat = concatenate_trials_with_padding(spike_train_l_, zero_padding_len_)
    time_proc_sum = calculate_t_term(spike_train_l_, trial_len_, sttc_dt_, verbose_)
    sttc_no_shift = sttc_fixed_2t(spike_train_concat, spike_train_concat, dt_=sttc_dt_,
                                  t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
    acf_l = [sttc_no_shift]
    if verbose_:
        print('T term for the concatenated signal time_proc_sum {}'.format(time_proc_sum))

    # generate signal shifts
    if lag_shift_ * n_lags_ == trial_len_:
        shifts_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shifts_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('Generated lag shifts: {}'.format(shifts_l))

    # calculated for shifted signal
    signal_length = trial_len_ * len(spike_train_l_) + (zero_padding_len_ - trial_len_)*(len(spike_train_l_) - 1)
    if verbose_:
        print('Concatenated signal length {}'.format(signal_length))
    for shift in shifts_l:
        if verbose_:
            print('Calculating sttc for lag shift {}'.format(shift))
        # Get shifted spike trains
        spike_1 = spike_train_concat[spike_train_concat >= shift]
        spike_2 = spike_train_concat[spike_train_concat < signal_length - shift]
        # align, only 1st
        spike_1_aligned = spike_1 - shift
        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_1_aligned.shape, spike_2.shape))
            print('spike_1 {}, spike_2 {}'.format(spike_1_aligned, spike_2))
        isttc = sttc_fixed_2t(spike_1_aligned, spike_2, dt_=sttc_dt_,
                              t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


# todo add calculating T terms without 0s
def deprecated_acf_sttc_trial_avg(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int,
                                  zero_padding_len_: int, verbose_: bool = True):
    """
    Trial average autocorrelation using STTC.
    :param sttc_dt_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param n_lags_: int, number of lags
    :param lag_shift_:
    :param zero_padding_len_:
    :param verbose_:
    :return:
    """
    if verbose_:
        print('Processing {} trials: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))
    acf_matrix = np.zeros((n_lags_, n_lags_))

    t_start = 0
    t_stop = (len(spike_train_l_) - 1) * zero_padding_len_ + lag_shift_
    if verbose_:
        print(t_start, t_stop, len(spike_train_l_))

    for i in np.arange(n_lags_ - 1):
        for j in np.arange(i + 1, n_lags_):  # filling i-th row
            lag_1_spikes_l, lag_2_spikes_l = get_lag_arrays(spike_train_l_, i, j,
                                                            lag_shift_=lag_shift_, zero_padding_len_=zero_padding_len_)
            l1_aligned = [spike - lag_shift_ * i for spike in lag_1_spikes_l]
            l2_aligned = [spike - lag_shift_ * j for spike in lag_2_spikes_l]
            sttc_lag = sttc(l1_aligned, l2_aligned, t_start, t_stop, sttc_dt_, verbose_=verbose_)
            acf_matrix[i, j] = sttc_lag
    np.fill_diagonal(acf_matrix, 1)

    acf_average = np.zeros((n_lags_,))
    for i in range(n_lags_):
        acf_average[i] = np.nanmean(np.diag(acf_matrix, k=i))

    return acf_matrix, acf_average


def deprecated_acf_sttc_trial_avg_v2(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int,
                                     zero_padding_len_: int, verbose_: bool = True):
    """
    Trial average autocorrelation using STTC. T term is calculated on lag signal without padding (lag_shift padding is
    used to concat all trials in 1d array).

    :param sttc_dt_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param n_lags_: int, number of lags
    :param lag_shift_:
    :param zero_padding_len_:
    :param verbose_:
    :return:
    """
    if verbose_:
        print('Processing {} trials: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))
    acf_matrix = np.zeros((n_lags_, n_lags_))

    t_start = 0
    t_stop = (len(spike_train_l_) - 1) * zero_padding_len_ + lag_shift_
    if verbose_:
        print(t_start, t_stop, len(spike_train_l_))

    for i in np.arange(n_lags_ - 1):
        for j in np.arange(i + 1, n_lags_):  # filling i-th row
            # get arrays for T term calculation - without zero padding
            lag_1_spikes_t_l, lag_2_spikes_t_l = get_lag_arrays(spike_train_l_, i, j,
                                                            lag_shift_=lag_shift_, zero_padding_len_=lag_shift_)
            l1_aligned_t = [spike - lag_shift_ * i for spike in lag_1_spikes_t_l]
            l2_aligned_t = [spike - lag_shift_ * j for spike in lag_2_spikes_t_l]
            l1_t = sttc_calculate_t(l1_aligned_t, len(l1_aligned_t), sttc_dt_, 0, lag_shift_*len(spike_train_l_), verbose_)[1]
            l2_t = sttc_calculate_t(l2_aligned_t, len(l2_aligned_t), sttc_dt_, 0, lag_shift_*len(spike_train_l_), verbose_)[1]
            if verbose_:
                print('l1_t {}, l2_t {}'.format(l1_t, l2_t))

            # get arrays for sttc - with zero padding
            lag_1_spikes_l, lag_2_spikes_l = get_lag_arrays(spike_train_l_, i, j,
                                                            lag_shift_=lag_shift_, zero_padding_len_=zero_padding_len_)
            l1_aligned = [spike - lag_shift_ * i for spike in lag_1_spikes_l]
            l2_aligned = [spike - lag_shift_ * j for spike in lag_2_spikes_l]
            sttc_lag = sttc_fixed_2t(l1_aligned, l2_aligned, sttc_dt_, t_a_=l1_t, t_b_=l2_t, verbose_=verbose_)
            acf_matrix[i, j] = sttc_lag

    np.fill_diagonal(acf_matrix, 1)

    acf_average = np.zeros((n_lags_,))
    for i in range(n_lags_):
        acf_average[i] = np.nanmean(np.diag(acf_matrix, k=i))

    return acf_matrix, acf_average

