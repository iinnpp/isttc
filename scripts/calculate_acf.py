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
    Correlation between lag_ and lag_-1 using Pearson correlation coefficient.
    :param signal_: numeric, 1d array.
    :param lag_: int, lag to calculate correlation
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: Pearson correlation coefficient, np.nan in case of error
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
    Autocorrelation function using Pearson correlation coefficient.
    If n_lags_ >= signal_length then n_lags_ = signal_length-1. For example, is the signal_len = 20 then for lag 20
    there are no values to correlate and for the lag 19 there is only 1 value per array. In this case Pearson
    is NaN because denominator is 0.
    ACF len is = n_lags_ + 1 for signal_length > n_lags_ + 1 (as in acf from statsmodels.tsa.stattools) otherwise
    ACF len is = n_lags_(n_lags_ == len(signal_) - 1) or ACF len is = n_lags_-1 (n_lags_ >= len(signal_)).
    :param signal_: numeric, 1d array.
    :param n_lags_: int, number of lags.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, numeric. Array len is = n_lags_ + 1 for
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

    for i in range(1, n_lags+1):
        acf_result.append(autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf_result)


def my_autocorr_pearsonr(signal_, lag_=1, verbose_=True):
    """
    Same as autocorr_pearsonr but without using pearsonr function. Just a sanity check.
    :param signal_:
    :param lag_:
    :param verbose_:
    :return:
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


def my_acf_pearsonr(signal_, n_lags_=2, verbose_=True):
    """
    Same as acf_pearsonr but without using pearsonr function. Just a sanity check.
    :param signal_:
    :param n_lags_:
    :param verbose_:
    :return:
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

    for i in range(1, n_lags+1):
        acf_result.append(my_autocorr_pearsonr(signal_, i, verbose_))
    return np.array(acf_result)


def my_autocorr(signal_, lag_=1, verbose_=True):
    """
    Same as acf (for 1 lag) but without using acf function. Just a sanity check.
    :param signal_:
    :param lag_:
    :param verbose_:
    :return:
    """
    if verbose_:
        print('Calc for lag {}, input length {}'.format(lag_, signal_.shape))

    signal_mean = np.mean(signal_)
    denominator = sum((signal_ - signal_mean) ** 2)
    if verbose_:
        print('y_mean = {}, denominator = {}'.format(signal_mean, denominator))
    numerator_p1 = signal_[lag_:] - signal_mean
    numerator_p2 = signal_[:-lag_] - signal_mean
    if verbose_:
        print('shape numerator_p1 {}, numerator_p2 {}'.format(numerator_p1.shape, numerator_p2.shape))
    numerator = sum(numerator_p1 * numerator_p2)
    ac_lag = numerator / denominator
    if verbose_:
        print('acf_lag {}'.format(ac_lag))
    return ac_lag


def my_acf(signal_, n_lags_=2, verbose_=True):
    """
    Same as acf but without using acf function. Just a sanity check.
    :param signal_:
    :param n_lags_:
    :param verbose_:
    :return:
    """
    acf_result = [1]
    if n_lags_ >= len(signal_):
        n_lags = n_lags_ - 1
        if verbose_:
            print('n_lags ({}) is >= signal length ({}). Setting n_lags to {}...'
                  .format(n_lags_, len(signal_), n_lags))
    else:
        n_lags = n_lags_

    for i in range(1, (n_lags+1)):
        acf_result.append(my_autocorr(signal_, i, verbose_))
    return np.array(acf_result)


def acf_pearsonr_trial_avg(trials_time_series_2d, n_lags_, verbose_=True):
    """
    Trial average autocorrelation using Pearson coefficient (inspired by Murray et al 2014).
    :param trials_time_series_2d: numeric, n_trial x n_bins
    :param n_lags_: int, number of lags
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
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


def sttc_calculate_t(spiketrain_, n_spikes_, dt_, t_start_, t_stop_, verbose_=True):
    """
    Calculate the proportion of the total recording time 'tiled' by spikes.
    """
    if n_spikes_ == 0:
        return 0, 0

    # Maximum possible time
    time_abs = 2 * n_spikes_ * dt_
    if verbose_:
        print(f'Initial time_abs: {time_abs}')

    # Handle single spike case
    if n_spikes_ == 1:
        # Adjust for the first spike
        if spiketrain_[0] - t_start_ < dt_:
            time_abs += spiketrain_[0] - dt_ - t_start_
        # Adjust for the last spike
        if t_stop_ - spiketrain_[-1] < dt_:
            time_abs = time_abs - dt_ - spiketrain_[0] + t_stop_
            time_abs += t_stop_ - (spiketrain_[-1] + dt_)

    else:  # Multiple spikes case
        diff = np.diff(spiketrain_)
        idx = diff < (2 * dt_)
        time_abs -= 2 * idx.sum() * dt_ - diff[idx].sum()
        # Adjust for the first spike
        if spiketrain_[0] - t_start_ < dt_:
            time_abs += spiketrain_[0] - dt_ - t_start_
        # Adjust for the last spike
        if t_stop_ - spiketrain_[-1] < dt_:
            time_abs += t_stop_ - (spiketrain_[-1] + dt_)

    # Calculate the proportion of the total recording time
    time_prop = time_abs / (t_stop_ - t_start_)
    if verbose_:
        print(f'Total time: {t_stop_ - t_start_}, Time proportion: {time_prop}')

    return time_abs, time_prop


def sttc_calculate_p(spiketrain_1_, spiketrain_2_, n_spikes_1_, n_spikes_2_, dt_):
    """
    Check every spike in train 1 to see if there's a spike in train 2 within dt.
    """
    n_tiled_spikes = 0
    j = 0

    for i in range(n_spikes_1_):
        while j < n_spikes_2_:
            time_diff = spiketrain_1_[i] - spiketrain_2_[j]

            if np.abs(time_diff) <= dt_:  # Spike in train 2 within dt of train 1
                n_tiled_spikes += 1
                break
            elif time_diff < 0:  # Current spike in train 2 is too late
                break
            else:  # Current spike in train 2 is too early; move to next spike
                j += 1

    return n_tiled_spikes


def sttc(spiketrain_1_l_, spiketrain_2_l_, t_start_, t_stop_, dt_, verbose_=True):
    """
    Calculate the Spike Time Tiling Coefficient (STTC) for two spike trains.
    """
    n_a, n_b = len(spiketrain_1_l_), len(spiketrain_2_l_)

    # Handle cases where one or both spike trains are empty
    if n_a == 0 or n_b == 0:
        return 0

    # Calculate tiling and proportion times for both spike trains
    time_a, t_a = sttc_calculate_t(spiketrain_1_l_, n_a, dt_, t_start_, t_stop_, verbose_)
    time_b, t_b = sttc_calculate_t(spiketrain_2_l_, n_b, dt_, t_start_, t_stop_, verbose_)

    # Calculate proportions of tiled spikes
    p_a = sttc_calculate_p(spiketrain_1_l_, spiketrain_2_l_, n_a, n_b, dt_) / n_a
    p_b = sttc_calculate_p(spiketrain_2_l_, spiketrain_1_l_, n_b, n_a, dt_) / n_b

    # Compute STTC result
    if t_a * p_b == 1 and t_b * p_a == 1:
        sttc_result = 1
    elif t_a * p_b == 1:
        sttc_result = 0.5 * (p_a - t_b) / (1 - p_a * t_b) + 0.5
    elif t_b * p_a == 1:
        sttc_result = 0.5 + 0.5 * (p_b - t_a) / (1 - p_b * t_a)
    else:
        sttc_result = 0.5 * (p_a - t_b) / (1 - p_a * t_b) + 0.5 * (p_b - t_a) / (1 - p_b * t_a)

    if verbose_:
        print(f'STTC: {sttc_result}')

    return sttc_result


def sttc_fixed_2t(spiketrain_1_l_, spiketrain_2_l_, dt_, t_a_, t_b_, verbose_=True):
    """
    Calculate the Spike Time Tiling Coefficient (STTC) for two spike trains with fixed t_a and t_b.
    """
    n_a, n_b = len(spiketrain_1_l_), len(spiketrain_2_l_)

    # Return 0 if either of the spike trains is empty
    if n_a == 0 or n_b == 0:
        return 0

    # Calculate the proportions of tiled spikes
    p_a = sttc_calculate_p(spiketrain_1_l_, spiketrain_2_l_, n_a, n_b, dt_) / n_a
    p_b = sttc_calculate_p(spiketrain_2_l_, spiketrain_1_l_, n_b, n_a, dt_) / n_b

    # Compute STTC result based on t_a, t_b, and p_a, p_b
    if t_a_ * p_b == 1 and t_b_ * p_a == 1:
        sttc_result = 1
    elif t_a_ * p_b == 1:
        sttc_result = 0.5 * (p_a - t_b_) / (1 - p_a * t_b_) + 0.5
    elif t_b_ * p_a == 1:
        sttc_result = 0.5 + 0.5 * (p_b - t_a_) / (1 - p_b * t_a_)
    else:
        sttc_result = 0.5 * (p_a - t_b_) / (1 - p_a * t_b_) + 0.5 * (p_b - t_a_) / (1 - p_b * t_a_)

    if verbose_:
        print(f'STTC: {sttc_result}')

    return sttc_result


def acf_sttc(signal_, n_lags_, lag_shift_, sttc_dt_, signal_length_, verbose_=True):
    """
    Autocorrelation function using STTC.
    :param signal_:
    :param n_lags_:
    :param lag_shift_:
    :param sttc_dt_:
    :param signal_length_:
    :param verbose_:
    :return:
    """
    # Determine the shift values for the autocorrelation calculation
    if lag_shift_ * n_lags_ == signal_length_:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('shift_ms_l {}'.format(shift_ms_l))

    # Calculate the STTC for the original (unshifted) signal
    sttc_no_shift = sttc(signal_, signal_, t_start_=0, t_stop_=signal_length_, dt_=sttc_dt_)
    acf_l = [sttc_no_shift]

    # Iterate through each shift and calculate the STTC for shifted signals
    for shift_ms in shift_ms_l:
        spike_1 = signal_[signal_ >= shift_ms]
        spike_2 = signal_[signal_ < signal_length_ - shift_ms]
        # align, only 1st
        spike_1_aligned = spike_1 - shift_ms
        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_1.shape, spike_2.shape))
        isttc = sttc(spike_1_aligned, spike_2, t_start_=0, t_stop_=signal_length_ - shift_ms, dt_=sttc_dt_, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


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


#todo add calculating T terms without 0s
def acf_sttc_trial_avg(spike_train_l_, lag_shift_, zero_padding_len_, fs_, sttc_dt_, verbose_=True):
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
            lag_1_spikes_l, lag_2_spikes_l = get_lag_arrays(spike_train_l_, i, j,
                                                            lag_shift_=lag_shift_, zero_padding_len_=zero_padding_len_)
            l1_aligned = [spike - lag_shift_ * i for spike in lag_1_spikes_l]
            l2_aligned = [spike - lag_shift_ * j for spike in lag_2_spikes_l]
            sttc_lag = sttc(l1_aligned, l2_aligned, t_start, t_stop, sttc_dt_, verbose_=verbose_)
            acf_matrix[i, j] = sttc_lag
    np.fill_diagonal(acf_matrix, 1)

    acf_average = np.zeros((n_bins,))
    for i in range(n_bins):
        acf_average[i] = np.nanmean(np.diag(acf_matrix, k=i))

    return acf_matrix, acf_average


def acf_sttc_trail_concat(spike_train_l_, n_lags_, lag_shift_, sttc_dt_, trial_len_, zero_padding_len_, verbose_=True):
    """
    Autocorrelation calculated on concatenated trials. Trials are concatenated with zero padding.
    :param trial_len_:
    :param verbose_:
    :param zero_padding_len_:
    :param sttc_dt_:
    :param lag_shift_:
    :param n_lags_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trails. Spike times are realigned (each trial starts at time 0).
    :return:
    """
    if verbose_:
        print('Processing {} trails: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))

    # concatenate trials in 1d signal
    spike_train_concat = spike_train_l_[0]
    for idx, trial in enumerate(spike_train_l_[1:]):
        spike_train_concat = np.hstack((spike_train_concat, trial + (idx + 1) * zero_padding_len_))
    signal_len_concat = len(spike_train_l_) * trial_len_ + (len(spike_train_l_) - 1) * (zero_padding_len_ - trial_len_)
    if verbose_:
        print('Length of concat signal {}'.format(signal_len_concat))

    # calculate T term for the sttc
    # T term is calculated based on trials without zero padding and is the same for all time lags
    if verbose_:
        print('Calculate T term for sttc...')
    time_abs_sum = 0
    for spike_trial in spike_train_l_:
        time_abs_trial, time_prop_trial = sttc_calculate_t(spike_trial, len(spike_trial), sttc_dt_,
                                                           t_start_=0, t_stop_=trial_len_, verbose_=verbose_)
        if verbose_:
            print('Spike train: {} \ntime_abs {}, time_proc {}'.format(spike_trial, time_abs_trial, time_prop_trial))
        time_abs_sum = time_abs_sum + time_abs_trial
    time_proc_sum = time_abs_sum / (len(spike_train_l_) * trial_len_)
    if verbose_:
        print('Calculated T term for sttc: time_abs_sum {}, time_proc_sum {}'.format(time_abs_sum, time_proc_sum))

    # generate signal shifts
    if lag_shift_ * n_lags_ == signal_len_concat:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('Generated lag shifts: {}'.format(shift_ms_l))

    acf_l = []
    sttc_no_shift = sttc_fixed_2t(spike_train_concat, spike_train_concat, t_start_=0, t_stop_=signal_len_concat,
                                 dt_=sttc_dt_, t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
    acf_l.append(sttc_no_shift)

    # correlated shifted signal
    for shift_ms in shift_ms_l:
        spike_1 = spike_train_concat[spike_train_concat >= shift_ms]
        spike_2 = spike_train_concat[spike_train_concat < signal_len_concat - shift_ms]
        # align, only 1st
        spike_1_aligned = [spike - shift_ms for spike in spike_1]
        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_1.shape, spike_2.shape))
        isttc = sttc_fixed_2t(spike_1_aligned, spike_2, t_start_=0, t_stop_=signal_len_concat - shift_ms, dt_=sttc_dt_,
                             t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


def acf_sttc_trail_concat_v2(spike_train_l_, n_lags_, lag_shift_, sttc_dt_, trial_len_, zero_padding_len_, verbose_=True):
    """
    Autocorrelation calculated on concatenated trials. Trials are concatenated with zero padding.
    :param trial_len_:
    :param verbose_:
    :param zero_padding_len_:
    :param sttc_dt_:
    :param lag_shift_:
    :param n_lags_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trails. Spike times are realigned (each trial starts at time 0).
    :return:
    """
    if verbose_:
        print('Processing {} trails: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))

    # concatenate trials in 1d signal
    spike_train_concat = spike_train_l_[0]
    for idx, trial in enumerate(spike_train_l_[1:]):
        spike_train_concat = np.hstack((spike_train_concat, trial + (idx + 1) * zero_padding_len_))
    signal_len_concat = len(spike_train_l_) * trial_len_ + (len(spike_train_l_) - 1) * (zero_padding_len_ - trial_len_)
    if verbose_:
        print('Length of concat signal {}'.format(signal_len_concat))

    # calculate T term for the sttc
    # T term is calculated based on trials without zero padding and is the same for all time lags
    if verbose_:
        print('Calculate T term for sttc...')
    time_abs_sum = 0
    for spike_trial in spike_train_l_:
        time_abs_trial, time_prop_trial = sttc_calculate_t(spike_trial, len(spike_trial), sttc_dt_,
                                                           t_start_=0, t_stop_=trial_len_, verbose_=verbose_)
        if verbose_:
            print('Spike train: {} \ntime_abs {}, time_proc {}'.format(spike_trial, time_abs_trial, time_prop_trial))
        time_abs_sum = time_abs_sum + time_abs_trial
    time_proc_sum = time_abs_sum / (len(spike_train_l_) * trial_len_)
    if verbose_:
        print('Calculated T term for sttc: time_abs_sum {}, time_proc_sum {}'.format(time_abs_sum, time_proc_sum))

    # generate signal shifts
    if lag_shift_ * n_lags_ == trial_len_:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('Generated lag shifts: {}'.format(shift_ms_l))

    acf_l = []
    sttc_no_shift = sttc_fixed_2t(spike_train_concat, spike_train_concat, t_start_=0, t_stop_=signal_len_concat,
                                 dt_=sttc_dt_, t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
    acf_l.append(sttc_no_shift)

    # correlated shifted signal
    for shift_ms in shift_ms_l:

        #print(spike_train_l_[0])
        spike_trial_1_concat_ = list(spike_train_l_[0][spike_train_l_[0] >= shift_ms])
        spike_trial_2_concat = list(spike_train_l_[0][spike_train_l_[0] < trial_len_ - shift_ms])
        # align, only 1st
        spike_trial_1_concat = [spike - shift_ms for spike in spike_trial_1_concat_]
        #print(spike_trial_1_concat, spike_trial_2_concat)

        for idx, trial in enumerate(spike_train_l_[1:]):
            spike_trial_1 = list(trial[trial >= shift_ms])
            spike_trial_2 = list(trial[trial < trial_len_ - shift_ms])
            # align, only 1st
            spike_trial_1_aligned = [spike - shift_ms for spike in spike_trial_1]
            spike_trial_1_concat = np.hstack((spike_trial_1_concat,
                                              np.asarray(spike_trial_1_aligned) + (idx + 1) * zero_padding_len_))
            spike_trial_2_concat = np.hstack((spike_trial_2_concat, np.asarray(spike_trial_2) + (idx + 1) * zero_padding_len_))

        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_trial_1_concat.shape, spike_trial_2_concat.shape))
            print(spike_trial_1_concat)
            print(spike_trial_2_concat)
        isttc = sttc_fixed_2t(spike_trial_1_concat, spike_trial_2_concat, t_start_=0,
                             t_stop_=signal_len_concat - shift_ms*len(spike_train_l_), dt_=sttc_dt_,
                             t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


def acf_sttc_trail_concat_v3(spike_train_l_, n_lags_, lag_shift_, sttc_dt_, trial_len_, zero_padding_len_, verbose_=True):
    """
    Autocorrelation calculated on concatenated trials. Trials are concatenated with zero padding.T term is calculated based
    on non-padded trails for every lag shift.
    :param trial_len_:
    :param verbose_:
    :param zero_padding_len_:
    :param sttc_dt_:
    :param lag_shift_:
    :param n_lags_:
    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trails. Spike times are realigned (each trial starts at time 0).
    :return:
    """
    if verbose_:
        print('Processing {} trails: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))

    # concatenate trials in 1d signal
    spike_train_concat = spike_train_l_[0]
    for idx, trial in enumerate(spike_train_l_[1:]):
        spike_train_concat = np.hstack((spike_train_concat, trial + (idx + 1) * zero_padding_len_))
    signal_len_concat = len(spike_train_l_) * trial_len_ + (len(spike_train_l_) - 1) * (zero_padding_len_ - trial_len_)
    if verbose_:
        print('Length of concat signal {}'.format(signal_len_concat))

    # calculate T term for the sttc
    # T term is calculated based on trials without zero padding and is the same for all time lags
    if verbose_:
        print('Calculate T term for sttc...')
    time_abs_sum = 0
    for spike_trial in spike_train_l_:
        time_abs_trial, time_prop_trial = sttc_calculate_t(spike_trial, len(spike_trial), sttc_dt_,
                                                           t_start_=0, t_stop_=trial_len_, verbose_=verbose_)
        if verbose_:
            print('Spike train: {} \ntime_abs {}, time_proc {}'.format(spike_trial, time_abs_trial, time_prop_trial))
        time_abs_sum = time_abs_sum + time_abs_trial
    time_proc_sum = time_abs_sum / (len(spike_train_l_) * trial_len_)
    if verbose_:
        print('Calculated T term for sttc: time_abs_sum {}, time_proc_sum {}'.format(time_abs_sum, time_proc_sum))

    # generate signal shifts
    if lag_shift_ * n_lags_ == trial_len_:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shift_ms_l = np.linspace(lag_shift_, lag_shift_ * n_lags_, n_lags_).astype(int)
    if verbose_:
        print('Generated lag shifts: {}'.format(shift_ms_l))

    acf_l = []
    sttc_no_shift = sttc_fixed_2t(spike_train_concat, spike_train_concat, t_start_=0, t_stop_=signal_len_concat,
                                 dt_=sttc_dt_, t_a_=time_proc_sum, t_b_=time_proc_sum, verbose_=verbose_)
    acf_l.append(sttc_no_shift)

    # correlated shifted signal
    for shift_ms in shift_ms_l:
        # print(shift_ms)
        spike_trial_1_concat_ = list(spike_train_l_[0][spike_train_l_[0] >= shift_ms])
        spike_trial_2_concat = list(spike_train_l_[0][spike_train_l_[0] < trial_len_ - shift_ms])
        # align, only 1st
        spike_trial_1_concat = [spike - shift_ms for spike in spike_trial_1_concat_]
        #print(spike_trial_1_concat, spike_trial_2_concat)

        time_abs_sum_shift_1 = 0
        time_abs_sum_shift_2 = 0
        time_abs_trial_shift_1, _ = sttc_calculate_t(spike_trial_1_concat, len(spike_trial_1_concat), sttc_dt_,
                                                           t_start_=0, t_stop_=trial_len_-shift_ms, verbose_=verbose_)
        time_abs_trial_shift_2, _ = sttc_calculate_t(spike_trial_2_concat, len(spike_trial_2_concat), sttc_dt_,
                                                           t_start_=0, t_stop_=trial_len_-shift_ms, verbose_=verbose_)
        time_abs_sum_shift_1 = time_abs_sum_shift_1 + time_abs_trial_shift_1
        time_abs_sum_shift_2 = time_abs_sum_shift_2 + time_abs_trial_shift_2

        for idx, trial in enumerate(spike_train_l_[1:]):
            spike_trial_1 = list(trial[trial >= shift_ms])
            spike_trial_2 = list(trial[trial < trial_len_ - shift_ms])
            # align, only 1st
            spike_trial_1_aligned = [spike - shift_ms for spike in spike_trial_1]
            spike_trial_1_concat = np.hstack((spike_trial_1_concat,
                                              np.asarray(spike_trial_1_aligned) + (idx + 1) * zero_padding_len_))
            spike_trial_2_concat = np.hstack((spike_trial_2_concat, np.asarray(spike_trial_2) + (idx + 1) * zero_padding_len_))

            time_abs_trial_shift_1, _ = sttc_calculate_t(spike_trial_1_aligned, len(spike_trial_1_aligned), sttc_dt_,
                                                         t_start_=0, t_stop_=trial_len_ - shift_ms, verbose_=verbose_)
            time_abs_trial_shift_2, _ = sttc_calculate_t(spike_trial_2, len(spike_trial_2), sttc_dt_,
                                                         t_start_=0, t_stop_=trial_len_ - shift_ms, verbose_=verbose_)
            time_abs_sum_shift_1 = time_abs_sum_shift_1 + time_abs_trial_shift_1
            time_abs_sum_shift_2 = time_abs_sum_shift_2 + time_abs_trial_shift_2

        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_trial_1_concat.shape, spike_trial_2_concat.shape))
            print(spike_trial_1_concat)
            print(spike_trial_2_concat)

        time_prop_sum_shift_1 = time_abs_sum_shift_1 / (len(spike_train_l_) * (trial_len_ - shift_ms))
        time_prop_sum_shift_2 = time_abs_sum_shift_2 / (len(spike_train_l_) * (trial_len_ - shift_ms))

        isttc = sttc_fixed_2t(spike_trial_1_concat, spike_trial_2_concat, t_start_=0,
                             t_stop_=signal_len_concat - shift_ms*len(spike_train_l_), dt_=sttc_dt_,
                             t_a_=time_prop_sum_shift_1, t_b_=time_prop_sum_shift_2, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l



