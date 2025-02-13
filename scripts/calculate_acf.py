"""
Functions to calculate autocorrelation function:
* using ACF equation (use directly acf function from stattools; my_acf and my_autocorr are the save),
* using Pearson equation,
* using iSTTC,
* trial average ACF using Pearson correlation (see monkey papers),
* trial average ACF using STTC (with 0-padding),
* trial concat ACF using STTC (with 0-padding).
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
    If n_lags_ >= signal_length then n_lags_ = signal_length-2. For example, is the signal_len = 20 then for lag 20
    there are no values to correlate and for the lag 19 there is only 1 value per array. In this case Pearson
    is NaN because denominator is 0. So the last lag to correlate is set to 18.
    ACF len is = n_lags_ + 1 for signal_length > n_lags_ + 1 (as in acf from statsmodels.tsa.stattools) otherwise
    ACF len is = len(signal_) - 1 (if n_lags_ >= len(signal_) or n_lags_ == len(signal_) - 1).
    :param signal_: numeric, 1d array.
    :param n_lags_: int, number of lags.
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, numeric. Array len is = n_lags_ + 1 for
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

    for i in range(1, n_lags + 1):
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

    for i in range(1, (n_lags + 1)):
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
    if verbose_:
        print(f'sttc_calculate_t  spiketrain {spiketrain_}, n spikes {n_spikes_}')
    if n_spikes_ == 0:
        return 0, 0

    # Maximum possible time
    time_abs = 2 * n_spikes_ * dt_
    if verbose_:
        print(f'Initial time_abs: {time_abs}')

    # Handle single spike case
    if n_spikes_ == 1:
        if verbose_:
            print('n spike 1')
        # Adjust for the first spike
        if spiketrain_[0] - t_start_ < dt_:
            time_abs += spiketrain_[0] - dt_ - t_start_
            if verbose_:
                print(f'n spike 1, adjustment for first spike: {time_abs}')
        # Adjust for the last spike
        if t_stop_ - spiketrain_[-1] < dt_:
            time_abs += t_stop_ - (spiketrain_[-1] + dt_)
            if verbose_:
                print(f'n spike 1, adjustment for last spike: {time_abs}')

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

    if verbose_:
        print(f'Adjusted time_abs: {time_abs}')

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

    Signal temporal resolution must be same as lag_shift_, sttc_dt_ and signal_length_ (e.g. all in ms or raw fs)

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
    sttc_no_shift = sttc(signal_, signal_, t_start_=0, t_stop_=signal_length_, dt_=sttc_dt_, verbose_=verbose_)
    acf_l = [sttc_no_shift]

    # Iterate through each shift and calculate the STTC for shifted signals
    for shift_ms in shift_ms_l:
        spike_1 = signal_[signal_ >= shift_ms]
        spike_2 = signal_[signal_ < signal_length_ - shift_ms]
        # align, only 1st
        spike_1_aligned = spike_1 - shift_ms
        if verbose_:
            print('spike_1 {}, spike_2 {}'.format(spike_1.shape, spike_2.shape))
        isttc = sttc(spike_1_aligned, spike_2, t_start_=0, t_stop_=signal_length_ - shift_ms, dt_=sttc_dt_,
                     verbose_=verbose_)
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


def get_lag_arrays(spike_train_l_: list, lag_1_idx_: int, lag_2_idx_: int, lag_shift_: int, zero_padding_len_: int):
    """

    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param lag_1_idx_: int, index for the first lag.
    :param lag_2_idx_: int, index for the first lag.
    :param lag_shift_: int, shift for a time lag (in time points)
    :param zero_padding_len_: int, len of zero padding (in time points).
    :return: Two 1D arrays containing spike times for lag 1 and lag 2.
    """

    def extract_lag(spike_train: np.ndarray, lag_idx: int, lag_shift: int) -> np.ndarray:
        """Extract spikes corresponding to a specific lag."""
        start = lag_idx * lag_shift
        end = start + lag_shift
        return spike_train[(spike_train > start) & (spike_train <= end)]

    def add_spacing(lag_list: list, spacing: int) -> list:
        """Add zero-padding spacing to lag arrays."""
        return [lag + i * spacing for i, lag in enumerate(lag_list)]

    # Extract spikes for both lags
    first_lag_l = [extract_lag(trial, lag_1_idx_, lag_shift_) for trial in spike_train_l_]
    second_lag_l = [extract_lag(trial, lag_2_idx_, lag_shift_) for trial in spike_train_l_]

    # Add padding zeros
    first_lag_spaced = add_spacing(first_lag_l, zero_padding_len_)
    second_lag_spaced = add_spacing(second_lag_l, zero_padding_len_)

    # Flatten arrays to 1D
    lag1_l = np.hstack(first_lag_spaced).tolist() if first_lag_spaced else []
    lag2_l = np.hstack(second_lag_spaced).tolist() if second_lag_spaced else []
    return lag1_l, lag2_l


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


def acf_sttc_trial_avg(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int, zero_padding_len_: int,
                       verbose_: bool = True):
    """
    Trial average autocorrelation using STTC. T term is calculated as in sttc_trail_concat, "trials" are chunks of
    lag_shift size. The length of the signal for T term calc is sum of all chunks (no zero padding in T term).

    :param spike_train_l_: list of spike trains, every element of the list contains spikes from 1 trial, length of the
    list is equal to the number of trials. Spike times are realigned (each trial starts at time 0).
    :param n_lags_: int, number of lags
    :param lag_shift_: int, shift for a time lag (in time points)
    :param sttc_dt_: int, dt parameter for STTC calculation
    :param zero_padding_len_: int, len of zero padding (in time points).
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 2d autocorrelation matrix n_lags x n_lags, 1d array, autocorrelation function.
    """
    def extract_lag(spike_train: np.ndarray, lag_idx: int, lag_shift: int) -> np.ndarray:
        """Extract spikes corresponding to a specific lag."""
        start = lag_idx * lag_shift
        end = start + lag_shift
        return spike_train[(spike_train > start) & (spike_train <= end)]

    def calculate_t_term(spike_train_l, trial_len, dt, verbose):
        abs_time_sum = sum(sttc_calculate_t(spike_train, len(spike_train), dt, 0, trial_len, verbose)[0]
                           for spike_train in spike_train_l)
        return abs_time_sum / (len(spike_train_l) * trial_len)

    if verbose_:
        print('Processing {} trials: n lags {}, lag shift {}, sttc dt {}, zero padding len {}'.
              format(len(spike_train_l_), n_lags_, lag_shift_, sttc_dt_, zero_padding_len_))
    acf_matrix = np.zeros((n_lags_, n_lags_))

    for i in np.arange(n_lags_ - 1):
        for j in np.arange(i + 1, n_lags_):  # filling i-th row
            # get arrays for T term calculation - without zero padding
            # Extract spikes for both lags
            first_lag_l = [extract_lag(trial, i, lag_shift_) for trial in spike_train_l_]
            second_lag_l = [extract_lag(trial, j, lag_shift_) for trial in spike_train_l_]
            first_lag_l_aligned = [trial - lag_shift_ * i for trial in first_lag_l]
            first_lag_2_aligned = [trial - lag_shift_ * j for trial in second_lag_l]
            l1_t = calculate_t_term(first_lag_l_aligned, lag_shift_, sttc_dt_, verbose_)
            l2_t = calculate_t_term(first_lag_2_aligned, lag_shift_, sttc_dt_, verbose_)

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


def acf_sttc_trial_concat(spike_train_l_: list, n_lags_: int, lag_shift_: int, sttc_dt_: int,
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
        time_prop_sum_shift_1 = calculate_t_term(spike_train_1_shifted_l, trial_len_ - shift, sttc_dt_, verbose_)
        time_prop_sum_shift_2 = calculate_t_term(spike_train_2_shifted_l, trial_len_ - shift, sttc_dt_, verbose_)
        # Calculate sttc for the shifted signals
        isttc = sttc_fixed_2t(spike_trial_1_concat, spike_trial_2_concat, dt_=sttc_dt_,
                              t_a_=time_prop_sum_shift_1, t_b_=time_prop_sum_shift_2, verbose_=verbose_)
        acf_l.append(isttc)

    return acf_l


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


# todo make a separate script - calculate_quality_metrics?
def calculate_acf_flags_t(acf_dict_in_out_, start_idx=1, end_idx=5):
    """
    Calculate ACF properties:
    1. Decline in [50ms, 200ms] period (see Cavanagh et al. 2016) for 50 ms bins; in general decline in [1, 5] bins
    period.
    2. ...

    Calculated properties are added to the input dict.

    :param acf_dict_in_out_: dict, dict with calculated ACF functions
    :param start_idx: first acf value index
    :param end_idx: last acf value index (not inclusive)
    :return: acf_dict_in_out_: dict, dict same as input with new fields
    """
    for k, v in acf_dict_in_out_.items():
        if np.all(np.diff(v['acf'][start_idx:end_idx]) <= 0):
            v['acf_decay_1_4'] = True
        else:
            v['acf_decay_1_4'] = False

    return acf_dict_in_out_


