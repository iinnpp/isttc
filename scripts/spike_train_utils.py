"""
todo
"""

import numpy as np
from random import randrange


def simulate_hawkes_thinning(fr_hz_, tau_ms_, alpha_, duration_ms_, seed_=None):
    """
    Simulate 1D Hawkes point process with exponential kernel using Ogata's thinning.

    :param fr_hz_: stationary firing rate stationary rate, hz
    :param tau_ms_: target time constant, ms
    :param alpha_: kernel integral (<1 for stability), means excitation strength
    :param duration_ms_: total sim time, ms
    :param seed_: seed for random generator
    :return 1d array of spike times
    """
    rng = np.random.default_rng(seed_)
    tau_kernel_ms = tau_ms_ * (1 - alpha_) # internal tau for generation
    tau = tau_kernel_ms / 1000.0
    mu = fr_hz_ * (1 - alpha_)  # baseline intensity (hz)

    t = 0.0
    events = []
    # current kernel sum K = sum_i (alpha/tau e^{-(t - t_i)/tau})
    K = 0.0

    # convert to sec for intensity calc
    T_sec = duration_ms_ / 1000.0

    while True:
        lambda_upper = mu + K  # global upper‐bound on intensity
        if lambda_upper <= 0:
            break
        # draw next candidate time increment (sec)
        w = rng.exponential(1.0 / lambda_upper)
        t += w
        if t >= T_sec:
            break
        # decay K over interval w
        K *= np.exp(-w / tau)
        # actual intensity at new t
        lambda_t = mu + K
        if rng.random() < lambda_t / lambda_upper:
            # accept
            events.append(t * 1000.0)  # store in ms
            # add kernel jump alpha/tau at event
            K += alpha_ / tau

    return np.array(events)


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


def bin_spike_train_fixed_len(spike_train_int_l_, bin_length_ms_, fs_, signal_len_, verbose_=False):
    """
    Bin spike train.

    :param spike_train_int_l_: list, list of spike times (int), sampling frequency fs_
    :param bin_length_ms_: int, bin length in ms
    :param fs_: int, sampling frequency in Hz
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: 1d array, binned spike train, each bin contains spike count
    """
    bin_length_fs = int(fs_ / 1000 * bin_length_ms_)
    n_bin_edges = int(signal_len_ / bin_length_fs)
    bins = np.linspace(0, bin_length_fs * n_bin_edges, n_bin_edges + 1).astype(int)
    binned_spike_train, _ = np.histogram(spike_train_int_l_, bins)

    if verbose_:
        print('Binning spike train: bin_length_ms {}, bin_length_fs {}'.format(bin_length_ms_, bin_length_fs))
        print('n bins {}, spike bin count: number of spikes in bin - number of bins {}'.format(binned_spike_train.shape,
                                                                                               np.unique(
                                                                                                   binned_spike_train,
                                                                                                   return_counts=True)))
    return binned_spike_train


def get_trials(spike_times_, signal_len_, n_trials_, trial_len_, verbose_=False):
    # get random trail starts and ends
    trials_start = [randrange(0, signal_len_-trial_len_+1) for i in range(n_trials_)]
    trials_end = [trial_start + trial_len_ for trial_start in trials_start]
    trial_intervals = np.vstack((trials_start, trials_end)).T
    if verbose_:
        print('N trials {}, trail len {}, n trial starts {}, \ntrial starts {}, \ntrial starts {}'.format(n_trials_, trial_len_,
                                                                                                          len(trials_start),
                                                                                                          trials_start, trials_end))
    # get spikes
    spikes_trials = []
    for i in range(n_trials_):
        spikes_trial = spike_times_[np.logical_and(spike_times_ >= trial_intervals[i,0], spike_times_ < trial_intervals[i,1])]
        spikes_trials.append(spikes_trial)

    # realign all trails to start with 0
    spikes_trials_realigned_l = []
    for idx, trial in enumerate(spikes_trials):
        spikes_trial_realigned = trial - trial_intervals[idx,0]
        spikes_trials_realigned_l.append(spikes_trial_realigned)

    return spikes_trials_realigned_l


def bin_trials(spikes_trials_l_, trial_len_, bin_size_):
    binned_spikes_trials_l = []

    n_bin_edges =  int(trial_len_/bin_size_)
    bins_ = np.linspace(0, bin_size_ * n_bin_edges, n_bin_edges + 1).astype(int)
    for trial in spikes_trials_l_:
        binned_spike_train, _ = np.histogram(trial, bins_)
        binned_spikes_trials_l.append(binned_spike_train)
    binned_spikes_trials_2d = np.asarray(binned_spikes_trials_l)

    return binned_spikes_trials_2d


def get_lv(spike_train_int_l_, verbose_=False):
    """
    Calculate Local Variation (Lv) based on Shinomoto, 2009.

    :param spike_train_int_l_: list, list of spike times (int)
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    :return: float, Lv. If spike train contains < 3 spikes -> lv = np.nan.
    """
    unit_spikes_ = np.asarray(spike_train_int_l_)
    isi = np.diff(unit_spikes_)

    n_isi = len(isi)
    if n_isi > 1:
        sum_ = 0
        for isi_idx_, isi_ in enumerate(isi[:-1]):
            top = isi[isi_idx_] - isi[isi_idx_+1]
            bottom = isi[isi_idx_] + isi[isi_idx_ + 1]
            if bottom == 0:
                sum_ = sum_
            else:
                sum_ = sum_ + (top / bottom) ** 2
        lv = (3/(n_isi - 1)) * sum_
    else:
        lv = np.nan
        if verbose_:
            print('n_isi < 2: skipping the unit...')

    return lv

# def get_firing_rate(spike_train_int_l_, fs_):
#     """
#     Calculate firing rate for a single unit.
#
#     :param spike_train_int_l_: list, list of spike times (int), sampling frequency fs_
#     :param fs_: int, sampling frequency in Hz
#     :return: float, firing rate, Hz
#     """
#     n_spikes = len(spike_train_int_l_)
#     last_spike_ts_s = spike_train_int_l_[-1] / fs_
#     fr = n_spikes / last_spike_ts_s
#
#     return fr



