import numpy as np
import pandas as pd
from random import randrange
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit, OptimizeWarning

import warnings
from scripts.calculate_acf import acf_sttc, acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat


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


def func_single_exp(x, a, b, c):
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
    t = np.linspace(0, len(ydata_to_fit_), len(ydata_to_fit_)).astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            popt, pcov = curve_fit(func_single_exp, t[start_idx_:], ydata_to_fit_[start_idx_:], maxfev=5000)
            fit_popt = popt
            fit_pcov = pcov
            tau = 1 / fit_popt[1]
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

    return tau


if __name__ == "__main__":
    data_folder = 'Q:\\Personal\\Irina\\projects\\isttc\\'
    # load data
    sua_list_30min = np.load(data_folder + 'results\\allen_mice\\dataset_full_split_check_30min\\sua_list_30min.npy',
                             allow_pickle=True)
    sua_list_30min_binned = np.load(
        data_folder + 'results\\allen_mice\\dataset_full_split_check_30min\\sua_list_30min_binned.npy',
        allow_pickle=True)

    print(f'len sua {len(sua_list_30min)}, len sua_binned {len(sua_list_30min_binned)}')

    # params
    num_lags = 20
    fs = 30000  # raw neuropixels
    bin_size = int(50 * (fs / 1000))
    sttc_dt = int(49 * (fs / 1000))
    signal_len = int(30 * 60 * fs)

    # make trials
    #n_trials = 30
    #trial_len = int(num_lags * bin_size)

    # spikes_trials_30 = get_trials(sua_list_30min[0], signal_len, n_trials, trial_len, verbose_=False)
    # spikes_trials_30_binned = bin_trials(spikes_trials_30, trial_len, bin_size)
    #
    # # calculate taus
    # # Pearson trial-average
    # acf_matrix_trail_avg, acf_average_trial_avg = acf_pearsonr_trial_avg(spikes_trials_30_binned, n_lags_=num_lags)
    # spike_train_trial_avg_tau = fit_single_exp(acf_average_trial_avg, start_idx_=0)
    # spike_train_trial_avg_tau_ms = spike_train_trial_avg_tau * bin_size
    # print(spike_train_trial_avg_tau_ms)
    #
    # # STTC trial-average
    # sttc_matrix_trail_avg, sttc_average_trial_avg = acf_sttc_trial_avg(spikes_trials_30,
    #                                                                    n_lags_=num_lags,
    #                                                                    lag_shift_=bin_size,
    #                                                                    sttc_dt_=sttc_dt,
    #                                                                    zero_padding_len_=int(150 * (fs/1000)),
    #                                                                    verbose_=False)
    # spike_train_trial_avg_sttc_tau = fit_single_exp(sttc_average_trial_avg, start_idx_=0)
    # spike_train_trial_avg_sttc_tau_ms = spike_train_trial_avg_sttc_tau * bin_size
    # print(spike_train_trial_avg_sttc_tau_ms)
    #
    # # STTC concat
    # acf_sttc_trials_concat = acf_sttc_trial_concat(spikes_trials_30,
    #                                                  n_lags_=num_lags,
    #                                                  lag_shift_=bin_size,
    #                                                  sttc_dt_=sttc_dt,
    #                                                  trial_len_=trial_len,
    #                                                  zero_padding_len_=int(2000 * (fs/1000)),
    #                                                  verbose_=False)
    # spike_train_trial_concat_sttc_tau = fit_single_exp(acf_sttc_trials_concat, start_idx_=0)
    # spike_train_trial_concat_sttc_tau_ms = spike_train_trial_concat_sttc_tau * bin_size
    # print(spike_train_trial_concat_sttc_tau_ms)

    n_signals = 50

    units = [randrange(0, len(sua_list_30min) + 1) for i in range(n_signals)]

    acf_full = []
    sttc_full = []
    pearsonr_avg_trial_med = []
    sttc_avg_trial_med = []
    sttc_concat_trial_med = []

    for i in units:
        print('Processing unit {}'.format(i))

        spike_times = sua_list_30min[i]
        ou_spiketrain_binned = sua_list_30min_binned[i]

        # on full signal
        # Using acf func
        spike_train_binned_acf = acf(ou_spiketrain_binned, nlags=num_lags)
        # print('spike_train_binned_acf shape {}, \nspike_train_binned_acf: {}'.format(spike_train_binned_acf.shape, spike_train_binned_acf))
        spike_train_binned_tau = fit_single_exp(spike_train_binned_acf, start_idx_=1)
        spike_train_binned_tau_ms = spike_train_binned_tau * bin_size
        # print('spike_train_binned_popt: {}, spike_train_binned_tau_ms: {}'.format(spike_train_binned_popt, spike_train_binned_tau_ms))
        acf_full.append(spike_train_binned_tau_ms)

        # Using isttc
        spike_train_acf = acf_sttc(spike_times, num_lags, lag_shift_=bin_size, sttc_dt_=sttc_dt,
                                   signal_length_=signal_len, verbose_=False)
        # print('spike_train_acf shape {}, \nspike_train_acf: {}'.format(len(spike_train_acf), spike_train_acf))
        spike_train_popt_tau = fit_single_exp(spike_train_acf, start_idx_=1)
        spike_train_tau_ms = spike_train_popt_tau * bin_size
        # print('spike_train_popt: {}, spike_train_tau_ms: {}'.format(spike_train_popt, spike_train_tau_ms))
        sttc_full.append(spike_train_tau_ms)

        # on trials
        ### Run for 500 realizations
        n_stims = 50

        n_trials = 50
        trial_len = int(num_lags * bin_size)

        pearson_avg_l = []
        sttc_avg_l = []
        sttc_concat_l = []
        stim_l = []

        for j in range(n_stims):
            #print(f'Run {j}')
            spikes_trials_stim = get_trials(spike_times, signal_len, n_trials, trial_len, verbose_=False)
            spikes_trials_binned_stim = bin_trials(spikes_trials_stim, trial_len, bin_size)

            # Pearson trial-average
            acf_matrix_trail_avg, acf_average_trial_avg = acf_pearsonr_trial_avg(spikes_trials_binned_stim, n_lags_=num_lags)
            spike_train_trial_avg_tau = fit_single_exp(acf_average_trial_avg, start_idx_=1)
            spike_train_trial_avg_tau_ms = spike_train_trial_avg_tau * bin_size
            #print(spike_train_trial_avg_tau_ms)

            # STTC trial-average
            sttc_matrix_trail_avg, sttc_average_trial_avg = acf_sttc_trial_avg(spikes_trials_stim,
                                                                               n_lags_=num_lags,
                                                                               lag_shift_=bin_size,
                                                                               sttc_dt_=sttc_dt,
                                                                               zero_padding_len_=int(150 * (fs/1000)),
                                                                               verbose_=False)
            spike_train_trial_avg_sttc_tau = fit_single_exp(sttc_average_trial_avg, start_idx_=1)
            spike_train_trial_avg_sttc_tau_ms = spike_train_trial_avg_sttc_tau * bin_size
            #print(spike_train_trial_avg_sttc_tau_ms)

            # STTC concat v3
            acf_sttc_trials_concat = acf_sttc_trial_concat(spikes_trials_stim,
                                                             n_lags_=num_lags,
                                                             lag_shift_=bin_size,
                                                             sttc_dt_=sttc_dt,
                                                             trial_len_=trial_len,
                                                             zero_padding_len_=int(2000 * (fs/1000)),
                                                             verbose_=False)
            spike_train_trial_concat_sttc_tau = fit_single_exp(acf_sttc_trials_concat, start_idx_=1)
            spike_train_trial_concat_sttc_tau_ms = spike_train_trial_concat_sttc_tau * bin_size
            #print(spike_train_trial_concat_sttc_v2_tau_ms)

            stim_l.append(j)
            pearson_avg_l.append(spike_train_trial_avg_tau_ms)
            sttc_avg_l.append(spike_train_trial_avg_sttc_tau_ms)
            sttc_concat_l.append(spike_train_trial_concat_sttc_tau_ms)

        tau_df = pd.DataFrame(np.vstack((stim_l, pearson_avg_l, sttc_avg_l, sttc_concat_l)).T,
                          columns=['run_id', 'tau_pearsonr_avg', 'tau_sttc_avg', 'tau_sttc_concat'])

        tau_df.to_pickle(data_folder + 'results\\allen_mice\\test_full_split\\tau_df_' + str(i) + '.pkl')

        print(f'acf {spike_train_binned_tau_ms}, sttc {spike_train_tau_ms}, p_avg {np.nanmedian(pearson_avg_l)}, '
              f'sttc avg {np.nanmedian(sttc_avg_l)}, sttc concat {np.nanmedian(sttc_concat_l)}')


