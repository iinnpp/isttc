import numpy as np
import pandas as pd
import csv
import os
from random import randrange
from statsmodels.tsa.stattools import acf

from scripts.calculate_acf import acf_sttc, acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat
from scripts.calculate_tau import fit_single_exp
from scripts.spike_train_utils import bin_spike_train_fixed_len


def write_sua_csv(csv_file_name_, sua_list_original_, sua_list_new_, verbose_=False):
    """
    Write binned spike trains in csv file and stores it on the disk.
    Each row is one unit: animal_id, age, unit_id, channel_id, non_zero_bins_ratio, fr, rpv, rec_length, bin1, ..., bin_n

    :param csv_file_name_: string, folder path
    :param sua_list_original_: string, brain area label
    :param sua_list_new_: string, bin size label (ms)
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    """
    try:
        os.remove(csv_file_name_)
        print('file {} removed'.format(csv_file_name_))
    except FileNotFoundError:
        print('file {} did not exist, nothing to remove'.format(csv_file_name_))

    with open(csv_file_name_, 'a', newline='') as f:
        writer = csv.writer(f)
        for unit_row_n, spike_train in enumerate(sua_list_new_):
            if verbose_:
                print('Writing unit {}'.format(unit_row_n))
            spike_train_l = spike_train.tolist()
            row = [sua_list_original_[unit_row_n][0]] + [sua_list_original_[unit_row_n][1]] \
                  + [sua_list_original_[unit_row_n][2]] + [sua_list_original_[unit_row_n][3]] + spike_train_l
            writer.writerow(row)


if __name__ == "__main__":
    # data_folder = 'Q:\\Personal\\Irina\\projects\\isttc\\results\\allen_mice\\'
    data_folder = 'E:\\isttc\\results\\allen_mice\\'
    fs = 30000  # neuropixels

    trim_spikes = False
    bin_spikes = False
    calculate_acf = True
    calculate_tau = False

    min_to_keep = 30

    if trim_spikes:
        csv_data_file = data_folder + 'dataset\\allen_func_conn_around30min_spont.csv'
        with open(csv_data_file, newline='') as f:
            reader = csv.reader(f)
            sua_list = list(reader)
        print(f'Loaded N units {len(sua_list)}')

        # use first min_to_keep of the signal
        sua_list_trimmed = []
        for i in range(len(sua_list)):
            spike_train_ = np.asarray(sua_list[i][4:]).astype(float)
            spike_train_fs = spike_train_ * fs  # csv is in sec
            spike_train_fs_int = spike_train_fs.astype(int)
            n_spikes_out = np.count_nonzero(spike_train_fs_int >= min_to_keep * 60 * fs)
            print(f'n spikes to remove {n_spikes_out}')
            sua_list_trimmed.append(spike_train_fs_int[spike_train_fs_int < min_to_keep * 60 * fs])

        # save
        write_sua_csv(data_folder + 'dataset\\cut_' + str(min_to_keep) + 'min\\sua_list.csv',
                      sua_list, sua_list_trimmed, verbose_=True)

    if bin_spikes:
        params_dict = {'25ms': {'bin_size': 25, 'bin_size_suffix': '25ms', 'calc': True},
                       '40ms': {'bin_size': 40, 'bin_size_suffix': '40ms', 'calc': True},
                       '50ms': {'bin_size': 50, 'bin_size_suffix': '50ms', 'calc': True},
                       '60ms': {'bin_size': 60, 'bin_size_suffix': '60ms', 'calc': True},
                       '75ms': {'bin_size': 75, 'bin_size_suffix': '75ms', 'calc': True},
                       '100ms': {'bin_size': 100, 'bin_size_suffix': '100ms', 'calc': True}
                       }
        csv_data_file = data_folder + 'dataset\\cut_30min\\sua_list.csv'
        with open(csv_data_file, newline='') as f:
            reader = csv.reader(f)
            sua_list = list(reader)
        print(f'Loaded N units {len(sua_list)}')
        signal_len = min_to_keep * 60 * fs
        for k, v in params_dict.items():
            print(f'processing {k}')
            sua_list_binned_l = []
            for j in range(len(sua_list)):
                binned_spike_train = bin_spike_train_fixed_len([int(spike) for spike in sua_list[j][4:]],
                                                               v['bin_size'], fs, signal_len,
                                                               verbose_=True)
                sua_list_binned_l.append(binned_spike_train)

            write_sua_csv(data_folder + 'dataset\\cut_30min\\sua_list_binned_' + k + '.csv',
                          sua_list, sua_list_binned_l, verbose_=True)

    if calculate_acf:
        params_dict = {'isttc_25_40': {'bin_size': 25, 'n_lags': 40, 'bin_size_suffix': '25ms', 'metric': 'isttc',
                                       'calc': False},
                       'isttc_40_25': {'bin_size': 40, 'n_lags': 25, 'bin_size_suffix': '40ms', 'metric': 'isttc',
                                       'calc': False},
                       'isttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'isttc',
                                       'calc': True},
                       'isttc_60_16': {'bin_size': 60, 'n_lags': 16, 'bin_size_suffix': '60ms', 'metric': 'isttc',
                                       'calc': False},
                       'isttc_75_13': {'bin_size': 75, 'n_lags': 13, 'bin_size_suffix': '75ms', 'metric': 'isttc',
                                       'calc': False},
                       'isttc_100_10': {'bin_size': 100, 'n_lags': 10, 'bin_size_suffix': '100ms', 'metric': 'isttc',
                                        'calc': False},
                       'acf_25_40': {'bin_size': 25, 'n_lags': 40, 'bin_size_suffix': '25ms', 'metric': 'acf',
                                     'calc': False},
                       'acf_40_25': {'bin_size': 40, 'n_lags': 25, 'bin_size_suffix': '40ms', 'metric': 'acf',
                                     'calc': False},
                       'acf_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'acf',
                                     'calc': False},
                       'acf_60_16': {'bin_size': 60, 'n_lags': 16, 'bin_size_suffix': '60ms', 'metric': 'acf',
                                     'calc': False},
                       'acf_75_13': {'bin_size': 75, 'n_lags': 13, 'bin_size_suffix': '75ms', 'metric': 'acf',
                                     'calc': False},
                       'acf_100_10': {'bin_size': 100, 'n_lags': 10, 'bin_size_suffix': '100ms', 'metric': 'acf',
                                      'calc': False}
                       }

        #num_lags = 20
        # bin_size = int(50 * (fs / 1000))
        # sttc_dt = int(49 * (fs / 1000))
        signal_len = int(min_to_keep * 60 * fs)

        for k, v in params_dict.items():
            print(f'processing {k}')
            if not v['calc']:
                print('Skipping...')
            else:
                if v['metric'] == 'isttc':
                    csv_data_file = data_folder + 'dataset\\cut_30min\\sua_list.csv'
                    print(f'Loading file {csv_data_file}')
                    with open(csv_data_file, newline='') as f:
                        reader = csv.reader(f)
                        sua_list = list(reader)
                    print(f'Loaded N units {len(sua_list)}')

                    isttc_full_l = []
                    for spike_train_idx, spike_train in enumerate(sua_list):
                        if spike_train_idx % 100 == 0:
                            print(f'Processing unit {spike_train_idx}')
                        spike_train_int = np.asarray([int(spike) for spike in spike_train[4:]])
                        lag_shift = int(v['bin_size'] * (fs / 1000)) + 1
                        sttc_dt = int(v['bin_size'] * (fs / 1000))
                        # print(lag_shift, sttc_dt)
                        spike_train_acf = np.asarray(acf_sttc(spike_train_int, v['n_lags'], lag_shift_=lag_shift, sttc_dt_=sttc_dt,
                                                   signal_length_=signal_len, verbose_=False))
                        # print(spike_train_acf)
                        isttc_full_l.append(spike_train_acf)

                    write_sua_csv(data_folder + 'dataset\\cut_30min\\acf_non_binned\\acf_non_binned_' + k + '.csv',
                                  sua_list, isttc_full_l, verbose_=True)

                if v['metric'] == 'acf':
                    csv_data_file = data_folder + 'dataset\\cut_30min\\sua_list_binned_' + v['bin_size_suffix'] + '.csv'
                    print(f'Loading file {csv_data_file}')
                    with open(csv_data_file, newline='') as f:
                        reader = csv.reader(f)
                        sua_list_binned = list(reader)
                    print(f'Loaded N units {len(sua_list_binned)}')

                    acf_full_l = []
                    for spike_train_binned_idx, spike_train_binned in enumerate(sua_list_binned):
                        if spike_train_binned_idx % 100 == 0:
                            print(f'Processing unit {spike_train_binned_idx}')
                        spike_train_binned_int = np.asarray([int(spike) for spike in spike_train_binned[4:]])
                        spike_train_binned_acf = acf(spike_train_binned_int, nlags=v['n_lags'])
                        acf_full_l.append(spike_train_binned_acf)

                    write_sua_csv(data_folder + 'dataset\\cut_30min\\acf_binned\\acf_binned_' + k + '.csv',
                                  sua_list_binned, acf_full_l, verbose_=True)

        # for spike_train_idx, spike_train in enumerate(sua_list):
        #     print('Processing unit {}'.format(spike_train_idx))
        #
        #     # Using isttc
        #     spike_train_acf = acf_sttc(spike_train, num_lags, lag_shift_=bin_size, sttc_dt_=sttc_dt,
        #                                signal_length_=signal_len, verbose_=False)
        #     sttc_full.append(spike_train_tau_ms)
        #
        #     # on full signal
        #     # Using acf func
        #     spike_train_binned_acf = acf(ou_spiketrain_binned, nlags=num_lags)
        #     # print('spike_train_binned_acf shape {}, \nspike_train_binned_acf: {}'.format(spike_train_binned_acf.shape, spike_train_binned_acf))
        #     spike_train_binned_tau = fit_single_exp(spike_train_binned_acf, start_idx_=1)
        #     spike_train_binned_tau_ms = spike_train_binned_tau * bin_size
        #     # print('spike_train_binned_popt: {}, spike_train_binned_tau_ms: {}'.format(spike_train_binned_popt, spike_train_binned_tau_ms))
        #     acf_full.append(spike_train_binned_tau_ms)
        #
        #     # Using isttc
        #     spike_train_acf = acf_sttc(spike_times, num_lags, lag_shift_=bin_size, sttc_dt_=sttc_dt,
        #                                signal_length_=signal_len, verbose_=False)
        #     # print('spike_train_acf shape {}, \nspike_train_acf: {}'.format(len(spike_train_acf), spike_train_acf))
        #     spike_train_popt_tau = fit_single_exp(spike_train_acf, start_idx_=1)
        #     spike_train_tau_ms = spike_train_popt_tau * bin_size
        #     # print('spike_train_popt: {}, spike_train_tau_ms: {}'.format(spike_train_popt, spike_train_tau_ms))
        #     sttc_full.append(spike_train_tau_ms)

    # print(f'len sua {len(sua_list_30min)}, len sua_binned {len(sua_list_30min_binned)}')

    # make trials
    # n_trials = 30
    # trial_len = int(num_lags * bin_size)

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

    # n_signals = 50
    #
    # units = [randrange(0, len(sua_list_30min) + 1) for i in range(n_signals)]
    #
    # acf_full = []
    # sttc_full = []
    # pearsonr_avg_trial_med = []
    # sttc_avg_trial_med = []
    # sttc_concat_trial_med = []
    #
    # for i in units:
    #     print('Processing unit {}'.format(i))
    #
    #     spike_times = sua_list_30min[i]
    #     ou_spiketrain_binned = sua_list_30min_binned[i]
    #
    #     # on full signal
    #     # Using acf func
    #     spike_train_binned_acf = acf(ou_spiketrain_binned, nlags=num_lags)
    #     # print('spike_train_binned_acf shape {}, \nspike_train_binned_acf: {}'.format(spike_train_binned_acf.shape, spike_train_binned_acf))
    #     spike_train_binned_tau = fit_single_exp(spike_train_binned_acf, start_idx_=1)
    #     spike_train_binned_tau_ms = spike_train_binned_tau * bin_size
    #     # print('spike_train_binned_popt: {}, spike_train_binned_tau_ms: {}'.format(spike_train_binned_popt, spike_train_binned_tau_ms))
    #     acf_full.append(spike_train_binned_tau_ms)
    #
    #     # Using isttc
    #     spike_train_acf = acf_sttc(spike_times, num_lags, lag_shift_=bin_size, sttc_dt_=sttc_dt,
    #                                signal_length_=signal_len, verbose_=False)
    #     # print('spike_train_acf shape {}, \nspike_train_acf: {}'.format(len(spike_train_acf), spike_train_acf))
    #     spike_train_popt_tau = fit_single_exp(spike_train_acf, start_idx_=1)
    #     spike_train_tau_ms = spike_train_popt_tau * bin_size
    #     # print('spike_train_popt: {}, spike_train_tau_ms: {}'.format(spike_train_popt, spike_train_tau_ms))
    #     sttc_full.append(spike_train_tau_ms)
    #
    #     # on trials
    #     ### Run for 500 realizations
    #     n_stims = 50
    #
    #     n_trials = 50
    #     trial_len = int(num_lags * bin_size)
    #
    #     pearson_avg_l = []
    #     sttc_avg_l = []
    #     sttc_concat_l = []
    #     stim_l = []
    #
    #     for j in range(n_stims):
    #         #print(f'Run {j}')
    #         spikes_trials_stim = get_trials(spike_times, signal_len, n_trials, trial_len, verbose_=False)
    #         spikes_trials_binned_stim = bin_trials(spikes_trials_stim, trial_len, bin_size)
    #
    #         # Pearson trial-average
    #         acf_matrix_trail_avg, acf_average_trial_avg = acf_pearsonr_trial_avg(spikes_trials_binned_stim, n_lags_=num_lags)
    #         spike_train_trial_avg_tau = fit_single_exp(acf_average_trial_avg, start_idx_=1)
    #         spike_train_trial_avg_tau_ms = spike_train_trial_avg_tau * bin_size
    #         #print(spike_train_trial_avg_tau_ms)
    #
    #         # STTC trial-average
    #         sttc_matrix_trail_avg, sttc_average_trial_avg = acf_sttc_trial_avg(spikes_trials_stim,
    #                                                                            n_lags_=num_lags,
    #                                                                            lag_shift_=bin_size,
    #                                                                            sttc_dt_=sttc_dt,
    #                                                                            zero_padding_len_=int(150 * (fs/1000)),
    #                                                                            verbose_=False)
    #         spike_train_trial_avg_sttc_tau = fit_single_exp(sttc_average_trial_avg, start_idx_=1)
    #         spike_train_trial_avg_sttc_tau_ms = spike_train_trial_avg_sttc_tau * bin_size
    #         #print(spike_train_trial_avg_sttc_tau_ms)
    #
    #         # STTC concat v3
    #         acf_sttc_trials_concat = acf_sttc_trial_concat(spikes_trials_stim,
    #                                                          n_lags_=num_lags,
    #                                                          lag_shift_=bin_size,
    #                                                          sttc_dt_=sttc_dt,
    #                                                          trial_len_=trial_len,
    #                                                          zero_padding_len_=int(2000 * (fs/1000)),
    #                                                          verbose_=False)
    #         spike_train_trial_concat_sttc_tau = fit_single_exp(acf_sttc_trials_concat, start_idx_=1)
    #         spike_train_trial_concat_sttc_tau_ms = spike_train_trial_concat_sttc_tau * bin_size
    #         #print(spike_train_trial_concat_sttc_v2_tau_ms)
    #
    #         stim_l.append(j)
    #         pearson_avg_l.append(spike_train_trial_avg_tau_ms)
    #         sttc_avg_l.append(spike_train_trial_avg_sttc_tau_ms)
    #         sttc_concat_l.append(spike_train_trial_concat_sttc_tau_ms)
    #
    #     tau_df = pd.DataFrame(np.vstack((stim_l, pearson_avg_l, sttc_avg_l, sttc_concat_l)).T,
    #                       columns=['run_id', 'tau_pearsonr_avg', 'tau_sttc_avg', 'tau_sttc_concat'])
    #
    #     tau_df.to_pickle(data_folder + 'results\\allen_mice\\test_full_split\\tau_df_' + str(i) + '.pkl')
    #
    #     print(f'acf {spike_train_binned_tau_ms}, sttc {spike_train_tau_ms}, p_avg {np.nanmedian(pearson_avg_l)}, '
    #           f'sttc avg {np.nanmedian(sttc_avg_l)}, sttc concat {np.nanmedian(sttc_concat_l)}')
