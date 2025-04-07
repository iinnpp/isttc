import numpy as np
from itertools import islice
import pickle
import csv
import os
import sys
from datetime import datetime
from statsmodels.tsa.stattools import acf

from scripts.calculate_acf import acf_sttc, acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat
from scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from scripts.spike_train_utils import bin_spike_train_fixed_len, get_trials, bin_trials
from scripts.cfg_global import project_folder_path


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
                  + [sua_list_original_[unit_row_n][2]] + [sua_list_original_[unit_row_n][3]] \
                  + [sua_list_original_[unit_row_n][4]] + [sua_list_original_[unit_row_n][5]] \
                  + [sua_list_original_[unit_row_n][6]] + [sua_list_original_[unit_row_n][7]] \
                  + spike_train_l
            writer.writerow(row)


if __name__ == "__main__":
    data_folder = project_folder_path + 'results\\allen_mice\\'
    fs = 30000  # neuropixels

    trim_spikes = False
    bin_spikes = False
    calculate_acf = False
    calculate_trials = True
    calculate_trials_pearsonr = True
    calculate_trials_sttc_avg = False
    calculate_trials_sttc_concat = False

    min_to_keep = 30

    if trim_spikes:
        csv_data_file = data_folder + 'dataset\\allen_func_conn_around30min_spont_with_quality_metrics.csv'
        with open(csv_data_file, newline='') as f:
            reader = csv.reader(f)
            sua_list = list(reader)
        print(f'Loaded N units {len(sua_list)}')

        # use first min_to_keep of the signal
        sua_list_trimmed = []
        for i in range(len(sua_list)):
            spike_train_ = np.asarray(sua_list[i][8:]).astype(float)
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
        csv_data_file = data_folder + 'dataset\\cut_30min\\sua_list_constrained.csv'
        with open(csv_data_file, newline='') as f:
            reader = csv.reader(f)
            sua_list = list(reader)
        print(f'Loaded N units {len(sua_list)}')
        signal_len = min_to_keep * 60 * fs
        for k, v in params_dict.items():
            print(f'processing {k}')
            sua_list_binned_l = []
            for j in range(len(sua_list)):
                binned_spike_train = bin_spike_train_fixed_len([int(spike) for spike in sua_list[j][8:]],
                                                               v['bin_size'], fs, signal_len,
                                                               verbose_=True)
                sua_list_binned_l.append(binned_spike_train)

            write_sua_csv(data_folder + 'dataset\\cut_30min\\sua_list_constrained_binned_' + k + '.csv',
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

        # num_lags = 20
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
                        spike_train_int = np.asarray([int(spike) for spike in spike_train[8:]])
                        lag_shift = int(v['bin_size'] * (fs / 1000)) + 1
                        sttc_dt = int(v['bin_size'] * (fs / 1000))
                        # print(lag_shift, sttc_dt)
                        spike_train_acf = np.asarray(
                            acf_sttc(spike_train_int, v['n_lags'], lag_shift_=lag_shift, sttc_dt_=sttc_dt,
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
                        spike_train_binned_int = np.asarray([int(spike) for spike in spike_train_binned[8:]])
                        spike_train_binned_acf = acf(spike_train_binned_int, nlags=v['n_lags'])
                        acf_full_l.append(spike_train_binned_acf)

                    write_sua_csv(data_folder + 'dataset\\cut_30min\\acf_binned\\acf_binned_' + k + '.csv',
                                  sua_list_binned, acf_full_l, verbose_=True)

    if calculate_trials:
        signal_len = int(30 * 60 * fs)
        n_lags = 20
        bin_size = 50  # in ms
        sttc_dt = int(49 * (fs / 1000))
        trial_len = int(n_lags * bin_size * (fs / 1000))

        n_trials = 40  # this is fixed based on experimental datasets
        m_iterations = 100

        # with open(data_folder + 'dataset\\cut_30min\\trial_dict.pkl', 'rb') as f:
        #     trial_dict = pickle.load(f)

        with open(data_folder + 'dataset\\cut_30min\\trial_binned_dict.pkl', 'rb') as f:
            trial_binned_dict = pickle.load(f)


        # output_log = data_folder + '\\dataset\\cut_30min\\trials_tau_log.txt'
        # old_stdout = sys.stdout
        # sys.stdout = open(output_log, 'w')

        items_to_process = 2000
        start_idx = 2000
        stop_idx = len(trial_binned_dict)

        if calculate_trials_pearsonr:
            print('Starting Pearsonr trial avg...')
            pearsonr_trial_avg_dict = {}
            for i, (k, v) in enumerate(islice(trial_binned_dict.items(), start_idx, stop_idx), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{(stop_idx-start_idx)}, {datetime.now()}')

                # on trials
                pearson_avg_l, pearson_avg_acf_l, pearson_avg_acf_matrix_l = [], [], []
                for m in range(m_iterations):
                    if (m % 50) == 0:
                        print(f'Sampling iteration {m}')
                    spikes_trials_binned = trial_binned_dict[k][m]

                    pearsonr_acf_matrix, pearsonr_acf_average = acf_pearsonr_trial_avg(spikes_trials_binned,
                                                                                       n_lags,
                                                                                       verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(pearsonr_acf_average,
                                                                                              start_idx_=1, exp_fun_=func_single_exp_monkey)
                    pearson_avg_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})
                    pearson_avg_acf_l.append(pearsonr_acf_average)
                    pearson_avg_acf_matrix_l.append(pearsonr_acf_matrix)

                pearsonr_trial_avg_dict[k] = {'taus': pearson_avg_l,
                                              'acf': pearson_avg_acf_l,
                                              'acf_matrix': pearson_avg_acf_matrix_l}

            with open(data_folder + '\\dataset\\cut_30min\\binned\\acf\\pearsonr_trial_avg_50ms_20lags_dict_test.pkl', "wb") as f:
                pickle.dump(pearsonr_trial_avg_dict, f)

        # if calculate_trials_sttc_avg:
        #     print('Starting STTC trial avg ...')
        #     sttc_trial_avg_dict = {}
        #     for i, (k, v) in enumerate(islice(trial_dict.items(), items_to_process), start=1):
        #         print(f'#####\nProcessing unit {k}, {i}/{items_to_process}, {datetime.now()}')
        #
        #         # on trials
        #         sttc_avg_l, sttc_avg_acf_l, sttc_avg_acf_matrix_l = [], [], []
        #         for m in range(m_iterations):
        #             if (m % 50) == 0:
        #                 print(f'Sampling iteration {m}')
        #             spikes_trials = trial_dict[k][m]
        #
        #             sttc_acf_matrix, sttc_acf_average = acf_sttc_trial_avg(spikes_trials,
        #                                                                    n_lags_=n_lags,
        #                                                                    lag_shift_=int(bin_size * (fs / 1000)),
        #                                                                    sttc_dt_=sttc_dt,
        #                                                                    zero_padding_len_=int(150 * (fs / 1000)),
        #                                                                    verbose_=False)
        #             fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(sttc_acf_average, start_idx_=1,
        #                                                                                                         exp_fun_=func_single_exp_monkey)
        #             sttc_avg_l.append({'tau':tau,
        #                                   'tau_lower':tau_ci[0],
        #                                   'tau_upper':tau_ci[1],
        #                                   'fit_r_squared': fit_r_squared,
        #                                   'explained_var': explained_var,
        #                                   'popt': fit_popt,
        #                                   'pcov': fit_pcov,
        #                                   'log_message': log_message})
        #             sttc_avg_acf_l.append(sttc_acf_average)
        #             sttc_avg_acf_matrix_l.append(sttc_acf_matrix)
        #
        #         sttc_trial_avg_dict[k] = {'taus': sttc_avg_l,
        #                                   'acf': sttc_avg_acf_l,
        #                                   'acf_matrix': sttc_avg_acf_matrix_l}
        #
        #     with open(data_folder + '\\dataset\\cut_30min\\non_binned\\acf\\sttc_trial_avg_50ms_20lags_dict_test.pkl', "wb") as f:
        #         pickle.dump(sttc_trial_avg_dict, f)

        # if calculate_trials_sttc_concat:
        #     print('Starting STTC trial concat ...')
        #     sttc_trial_concat_dict = {}
        #     for i, (k, v) in enumerate(islice(trial_dict.items(), items_to_process), start=1):
        #         print(f'#####\nProcessing unit {k}, {i}/{items_to_process}, {datetime.now()}')
        #
        #         # on trials
        #         sttc_concat_l, sttc_concat_acf_l= [], []
        #         for m in range(m_iterations):
        #             if (m % 50) == 0:
        #                 print(f'Sampling iteration {m}')
        #             spikes_trials = trial_dict[k][m]
        #
        #             acf_concat = acf_sttc_trial_concat(spikes_trials,
        #                                                n_lags_=n_lags,
        #                                                lag_shift_=int(bin_size * (fs / 1000)),
        #                                                sttc_dt_=sttc_dt,
        #                                                trial_len_=trial_len,
        #                                                zero_padding_len_=int(3000 * (fs / 1000)),
        #                                                verbose_=False)
        #             fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(acf_concat, start_idx_=1,
        #                                                                                                         exp_fun_=func_single_exp_monkey)
        #             sttc_concat_l.append({'tau':tau,
        #                                   'tau_lower':tau_ci[0],
        #                                   'tau_upper':tau_ci[1],
        #                                   'fit_r_squared': fit_r_squared,
        #                                   'explained_var': explained_var,
        #                                   'popt': fit_popt,
        #                                   'pcov': fit_pcov,
        #                                   'log_message': log_message})
        #             sttc_concat_acf_l.append(acf_concat)
        #
        #         sttc_trial_concat_dict[k] = {'taus': sttc_concat_l,
        #                                      'acf': sttc_concat_acf_l}
        #
        #     with open(data_folder + '\\dataset\\cut_30min\\non_binned\\acf\\sttc_trial_concat_50ms_20lags_dict_test.pkl', "wb") as f:
        #         pickle.dump(sttc_trial_concat_dict, f)

        # sys.stdout = old_stdout








