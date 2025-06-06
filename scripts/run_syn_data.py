import numpy as np
from itertools import islice
import pickle
from datetime import datetime
from statsmodels.tsa.stattools import acf

from scripts.calculate_acf import acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat, acf_sttc
from scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from scripts.cfg_global import project_folder_path
from scripts.spike_train_utils import bin_spike_train_fixed_len

if __name__ == "__main__":
    dataset_folder = project_folder_path + 'results\\synthetic\\dataset\\'
    results_folder = project_folder_path + 'results\\synthetic\\results\\fixed_params_sttc_vs_zeros\\'

    calculate_trials = True

    calculate_acf_full = False
    calculate_sttc_full = False
    calculate_trials_pearsonr = False
    calculate_trials_sttc_avg = True
    calculate_trials_sttc_concat = True

    # params
    fs = 1000
    signal_len = int(10 * 60 * fs)
    n_lags = 20
    bin_size = int(50 * (fs / 1000))
    sttc_dt_full = int(25 * (fs / 1000))

    if calculate_acf_full or calculate_sttc_full:
        with open(dataset_folder + 'spike_trains_3params_var_len600sec_100000.pkl', 'rb') as f:
            data = pickle.load(f)

        spike_trains = data['spike_trains']
        print(f'n spike trains {len(spike_trains)}, len {spike_trains[0][-1] / 1000}')


        if calculate_acf_full:
            # bin
            all_spike_trains_binned_l = []
            for i in range(len(spike_trains)):
                binned_spike_train = bin_spike_train_fixed_len([int(spike) for spike in spike_trains[i]],
                                                               bin_size, fs, signal_len,
                                                               verbose_=False)
                all_spike_trains_binned_l.append(binned_spike_train)

            # get acf
            acf_full_l = []
            for unit_idx, unit in enumerate(all_spike_trains_binned_l):
                if unit_idx % 100 == 0:
                    print(f'Processing unit {unit_idx}')
                spike_train_binned_acf = acf(unit, nlags=n_lags)
                acf_full_l.append(spike_train_binned_acf)

            # get tau
            acf_full_dict = {}
            for unit_id_idx, unit_acf in enumerate(acf_full_l):
                if unit_id_idx % 100 == 0:
                    print(
                        f'#####\nProcessing unit {unit_id_idx + 1}/{len(acf_full_l)}, {datetime.now()}')
                fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(
                    unit_acf, start_idx_=1, exp_fun_=func_single_exp_monkey)
                taus = {'tau': tau,
                        'tau_lower': tau_ci[0],
                        'tau_upper': tau_ci[1],
                        'fit_r_squared': fit_r_squared,
                        'explained_var': explained_var,
                        'popt': fit_popt,
                        'pcov': fit_pcov,
                        'log_message': log_message}
                acf_full_dict[unit_id_idx] = {'taus': taus, 'acf': unit_acf}

            # save
            with open(results_folder + 'acf_full_50ms_20lags_dict.pkl', "wb") as f:
                pickle.dump(acf_full_dict, f)

        if calculate_sttc_full:
            # get acf
            acf_isttc_full_l = []
            for unit_idx, unit in enumerate(spike_trains):
                if unit_idx % 100 == 0:
                    print(f'Processing unit {unit_idx}')
                spike_train_int = np.asarray([int(spike) for spike in unit])
                spike_train_acf = acf_sttc(spike_train_int, n_lags, bin_size, sttc_dt_full, signal_len, verbose_=False)
                acf_isttc_full_l.append(spike_train_acf)

            # get tau
            isttc_full_dict = {}
            for unit_id_idx, unit_acf in enumerate(acf_isttc_full_l):
                if unit_id_idx % 100 == 0:
                    print(
                        f'#####\nProcessing unit {unit_id_idx + 1}/{len(acf_isttc_full_l)}, {datetime.now()}')
                fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(
                    unit_acf, start_idx_=1, exp_fun_=func_single_exp_monkey)
                taus = {'tau': tau,
                        'tau_lower': tau_ci[0],
                        'tau_upper': tau_ci[1],
                        'fit_r_squared': fit_r_squared,
                        'explained_var': explained_var,
                        'popt': fit_popt,
                        'pcov': fit_pcov,
                        'log_message': log_message}
                isttc_full_dict[unit_id_idx] = {'taus': taus, 'acf': unit_acf}

            # save
            with open(results_folder + 'acf_isttc_full_50ms_20lags_dict.pkl', "wb") as f:
                pickle.dump(isttc_full_dict, f)


    if calculate_trials:

        # n_lags = 20
        #bin_size = 50  # in ms
        sttc_dt_avg = int(50 * (fs / 1000) - 1)
        sttc_dt_concat = int(25 * (fs / 1000))
        trial_len = int(n_lags * bin_size * (fs / 1000))

        # n_trials = 40
        m_iterations = 1 # 100

        with open(dataset_folder + '0_trial_tau100ms_alpha0_3_fr3_5hz_len600sec_1000_dict.pkl', 'rb') as f:
        # with open(dataset_folder + '1_trial_3params_var_len600sec_100000_dict.pkl', 'rb') as f:
            trial_dict_full = pickle.load(f)
        trial_dict = trial_dict_full['trial_dict']
        n_trials_all = trial_dict_full['n_trials']
        trial_lens_all = trial_dict_full['trial_lens']

        with open(dataset_folder + '0_trial_binned_tau100ms_alpha0_3_fr3_5hz_len600sec_1000_dict.pkl', 'rb') as f:
        # with open(dataset_folder + '1_trial_binned_3params_var_len600sec_100000_dict.pkl',
        #              'rb') as f:
            trial_binned_dict_full = pickle.load(f)
        trial_binned_dict = trial_binned_dict_full['trial_dict']
        # n_trials_all = trial_binned_dict_full['n_trials']
        # trial_lens_all = trial_binned_dict_full['trial_lens']


        # output_log = data_folder + '\\dataset\\cut_30min\\trials_tau_log.txt'
        # old_stdout = sys.stdout
        # sys.stdout = open(output_log, 'w')

        #items_to_process = 10
        start_idx = 0
        stop_idx = 100 # len(trial_dict)

        if calculate_trials_pearsonr:
            print('Starting Pearsonr trial avg...')
            pearsonr_trial_avg_dict = {}
            for i, (k, v) in enumerate(islice(trial_binned_dict.items(), start_idx, stop_idx), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{(stop_idx-start_idx)}, {datetime.now()}')

                n_trials = n_trials_all[k]
                trial_len = trial_lens_all[k]
                n_lags = int(trial_len / bin_size)
                print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

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

            with open(results_folder + 'pearsonr_trial_avg_50ms_dict.pkl', "wb") as f:
                pickle.dump(pearsonr_trial_avg_dict, f)

        if calculate_trials_sttc_avg:
            print('Starting STTC trial avg ...')
            sttc_trial_avg_dict = {}
            for i, (k, v) in enumerate(islice(trial_dict.items(), start_idx, stop_idx), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{(stop_idx-start_idx)}, {datetime.now()}')

                n_trials = n_trials_all[k]
                trial_len = trial_lens_all[k]
                n_lags = int(trial_len / bin_size)
                print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

                # on trials
                sttc_avg_l, sttc_avg_acf_l, sttc_avg_acf_matrix_l = [], [], []
                for m in range(m_iterations):
                    if (m % 50) == 0:
                        print(f'Sampling iteration {m}')
                    spikes_trials = trial_dict[k][m]

                    sttc_acf_matrix, sttc_acf_average = acf_sttc_trial_avg(spikes_trials,
                                                                           n_lags_=n_lags,
                                                                           lag_shift_=int(bin_size * (fs / 1000)),
                                                                           sttc_dt_=sttc_dt_avg,
                                                                           zero_padding_len_=int(50 * (fs / 1000)), # 150 is default
                                                                           verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(sttc_acf_average, start_idx_=1,
                                                                                                                exp_fun_=func_single_exp_monkey)
                    sttc_avg_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})
                    sttc_avg_acf_l.append(sttc_acf_average)
                    sttc_avg_acf_matrix_l.append(sttc_acf_matrix)

                sttc_trial_avg_dict[k] = {'taus': sttc_avg_l,
                                          'acf': sttc_avg_acf_l,
                                          'acf_matrix': sttc_avg_acf_matrix_l}

            with open(results_folder + 'sttc_trial_avg_50ms_dict.pkl', "wb") as f:
                pickle.dump(sttc_trial_avg_dict, f)

        if calculate_trials_sttc_concat:
            print('Starting STTC trial concat ...')
            sttc_trial_concat_dict = {}
            for i, (k, v) in enumerate(islice(trial_dict.items(), start_idx, stop_idx), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{(stop_idx - start_idx)}, {datetime.now()}')

                n_trials = n_trials_all[k]
                trial_len = trial_lens_all[k]
                n_lags = int(trial_len / bin_size)
                print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

                # on trials
                sttc_concat_l, sttc_concat_acf_l= [], []
                for m in range(m_iterations):
                    if (m % 50) == 0:
                        print(f'Sampling iteration {m}')
                    spikes_trials = trial_dict[k][m]

                    acf_concat = acf_sttc_trial_concat(spikes_trials,
                                                       n_lags_=n_lags,
                                                       lag_shift_=int(bin_size * (fs / 1000)),
                                                       sttc_dt_=sttc_dt_concat,
                                                       trial_len_=trial_len,
                                                       zero_padding_len_=int(1*trial_len * (fs / 1000)), # 3 is default
                                                       verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(acf_concat, start_idx_=1,
                                                                                                                exp_fun_=func_single_exp_monkey)
                    sttc_concat_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})
                    sttc_concat_acf_l.append(acf_concat)

                sttc_trial_concat_dict[k] = {'taus': sttc_concat_l,
                                             'acf': sttc_concat_acf_l}

            with open(results_folder + 'sttc_trial_concat_50ms_dict.pkl', "wb") as f:
                pickle.dump(sttc_trial_concat_dict, f)

        # sys.stdout = old_stdout








