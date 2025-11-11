import numpy as np
from itertools import islice
import pickle
from datetime import datetime
from statsmodels.tsa.stattools import acf

from src.isttc.scripts.calculate_acf import acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat, acf_sttc
from src.isttc.scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from src.isttc.scripts.cfg_global import project_folder_path
from src.isttc.scripts.spike_train_utils import bin_spike_train_fixed_len

# ========== Parameters ==========
# ACF calculation params
fs = 1000  # data sampling frequency in Hz
n_lags = 20 # number of lags to calculate autocorrelation
bin_size = int(50 * fs / 1000) # in ms
sttc_dt_full = int(25 * fs / 1000) # dt for iSTTC on full signal and on concat trials, in ms
sttc_dt_avg = int(50 * (fs / 1000) - 1) # dt for iSTTC in trial-averaged style (like PearsonR)
m_iterations = 1 # number of trials sampling iterations

# File paths
dataset_folder = project_folder_path + 'synthetic_dataset\\'
results_folder = project_folder_path + 'results\\synthetic\\results\\param_fr_alpha_tau_zeros\\'

# Execution flags
calculate_acf_full = False
calculate_sttc_full = False
calculate_trials_pearsonr = False
calculate_trials_sttc_avg = False
calculate_trials_sttc_concat = True

# ========== Main ==========
if __name__ == "__main__":

    # === Classic ACF Analysis ===
    if calculate_acf_full:
        print('[FULL] Starting ACF...')
        with open(dataset_folder + '1_trial_3params_var_len450sec_100000_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        spike_trains = data['spike_trains']
        duration_ms = data['duration_ms']
        print(f'[INFO] Loaded {len(spike_trains)} spike trains')

        # bin
        all_binned = [
            bin_spike_train_fixed_len(
                [int(spike) for spike in st], bin_size, fs, duration_ms[i], verbose_=False
            )
            for i, st in enumerate(spike_trains)
        ]
        # get acf
        acf_full_l = []
        for idx, binned in enumerate(all_binned):
            if idx % 100 == 0:
                print(f'[ACF] Processing unit {idx}')
            acf_full_l.append(acf(binned, nlags=n_lags))
        # get tau
        acf_results = {}
        for idx, acf_vals in enumerate(acf_full_l):
            if idx % 100 == 0:
                print(f'[ACF] Fitting unit {idx + 1}/{len(acf_full_l)} ({datetime.now()})')
            fit = fit_single_exp(acf_vals, start_idx_=1, exp_fun_=func_single_exp_monkey)
            acf_results[idx] = {
                'taus': {
                    'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                    'fit_r_squared': fit[4], 'explained_var': fit[5],
                    'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                },
                'acf': acf_vals
            }
        # save
        with open(results_folder + 'acf_full_50ms_20lags_len_300sec_dict.pkl', "wb") as f:
            pickle.dump(acf_results, f)

    # === iSTTC Analysis ===
    if calculate_sttc_full:
        print('[FULL] Starting iSTTC...')
        with open(dataset_folder + '1_trial_3params_var_len450sec_100000_dict.pkl', 'rb') as f:
            data = pickle.load(f)
        spike_trains = data['spike_trains']
        duration_ms = data['duration_ms']
        print(f'[INFO] Loaded {len(spike_trains)} spike trains')

        # get acf
        isttc_full_l = []
        for idx, st in enumerate(spike_trains):
            if idx % 100 == 0:
                print(f'[iSTTC] Processing unit {idx}')
            st_int = np.asarray([int(spike) for spike in st])
            isttc_full_l.append(
                acf_sttc(st_int, n_lags, bin_size, sttc_dt_full, duration_ms[idx], verbose_=False)
            )
        # get tau
        isttc_results = {}
        for idx, acf_vals in enumerate(isttc_full_l):
            if idx % 100 == 0:
                print(f'[iSTTC] Fitting unit {idx + 1}/{len(isttc_full_l)} ({datetime.now()})')
            fit = fit_single_exp(acf_vals, start_idx_=1, exp_fun_=func_single_exp_monkey)
            isttc_results[idx] = {
                'taus': {
                    'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                    'fit_r_squared': fit[4], 'explained_var': fit[5],
                    'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                },
                'acf': acf_vals
            }
        # save
        with open(results_folder + 'acf_isttc_full_50ms_20lags_len_450sec_dict.pkl', "wb") as f:
            pickle.dump(isttc_results, f)

    # === Pearsonr Trial Avg ===
    if calculate_trials_pearsonr:
        print('[TRIAL] Starting Pearsonr trial avg...')
        with open(dataset_folder + '1_trial_binned_3params_var_len600sec_100000_80trials_dict.pkl', 'rb') as f:
            trial_binned_dict_full = pickle.load(f)
        trial_binned_dict = trial_binned_dict_full['trial_dict']
        n_trials_all = trial_binned_dict_full['n_trials']
        trial_lens_all = trial_binned_dict_full['trial_lens']

        pearsonr_results = {}
        start_idx, stop_idx = 0, len(trial_binned_dict)
        for i, (k, v) in enumerate(islice(trial_binned_dict.items(), start_idx, stop_idx), start=1):
            print(f'[TRIAL] Pearsonr - Unit {k} ({i}/{stop_idx - start_idx}, {datetime.now()}')
            n_trials = n_trials_all[k]
            trial_len = trial_lens_all[k]
            n_lags = int(trial_len / bin_size)
            print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

            pearsonr_taus_l, pearsonr_acf_l, pearsonr_acf_matrix_l = [], [], []
            for m in range(m_iterations):
                if (m % 50) == 0:
                    print(f'Sampling iteration {m}')
                spikes_trials_binned = trial_binned_dict[k][m]
                acf_matrix, acf_avg = acf_pearsonr_trial_avg(spikes_trials_binned, n_lags, verbose_=False)
                fit = fit_single_exp(acf_avg, start_idx_=1, exp_fun_=func_single_exp_monkey)
                pearsonr_taus_l.append({
                    'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                    'fit_r_squared': fit[4], 'explained_var': fit[5],
                    'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                })
                pearsonr_acf_l.append(acf_avg)
                pearsonr_acf_matrix_l.append(acf_matrix)
            pearsonr_results[k] = {'taus': pearsonr_taus_l,
                                   'acf': pearsonr_acf_l,
                                   'acf_matrix': pearsonr_acf_matrix_l}

        with open(results_folder + 'pearsonr_trial_avg_50ms_dict.pkl', "wb") as f:
            pickle.dump(pearsonr_results, f)

    # === iSTTC Trial Avg ===
    if calculate_trials_sttc_avg:
        print('[TRIAL] Starting iSTTC trial avg...')
        with open(dataset_folder + '1_trial_3params_var_len600sec_100000_80trials_dict.pkl', 'rb') as f:
            trial_dict_full = pickle.load(f)
        trial_dict = trial_dict_full['trial_dict']
        n_trials_all = trial_dict_full['n_trials']
        trial_lens_all = trial_dict_full['trial_lens']

        isttc_avg_results = {}
        start_idx, stop_idx = 0, len(trial_dict)
        for i, (k, v) in enumerate(islice(trial_dict.items(), start_idx, stop_idx), start=1):
            print(f'[TRIAL] iSTTC avg - Unit {k} ({i}/{stop_idx - start_idx}, {datetime.now()}')
            n_trials = n_trials_all[k]
            trial_len = trial_lens_all[k]
            n_lags = int(trial_len / bin_size)
            print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

            isttc_avg_taus_l, isttc_avg_acf_l, isttc_avg_acf_matrix_l = [], [], []
            for m in range(m_iterations):
                if (m % 50) == 0:
                    print(f'Sampling iteration {m}')
                spikes_trials = trial_dict[k][m]
                acf_matrix, acf_avg = acf_sttc_trial_avg(spikes_trials,
                                                         n_lags_=n_lags,
                                                         lag_shift_=bin_size,
                                                         sttc_dt_=sttc_dt_avg,
                                                         zero_padding_len_=int(150 * (fs / 1000)), # 150 ms is default
                                                         verbose_=False)
                fit = fit_single_exp(acf_avg, start_idx_=1, exp_fun_=func_single_exp_monkey)
                isttc_avg_taus_l.append({
                    'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                    'fit_r_squared': fit[4], 'explained_var': fit[5],
                    'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                })
                isttc_avg_acf_l.append(acf_avg)
                isttc_avg_acf_matrix_l.append(acf_matrix)
            isttc_avg_results[k] = {'taus': isttc_avg_taus_l,
                                    'acf': isttc_avg_acf_l,
                                    'acf_matrix': isttc_avg_acf_matrix_l}

        with open(results_folder + 'sttc_trial_avg_50ms_dict.pkl', "wb") as f:
            pickle.dump(isttc_avg_results, f)

    # === iSTTC Trial Concat ===
    if calculate_trials_sttc_concat:
        print('[TRIAL] Starting iSTTC trial concat...')
        with open(dataset_folder + 'trials40.pkl', 'rb') as f:
            trial_dict_full = pickle.load(f)
        trial_dict = trial_dict_full['trial_dict']
        n_trials_all = trial_dict_full['n_trials']
        trial_lens_all = trial_dict_full['trial_lens']

        isttc_concat_results = {}
        start_idx, stop_idx = 0, 100 # len(trial_dict)
        for i, (k, v) in enumerate(islice(trial_dict.items(), start_idx, stop_idx), start=1):
            print(f'[TRIAL] iSTTC concat - Unit {k} ({i}/{stop_idx - start_idx}, {datetime.now()}')
            n_trials = n_trials_all[k]
            trial_len = trial_lens_all[k]
            n_lags = int(trial_len / bin_size)
            print(f'\nn_trials: {n_trials}, trial_len: {trial_len}, n_lags: {n_lags}')

            isttc_concat_taus_l, isttc_concat_acf_l= [], []
            for m in range(m_iterations):
                if (m % 50) == 0:
                    print(f'Sampling iteration {m}')
                spikes_trials = trial_dict[k][m]
                acf_concat = acf_sttc_trial_concat(spikes_trials,
                                                   n_lags_=n_lags,
                                                   lag_shift_=bin_size,
                                                   sttc_dt_=sttc_dt_full,
                                                   trial_len_=trial_len,
                                                   zero_padding_len_=int(3*trial_len * (fs / 1000)), # 3 is default
                                                   verbose_=False)
                fit = fit_single_exp(acf_concat, start_idx_=1, exp_fun_=func_single_exp_monkey)
                isttc_concat_taus_l.append({
                    'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                    'fit_r_squared': fit[4], 'explained_var': fit[5],
                    'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                })
                isttc_concat_acf_l.append(acf_concat)
            isttc_concat_results[k] = {'taus': isttc_concat_taus_l,
                                       'acf': isttc_concat_acf_l}

        with open(results_folder + 'sttc_trial_concat_50ms_40_trials_dict.pkl', "wb") as f:
            pickle.dump(isttc_concat_results, f)









