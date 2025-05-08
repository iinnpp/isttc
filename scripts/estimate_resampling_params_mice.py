import csv
import sys
import pickle
import numpy as np
import pandas as pd
import random
from scipy import stats

from scripts.calculate_acf import acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat
from scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from scripts.spike_train_utils import get_trials, bin_trials
from scripts.cfg_global import project_folder_path


def sample_signals(units_df, min_per_area, n_total):
    sampled_units = []

    area_counts = units_df["ecephys_structure_acronym"].value_counts()
    base_samples = {area: min(min_per_area, count) for area, count in area_counts.items()}
    remaining_samples = n_total - sum(base_samples.values())
    total_remaining = sum(area_counts) - sum(base_samples.values())
    print(
        f'total {n_total}, min per area {min_per_area}. Base samples {base_samples}, \nproportional samples {remaining_samples}')

    # get min_per_area, then sample proportionally
    for area, count in area_counts.items():
        base_samples = min(min_per_area, count)
        extra_samples = int((count / total_remaining) * remaining_samples) if total_remaining > 0 else 0
        n_samples = base_samples + extra_samples
        sampled_units.extend(units_info_df[units_info_df["ecephys_structure_acronym"] == area]
                             .sample(n=min(n_samples, count), random_state=42)["unit_id"].tolist())

    return sampled_units

def load_sampled_signals(file_path_):
    with open(file_path_, "rb") as f:
        signal_tau_dict_ = pickle.load(f)
    print(f'N signals {len(signal_tau_dict_)}')
    sampled_units = list(signal_tau_dict_.keys())
    return sampled_units


if __name__ == "__main__":
    dataset_folder = project_folder_path + 'results\\allen_mice\\dataset\\cut_30min\\'

    # load data
    csv_data_file = dataset_folder + 'sua_list_constrained.csv'
    with open(csv_data_file, newline='') as f:
        reader = csv.reader(f)
        sua_list = list(reader)
    print(f'Loaded N units {len(sua_list)}')
    units_info_df = pd.read_pickle(dataset_folder + 'sua_list_constrained_units_df.pkl')

    # estimate M
    fs = 30000
    signal_len = int(30 * 60 * fs)
    n_lags = 20
    bin_size = 50  # in ms
    sttc_dt_avg = int(50 * (fs / 1000) - 1)
    sttc_dt_concat = int(25 * (fs / 1000))
    trial_len = int(n_lags * bin_size * (fs / 1000))

    n_trials = 40  # this is fixed based on experimental datasets
    m_iterations = [50, 100, 200, 500, 1000]

    n_total_signal_l = [100]
    min_signal_per_area_l = [10]

    for n_total_signals, min_signal_per_area in zip(n_total_signal_l, min_signal_per_area_l):
        print(f'Running for n_total_signals = {n_total_signals}, min_signal_per_area = {min_signal_per_area}')
        # units_to_sample = sample_signals(units_info_df, min_signal_per_area, n_total_signals)
        units_to_sample = load_sampled_signals(dataset_folder + f'resampling//signal_tau_dict_100_signals_pearsonr.pkl')
        #print(units_to_sample)
        random_signals = [item for item in sua_list if item[2] in units_to_sample]
        print(len(random_signals))
        #random_signals = random.sample(sua_list, n_signal)

        # output_log = dataset_folder + f'resampling//resampling_params_estimate_{n_total_signals}_signals.txt'
        # old_stdout = sys.stdout
        # sys.stdout = open(output_log, 'w')

        # dict: key is unit_id and value taus for all m_iterations
        pearsonr_avg_taus_dict, sttc_avg_taus_dict, sttc_concat_taus_dict = {}, {}, {}

        for signal_idx, signal in enumerate(random_signals):
            print(f'###\nCalculating for {signal_idx} signal')
            spikes = np.asarray([int(spike) for spike in signal[8:]])
            # dict: key is m_iteration and value taus for all iterations (e.g. for m_iteration=50 50 taus)
            pearsonr_avg_iteration_taus_dict, sttc_avg_iteration_taus_dict, sttc_concat_iteration_taus_dict = {}, {}, {}

            for m_iteration in m_iterations:
                print(f'calculating for {m_iteration} resampling iterations')
                pearsonr_avg_taus_l, sttc_avg_taus_l, sttc_concat_taus_l = [], [], []
                for m in range(m_iteration):
                    spikes_trials = get_trials(spikes, signal_len, n_trials, trial_len, verbose_=False)
                    spikes_trials_binned = bin_trials(spikes_trials, trial_len, int(bin_size * (fs / 1000)))
                    # pearsonr
                    _, acf_average = acf_pearsonr_trial_avg(spikes_trials_binned, n_lags, verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(
                        acf_average, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    pearsonr_avg_taus_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})
                    # sttc avg
                    _, sttc_acf_average = acf_sttc_trial_avg(spikes_trials, n_lags_=n_lags,
                                                                           lag_shift_=int(bin_size * (fs / 1000)),
                                                                           sttc_dt_=sttc_dt_avg,
                                                                           zero_padding_len_=int(150 * (fs / 1000)),
                                                                           verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(
                        sttc_acf_average, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    sttc_avg_taus_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})
                    # sttc concat
                    acf_concat = acf_sttc_trial_concat(spikes_trials, n_lags_=n_lags,
                                                       lag_shift_=int(bin_size * (fs / 1000)),
                                                       sttc_dt_=sttc_dt_concat,
                                                       trial_len_=trial_len,
                                                       zero_padding_len_=int(3000 * (fs / 1000)),
                                                       verbose_=False)
                    fit_popt, fit_pcov, tau, tau_ci, fit_r_squared, explained_var, log_message = fit_single_exp(
                        acf_concat, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    sttc_concat_taus_l.append({'tau':tau,
                                          'tau_lower':tau_ci[0],
                                          'tau_upper':tau_ci[1],
                                          'fit_r_squared': fit_r_squared,
                                          'explained_var': explained_var,
                                          'popt': fit_popt,
                                          'pcov': fit_pcov,
                                          'log_message': log_message})

                pearsonr_avg_iteration_taus_dict[m_iteration] = pearsonr_avg_taus_l
                sttc_avg_iteration_taus_dict[m_iteration] = sttc_avg_taus_l
                sttc_concat_iteration_taus_dict[m_iteration] = sttc_concat_taus_l

            pearsonr_avg_taus_dict[signal[2]] = pearsonr_avg_iteration_taus_dict
            sttc_avg_taus_dict[signal[2]] = sttc_avg_iteration_taus_dict
            sttc_concat_taus_dict[signal[2]] = sttc_concat_iteration_taus_dict

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_pearsonr_avg.pkl', "wb") as f:
            pickle.dump(pearsonr_avg_taus_dict, f)

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_sttc_avg.pkl', "wb") as f:
            pickle.dump(sttc_avg_taus_dict, f)

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_sttc_concat.pkl', "wb") as f:
            pickle.dump(sttc_concat_taus_dict, f)

        # sys.stdout = old_stdout
