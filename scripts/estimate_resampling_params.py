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
    trial_len = int(n_lags * bin_size * (fs / 1000))

    n_trials = 40  # this is fixed based on experimental datasets
    #m_iterations = [20, 40, 60, 80, 100, 150, 200, 500, 1000]
    m_iterations = [50, 100, 200, 500, 1000]

    n_total_signal_l = [100]
    min_signal_per_area_l = [10]

    for n_total_signals, min_signal_per_area in zip(n_total_signal_l, min_signal_per_area_l):
        print(f'Running for n_total_signals = {n_total_signals}, min_signal_per_area = {min_signal_per_area}')
        units_to_sample = sample_signals(units_info_df, min_signal_per_area, n_total_signals)
        #print(units_to_sample)
        random_signals = [item for item in sua_list if item[2] in units_to_sample]
        print(len(random_signals))
        #random_signals = random.sample(sua_list, n_signal)

        output_log = dataset_folder + f'resampling//resampling_params_estimate_{n_total_signals}_signals.txt'
        old_stdout = sys.stdout
        sys.stdout = open(output_log, 'w')

        signal_tau_dict = {}

        for signal_idx, signal in enumerate(random_signals):
            print(f'###\nCalculating for {signal_idx} signal')
            spikes = np.asarray([int(spike) for spike in signal[8:]])

            tau_dict = {}
            for m_iteration in m_iterations:
                print(f'calculating for {m_iteration} resampling iterations')
                tau_l = []
                for m in range(m_iteration):
                    spikes_trials = get_trials(spikes, signal_len, n_trials, trial_len, verbose_=False)
                    spikes_trials_binned = bin_trials(spikes_trials, trial_len, int(bin_size * (fs / 1000)))
                    # get taus
                    _, acf_average = acf_pearsonr_trial_avg(spikes_trials_binned, n_lags, verbose_=False)
                    _, _, tau, _, _, _, _ = fit_single_exp(acf_average, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    tau_l.append(tau)
                tau_dict[m_iteration] = tau_l

            signal_tau_dict[signal[2]] = tau_dict

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals.pkl', "wb") as f:
            pickle.dump(signal_tau_dict, f)

        sys.stdout = old_stdout
