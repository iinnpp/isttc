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

if __name__ == "__main__":
    dataset_folder = project_folder_path + 'results\\allen_mice\\dataset\\cut_30min\\'

    # load data
    csv_data_file = dataset_folder + 'sua_list_constrained.csv'
    with open(csv_data_file, newline='') as f:
        reader = csv.reader(f)
        sua_list = list(reader)
    print(f'Loaded N units {len(sua_list)}')

    # estimate M
    fs = 30000
    signal_len = int(30 * 60 * fs)
    n_lags = 20
    bin_size = 50  # in ms
    trial_len = int(n_lags * bin_size * (fs / 1000))

    n_trials = 40  # this is fixed based on experimental datasets
    m_iterations = [20, 40, 60, 80, 100, 150, 200, 500, 1000]

    n_signals = [20, 50, 100]

    for n_signal in n_signals:
        print(f'Running for n_signal = {n_signal}')
        random_signals = random.sample(sua_list, n_signal)

        output_log = dataset_folder + f'resampling//resampling_params_estimate_{n_signal}_signals.txt'
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

            signal_tau_dict[signal_idx] = tau_dict

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_signal}_signals.pkl', "wb") as f:
            pickle.dump(signal_tau_dict, f)

        sys.stdout = old_stdout
