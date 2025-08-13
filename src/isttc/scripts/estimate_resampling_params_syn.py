import pickle
import numpy as np

from src.isttc.scripts.calculate_acf import acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat
from src.isttc.scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from src.isttc.scripts.spike_train_utils import get_trials, bin_trials
from src.isttc.scripts.cfg_global import project_folder_path


# ========== Parameters ==========
# ACF calculation params
fs = 1000  # data sampling frequency in Hz
n_lags = 20 # number of lags to calculate autocorrelation
bin_size = int(50 * fs / 1000) # in ms
sttc_dt_full = int(25 * fs / 1000) # dt for iSTTC on full signal and on concat trials, in ms
sttc_dt_avg = int(50 * (fs / 1000) - 1) # dt for iSTTC in trial-averaged style (like PearsonR)
n_trials = 40  # this is fixed based on experimental datasets
trial_len = int(n_lags * bin_size * (fs / 1000))
m_iterations = [50, 100, 200, 500, 1000]
n_total_signal_l = [100] # sampled from the whole dataset
signal_len = int(10 * 60 * fs)  # duration_ms

# File paths
dataset_folder = project_folder_path + 'results\\synthetic_data\\dataset\\'

# ========== Main ==========
if __name__ == "__main__":
    np.random.seed(42)
    # load data
    spike_trains_10min = np.load(dataset_folder + 'spike_trains_tau100ms_alpha0_3_fr3_5hz_len600sec_1000.npy', allow_pickle=True)
    print(f'n spike trains {len(spike_trains_10min)}, len {spike_trains_10min[0][-1]/1000}')

    # estimate M
    for n_total_signals in n_total_signal_l:
        print(f'Running for n_total_signals = {n_total_signals}')
        units_to_sample = np.random.choice(len(spike_trains_10min), size=n_total_signals, replace=False)
        random_signals = spike_trains_10min[units_to_sample]
        print(len(random_signals))

        # output_log = dataset_folder + f'resampling//resampling_params_estimate_{n_total_signals}_signals.txt'
        # old_stdout = sys.stdout
        # sys.stdout = open(output_log, 'w')

        # dict: key is unit_id and value taus for all m_iterations
        pearsonr_avg_taus_dict, sttc_avg_taus_dict, sttc_concat_taus_dict = {}, {}, {}
        for signal_idx, signal in enumerate(random_signals):
            print(f'###\nCalculating for {signal_idx} signal')
            spikes = np.asarray([int(spike) for spike in signal])
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
                    fit = fit_single_exp(acf_average, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    pearsonr_avg_taus_l.append({
                        'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                        'fit_r_squared': fit[4], 'explained_var': fit[5],
                        'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                    })
                    # isttc avg
                    _, sttc_acf_average = acf_sttc_trial_avg(spikes_trials, n_lags_=n_lags,
                                                                           lag_shift_=bin_size,
                                                                           sttc_dt_=sttc_dt_avg,
                                                                           zero_padding_len_=int(150 * (fs / 1000)),
                                                                           verbose_=False)
                    fit = fit_single_exp(sttc_acf_average, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    sttc_avg_taus_l.append({
                        'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                        'fit_r_squared': fit[4], 'explained_var': fit[5],
                        'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                    })
                    # isttc concat
                    acf_concat = acf_sttc_trial_concat(spikes_trials, n_lags_=n_lags,
                                                       lag_shift_=bin_size,
                                                       sttc_dt_=sttc_dt_full,
                                                       trial_len_=trial_len,
                                                       zero_padding_len_=int(3000 * (fs / 1000)),
                                                       verbose_=False)
                    fit = fit_single_exp(acf_concat, start_idx_=1, exp_fun_=func_single_exp_monkey)
                    sttc_concat_taus_l.append({
                        'tau': fit[2], 'tau_lower': fit[3][0], 'tau_upper': fit[3][1],
                        'fit_r_squared': fit[4], 'explained_var': fit[5],
                        'popt': fit[0], 'pcov': fit[1], 'log_message': fit[6]
                    })

                pearsonr_avg_iteration_taus_dict[m_iteration] = pearsonr_avg_taus_l
                sttc_avg_iteration_taus_dict[m_iteration] = sttc_avg_taus_l
                sttc_concat_iteration_taus_dict[m_iteration] = sttc_concat_taus_l

            pearsonr_avg_taus_dict[units_to_sample[signal_idx]] = pearsonr_avg_iteration_taus_dict
            sttc_avg_taus_dict[units_to_sample[signal_idx]] = sttc_avg_iteration_taus_dict
            sttc_concat_taus_dict[units_to_sample[signal_idx]] = sttc_concat_iteration_taus_dict

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_pearsonr_avg.pkl', "wb") as f:
            pickle.dump(pearsonr_avg_taus_dict, f)

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_sttc_avg.pkl', "wb") as f:
            pickle.dump(sttc_avg_taus_dict, f)

        with open(dataset_folder + f'resampling//signal_tau_dict_{n_total_signals}_signals_sttc_concat.pkl', "wb") as f:
            pickle.dump(sttc_concat_taus_dict, f)

        # sys.stdout = old_stdout
