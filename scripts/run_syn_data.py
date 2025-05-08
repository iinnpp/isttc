from itertools import islice
import pickle
from datetime import datetime
from statsmodels.tsa.stattools import acf

from scripts.calculate_acf import acf_pearsonr_trial_avg, acf_sttc_trial_avg, acf_sttc_trial_concat
from scripts.calculate_tau import fit_single_exp, func_single_exp_monkey
from scripts.cfg_global import project_folder_path


if __name__ == "__main__":
    data_folder = project_folder_path + 'results\\synthetic_data\\'
    fs = 1000

    calculate_trials = True
    calculate_trials_pearsonr = False
    calculate_trials_sttc_avg = False
    calculate_trials_sttc_concat = True


    if calculate_trials:
        signal_len = int(10 * 60 * fs)
        n_lags = 20
        bin_size = 50  # in ms
        sttc_dt_avg = int(50 * (fs / 1000) - 1)
        sttc_dt_concat = int(25 * (fs / 1000))
        trial_len = int(n_lags * bin_size * (fs / 1000))

        n_trials = 40  # this is fixed based on experimental datasets
        m_iterations = 100

        with open(data_folder + 'dataset\\trial_dict.pkl', 'rb') as f:
            trial_dict = pickle.load(f)

        with open(data_folder + 'dataset\\trial_binned_dict.pkl', 'rb') as f:
            trial_binned_dict = pickle.load(f)


        # output_log = data_folder + '\\dataset\\cut_30min\\trials_tau_log.txt'
        # old_stdout = sys.stdout
        # sys.stdout = open(output_log, 'w')

        items_to_process = 1000
        start_idx = 0
        stop_idx = len(trial_dict)

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

            with open(data_folder + 'results\\pearsonr_trial_avg_50ms_20lags_dict.pkl', "wb") as f:
                pickle.dump(pearsonr_trial_avg_dict, f)

        if calculate_trials_sttc_avg:
            print('Starting STTC trial avg ...')
            sttc_trial_avg_dict = {}
            for i, (k, v) in enumerate(islice(trial_dict.items(), items_to_process), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{items_to_process}, {datetime.now()}')

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
                                                                           zero_padding_len_=int(150 * (fs / 1000)),
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

            with open(data_folder + '\\results\\sttc_trial_avg_50ms_20lags_dict_test.pkl', "wb") as f:
                pickle.dump(sttc_trial_avg_dict, f)

        if calculate_trials_sttc_concat:
            print('Starting STTC trial concat ...')
            sttc_trial_concat_dict = {}
            for i, (k, v) in enumerate(islice(trial_dict.items(), start_idx, stop_idx), start=1):
                print(f'#####\nProcessing unit {k}, {i}/{(stop_idx - start_idx)}, {datetime.now()}')

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
                                                       zero_padding_len_=int(3000 * (fs / 1000)),
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

            with open(data_folder + '\\results\\sttc_trial_concat_50ms_20lags_dict_dt25_1000_end.pkl', "wb") as f:
                pickle.dump(sttc_trial_concat_dict, f)

        # sys.stdout = old_stdout








