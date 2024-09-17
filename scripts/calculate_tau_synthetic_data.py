"""
Estimates time constants for synthetic dataset.
"""
from scripts.cfg_global import isttc_results_folder_path
from scripts.spike_train_acf_utils import bin_spike_train, calculate_acf_pearson, fit_single_exp, calculate_acf_isttc


def calculate_tau_isttc(spike_train_ms_int_l_, n_lags_, resolution_ms_):
    acf_not_binned = calculate_acf_isttc(spike_train_ms_int_l_, n_lags_, resolution_ms_)
    fit_popt, fit_pcov, tau, fit_r_squared = fit_single_exp(acf_not_binned, start_idx_=1)
    return fit_popt, fit_pcov, tau, fit_r_squared


def calculate_tau_pearson(spike_train_int_l_, bin_length_ms_, n_lags_, fs_):
    binned_spike_train = bin_spike_train(spike_train_int_l_, bin_length_ms_, fs_, verbose_=False)
    acf_binned = calculate_acf_pearson(binned_spike_train, n_lags_)
    fit_popt, fit_pcov, tau, fit_r_squared = fit_single_exp(acf_binned, start_idx_=1)
    return fit_popt, fit_pcov, tau, fit_r_squared


if __name__ == "__main__":
    # fs_np = 30000
    results_folder = isttc_results_folder_path + 'synthetic_data\\'

    params_dict = {'sttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'sttc',
                                  'calc': False},

                   'pear_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'pearson',
                                  'calc': True}
                   }

    # for k, v in params_dict.items():
    #     print('Run {}, params {}'.format(k, v))
    #     calc_sttc = False
    #     calc_pearson = False
    #     if v['calc']:
    #
    #         if v['metric'] == 'sttc':
    #             calc_sttc = True
    #         else:
    #             calc_pearson = True
    #
    #         bin_size = v['bin_size']
    #         bin_size_suffix = v['bin_size_suffix']
    #         n_lags = v['n_lags']
    #
    #         if calc_pearson:
    #             calculate_tau_pearson(allen_results_folder,
    #                                    allen_results_folder,bin_size_suffix,
    #                                    n_lags,
    #                                    bin_size, fs_np)
    #
    #         if calc_sttc:
    #             calculate_tau_isttc(dataset_folder, allen_results_folder, bin_size_suffix,
    #                                 n_lags,
    #                                 bin_size, fs_np)
