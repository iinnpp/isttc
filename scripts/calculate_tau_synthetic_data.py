"""
Estimates time constants for synthetic dataset.
"""
import numpy as np
import pandas as pd
from scripts.cfg_global import isttc_results_folder_path
from scripts.spike_train_acf_utils import bin_spike_train, calculate_acf_pearson, fit_single_exp, calculate_acf_isttc

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calculate_tau_isttc(spike_train_ms_int_l_, resolution_ms_, n_lags_):
    acf_not_binned = calculate_acf_isttc(spike_train_ms_int_l_, n_lags_, resolution_ms_)
    fit_popt, fit_pcov, tau, fit_r_squared = fit_single_exp(acf_not_binned, start_idx_=0)
    return fit_popt, fit_pcov, tau, fit_r_squared


def calculate_tau_pearson(spike_train_int_l_, bin_length_ms_, n_lags_, fs_):
    binned_spike_train = bin_spike_train(spike_train_int_l_, bin_length_ms_, fs_, verbose_=False)
    acf_binned = calculate_acf_pearson(binned_spike_train, n_lags_)
    fit_popt, fit_pcov, tau, fit_r_squared = fit_single_exp(acf_binned, start_idx_=0)
    return fit_popt, fit_pcov, tau, fit_r_squared


if __name__ == "__main__":
    # fs_np = 30000
    results_folder = isttc_results_folder_path + 'synthetic_data\\'
    dataset_folder = isttc_results_folder_path + 'synthetic_data\\dataset\\'

    params_dict = {'sttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'sttc',
                                  'calc': False},

                   'pear_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'pearson',
                                  'calc': True}
                   }
    data = np.load(dataset_folder + 'spike_train_tau200ms_500trials_5000ms.npy')
    data_int = data.astype(int)

    #data_int = data_int[:, :2000]

    bin_length_ms = 50
    n_lags = 20
    fs = 1000

    # calculate tau pearson
    tau_pears_l = []
    for i in range(data_int.shape[0]):
        tau_pears_l.append(calculate_tau_pearson(np.where(data_int[i,:] == 1)[0], bin_length_ms, n_lags, fs))

    tau_pears_tau = [tau[2] for tau in tau_pears_l]
    tau_pears_r_squared = [tau[3] for tau in tau_pears_l]
    tau_pears_tau_ms = [tau*bin_length_ms for tau in tau_pears_tau]

    tau_pears_df = pd.DataFrame(np.vstack((tau_pears_tau, tau_pears_tau_ms, tau_pears_r_squared)).T,
                                columns=['tau', 'tau_ms', 'r_squared'])

    # calculated acf pearson
    acf_pearson_l = []
    for i in range(data_int.shape[0]):
        binned_spike_train = bin_spike_train(np.where(data_int[i,:] == 1)[0], bin_length_ms, fs)
        acf_binned = calculate_acf_pearson(binned_spike_train, n_lags)
        acf_pearson_l.append(acf_binned)

    # save pearson results
    tau_pears_df.to_pickle(results_folder + 'tau_pears_df.pkl')
    np.save(results_folder + 'acf_pearson_l.npy', acf_pearson_l)

    # calculate tau isttc
    tau_isttc_l = []
    for i in range(data_int.shape[0]):
        tau_isttc_l.append(calculate_tau_isttc(np.where(data_int[i,:] == 1)[0], bin_length_ms, n_lags))

    tau_isttc_tau = [tau[2] for tau in tau_isttc_l]
    tau_isttc_r_squared = [tau[3] for tau in tau_isttc_l]
    tau_isttc_tau_ms = [tau*bin_length_ms for tau in tau_isttc_tau]

    tau_isttc_df = pd.DataFrame(np.vstack((tau_isttc_tau, tau_isttc_tau_ms, tau_isttc_r_squared)).T,
                                columns=['tau', 'tau_ms', 'r_squared'])

    # calculated acf isttc
    acf_isttc_l = []
    for i in range(data_int.shape[0]):
        acf_not_binned = calculate_acf_isttc(np.where(data_int[i,:] == 1)[0], n_lags, bin_length_ms)
        acf_isttc_l.append(acf_not_binned)

    # save isttc results
    tau_isttc_df.to_pickle(results_folder + 'tau_isttc_df.pkl')
    np.save(results_folder + 'acf_isttc_l.npy', acf_isttc_l)


