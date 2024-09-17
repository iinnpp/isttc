"""
Test: calculating Pearson and iSTTC autocorrelation for Allen mouse data.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit, OptimizeWarning
from sklearn.metrics import r2_score
import neo
import csv
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient

import warnings

#from scripts.cfg_global import isttc_results_folder_path, dataset_folder_path
#from scripts.cfg_global import isttc_results_folder_path




def calculate_acf_flags_t(acf_dict_in_out_, start_idx=1, end_idx=5):
    """
    Calculate ACF properties:
    1. Decline in [50ms, 200ms] period (see Cavanagh et al. 2016) for 50 ms bins; in general decline in [1, 5] bins
    period.
    2. ...

    Calculated properties are added to the input dict.

    :param acf_dict_in_out_: dict, dict with calculated ACF functions
    :param start_idx: first acf value index
    :param end_idx: last acf value index (not inclusive)
    :return: acf_dict_in_out_: dict, dict same as input with new fields
    """
    for k, v in acf_dict_in_out_.items():
        if np.all(np.diff(v['acf'][start_idx:end_idx]) <= 0):
            v['acf_decay_1_4'] = True
        else:
            v['acf_decay_1_4'] = False

    return acf_dict_in_out_


def dict_to_df_sttc_t(acf_dict_):
    """
    Transforms dict to data frame.

    :param acf_dict_:
    :return:
    """
    acf_df = pd.DataFrame.from_dict(acf_dict_, orient='index')

    acf_2d_a = np.vstack(acf_df['acf'].values)
    n_acf_lags = acf_2d_a.shape[1]
    acf_columns_labels = ['acf_' + str(i) for i in range(n_acf_lags)]

    popt_l = acf_df['popt'].values.tolist()
    popt_split_l = [[np.nan, np.nan, np.nan] if np.any(np.isnan(popt)) else [popt[0], popt[1], popt[2]] for popt in
                    popt_l]
    popt_2d_a = np.vstack(popt_split_l)
    popt_columns_label = ['popt_0', 'popt_1', 'popt_2']

    acf_df = pd.concat([acf_df,
                        pd.DataFrame(columns=acf_columns_labels, data=acf_2d_a),
                        pd.DataFrame(columns=popt_columns_label, data=popt_2d_a)], axis=1)

    for col in ['unit_id', 'session_id', 'animal_id']:
        acf_df[col] = acf_df[col].astype(int)
    for col in ['tau_ms', 'tau', 'r_squared'] + acf_columns_labels + popt_columns_label:
        acf_df[col] = acf_df[col].astype(float)

    acf_df['acf_fit_failed'] = acf_df.apply(lambda row: True if np.isnan(row['tau_ms']) else False, axis=1)
    # line below is commented for a reason - astype(bool) makes everything True
    # acf_df['acf_decay_1_4'] = acf_df['acf_decay_1_4'].astype(bool)

    return acf_df


def dict_to_df_pearson_t(acf_dict_):
    """
    Transforms dict to data frame.

    :param acf_dict_:
    :return:
    """
    acf_df = pd.DataFrame.from_dict(acf_dict_, orient='index')

    acf_2d_a = np.vstack(acf_df['acf'].values)
    n_acf_lags = acf_2d_a.shape[1]
    acf_columns_labels = ['acf_' + str(i) for i in range(n_acf_lags)]

    popt_l = acf_df['popt'].values.tolist()
    popt_split_l = [[np.nan, np.nan, np.nan] if np.any(np.isnan(popt)) else [popt[0], popt[1], popt[2]] for popt in
                    popt_l]
    popt_2d_a = np.vstack(popt_split_l)
    popt_columns_label = ['popt_0', 'popt_1', 'popt_2']

    acf_df = pd.concat([acf_df,
                        pd.DataFrame(columns=acf_columns_labels, data=acf_2d_a),
                        pd.DataFrame(columns=popt_columns_label, data=popt_2d_a)], axis=1)

    for col in ['unit_id', 'rec_length', 'session_id', 'animal_id']:
        acf_df[col] = acf_df[col].astype(int)
    for col in ['fr_hz', 'tau_ms', 'tau',  'r_squared'] + acf_columns_labels + \
               popt_columns_label:
        acf_df[col] = acf_df[col].astype(float)
    acf_df['acf_fit_failed'] = acf_df.apply(lambda row: True if np.isnan(row['tau_ms']) else False, axis=1)
    # line below is commented for a reason - astype(bool) makes everything True
    # acf_df['acf_decay_1_4'] = acf_df['acf_decay_1_4'].astype(bool)

    return acf_df


def calculate_area_sttc(sua_data_folder_, results_data_folder_, resolution_suffix_, n_lags_,
                        resolution_ms_, fs_, duration_ms_=None):
    # sua_list = load_csv(sua_data_folder_, area_file_prefix_)
    csv_data_file = sua_data_folder_ + 'allen_test_full_v2.csv'

    with open(csv_data_file, newline='') as f:
        reader = csv.reader(f)
        sua_list = list(reader)

    print('Loaded N units {}'.format(len(sua_list)))

    acf_dict = calculate_acf_sttc_t(sua_list, n_lags_, resolution_ms_, fs_, duration_ms_)
    acf_dict = calculate_acf_flags_t(acf_dict, start_idx=0, end_idx=4)
    acf_dict = fit_exp_t(acf_dict, n_lags_, start_idx_=1)
    acf_dict = calculate_tau_t(acf_dict, resolution_ms_)

    acf_df = dict_to_df_sttc_t(acf_dict)
    if duration_ms_ is not None:
        output_filename = results_data_folder_ + 'not_binned_sttc\\' + 'tau_bsl_sua_sttc_' \
                          + resolution_suffix_ + '_' + str(n_lags_) + 'lags_' \
                          + str(np.round(duration_ms_/1000/60).astype(int)) + 'min_df.pkl'
    else:
        output_filename = results_data_folder_ + 'not_binned_sttc\\' + 'tau_bsl_sua_sttc_' \
                          + resolution_suffix_ + '_' + str(n_lags_) + 'lags_df.pkl'
    acf_df.to_pickle(output_filename)


def calculate_area_pearson(sua_data_folder_, results_data_folder_, bin_size_suffix_, n_lags_,
                           bin_size_, fs_, duration_ms_=None):
    # sua_binned_list = load_csv_binned(sua_data_folder_ + 'binned_pearson\\binned_spikes\\', area_file_prefix_,
    #                                   bin_size_suffix_)

    csv_data_file = sua_data_folder_ + 'binned_pearson\\binned_spikes\\' + 'bsl_sua_binned_' + bin_size_suffix_ + '.csv'
    with open(csv_data_file, newline='') as f:
        reader = csv.reader(f)
        sua_binned_list = list(reader)
    print('Loaded N units {}'.format(len(sua_binned_list)))

    duration_bins = np.floor(duration_ms_ / bin_size_).astype(int) if duration_ms_ is not None else None
    acf_dict = calculate_acf_pearson_t(sua_binned_list, n_lags_, duration_bins)
    acf_dict = calculate_acf_flags_t(acf_dict, start_idx=0, end_idx=4)
    acf_dict = fit_exp_t(acf_dict, n_lags_, start_idx_=1)
    acf_dict = calculate_tau_t(acf_dict, bin_size_)

    acf_df = dict_to_df_pearson_t(acf_dict)

    acf_df['rec_length_ms'] = acf_df['rec_length'] / (fs_ / 1000)  # todo move to another place
    if duration_ms_ is not None:
        output_filename = results_data_folder_ + 'binned_pearson\\' + 'tau_bsl_sua_pearson_' \
                          + bin_size_suffix_ + '_' + str(n_lags_) + 'lags_' \
                          + str(np.round(duration_ms_/1000/60).astype(int)) + 'min_df.pkl'
    else:
        output_filename = results_data_folder_ + 'binned_pearson\\' + 'tau_bsl_sua_pearson_' \
                          + bin_size_suffix_ + '_' + str(n_lags_) + 'lags_df.pkl'
    acf_df.to_pickle(output_filename)


if __name__ == "__main__":
    fs_np = 30000
    duration_ms = None

    # todo import from cfg_global is not working, fix later
    dataset_folder = 'Q:\\Personal\\Irina\\projects\\isttc\\'
    isttc_results_folder_path = 'Q:\\Personal\\Irina\\projects\\isttc\\results\\'

    allen_suffix = 'allen_mice'
    allen_results_folder = isttc_results_folder_path + allen_suffix + '\\'

    # params_dict = {'sttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'sttc',
    #                               'calc': False},
    #
    #                'pear_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'pearson',
    #                               'calc': True}
    #                }

    params_dict = {'sttc_25_40': {'bin_size': 25, 'n_lags': 40, 'bin_size_suffix': '25ms', 'metric': 'sttc',
                                  'calc': True},
                   'sttc_40_25': {'bin_size': 40, 'n_lags': 25, 'bin_size_suffix': '40ms', 'metric': 'sttc',
                                  'calc': True},
                   'sttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'sttc',
                                  'calc': False},
                   'sttc_60_16': {'bin_size': 60, 'n_lags': 16, 'bin_size_suffix': '60ms', 'metric': 'sttc',
                                  'calc': True},
                   'sttc_75_13': {'bin_size': 75, 'n_lags': 13, 'bin_size_suffix': '75ms', 'metric': 'sttc',
                                  'calc': True},
                   'sttc_100_10': {'bin_size': 100, 'n_lags': 10, 'bin_size_suffix': '100ms', 'metric': 'sttc',
                                   'calc': True},
                   'pear_25_40': {'bin_size': 25, 'n_lags': 40, 'bin_size_suffix': '25ms', 'metric': 'pearson',
                                  'calc': True},
                   'pear_40_25': {'bin_size': 40, 'n_lags': 25, 'bin_size_suffix': '40ms', 'metric': 'pearson',
                                  'calc': True},
                   'pear_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'pearson',
                                  'calc': False},
                   'pear_60_16': {'bin_size': 60, 'n_lags': 16, 'bin_size_suffix': '60ms', 'metric': 'pearson',
                                  'calc': True},
                   'pear_75_13': {'bin_size': 75, 'n_lags': 13, 'bin_size_suffix': '75ms', 'metric': 'pearson',
                                  'calc': True},
                   'pear_100_10': {'bin_size': 100, 'n_lags': 10, 'bin_size_suffix': '100ms', 'metric': 'pearson',
                                   'calc': True}
                   }

    for k, v in params_dict.items():
        print('Run {}, params {}'.format(k, v))
        calc_sttc = False
        calc_pearson = False
        if v['calc']:

            if v['metric'] == 'sttc':
                calc_sttc = True
            else:
                calc_pearson = True

            bin_size = v['bin_size']
            bin_size_suffix = v['bin_size_suffix']
            n_lags = v['n_lags']

            if calc_pearson:
                calculate_area_pearson(allen_results_folder,
                                       allen_results_folder,bin_size_suffix,
                                       n_lags,
                                       bin_size, fs_np, duration_ms)

            if calc_sttc:
                calculate_area_sttc(dataset_folder, allen_results_folder, bin_size_suffix,
                                    n_lags,
                                    bin_size, fs_np, duration_ms)
