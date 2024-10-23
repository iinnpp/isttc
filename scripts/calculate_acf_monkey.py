"""
Test: calculating Pearson and iSTTC autocorrelation for Monkey data.
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

    for col in ['unit_id', 'trial_id', 'condition_id']:
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

    for col in ['unit_id', 'trial_id', 'condition_id']:
        acf_df[col] = acf_df[col].astype(int)
    for col in ['tau_ms', 'tau',  'r_squared'] + acf_columns_labels + \
               popt_columns_label:
        acf_df[col] = acf_df[col].astype(float)
    acf_df['acf_fit_failed'] = acf_df.apply(lambda row: True if np.isnan(row['tau_ms']) else False, axis=1)
    # line below is commented for a reason - astype(bool) makes everything True
    # acf_df['acf_decay_1_4'] = acf_df['acf_decay_1_4'].astype(bool)

    return acf_df


# copied from timescales project
def calculate_acf_pearson_t(sua_binned_l_, n_lags_, duration_bins_=None):
    """
    Calculate ACF based on binned spiking data (spike counts).

    :param sua_binned_l_: list
    list of binned spike data together with meta info (elements of the list are rows from csv file)
    every row: animal_id, age, unit_id, channel_id, non_zero_bins_ratio, fr, rpv, rec_length, bin1, ..., bin_n
    first bin: idx 8
    :param n_lags_: int, number of time lags to calculate ACF on
    :param duration_bins_: int, recording portion to use for calculation, [0, duration_ms_]
    :return: acf_dict: dict, dict with all animal/unit meta info and calculated ACF
    """
    acf_dict = {}

    for row_idx, row in enumerate(sua_binned_l_):
        binned_spike_train = row[3:]

        if duration_bins_ is not None:
            binned_spike_train = binned_spike_train[:duration_bins_+1]
            print('Calculating for duration {} bins, last bin is at {}'.format(duration_bins_, len(binned_spike_train)))

        acf_ = acf(binned_spike_train, nlags=n_lags_)
        acf_dict[row_idx] = {'unit_id': row[0],
                             'trial_id': row[1],
                             'condition_id': row[2],
                             'acf': acf_}
    return acf_dict


# todo can be shorter
def calculate_acf_sttc_t(sua_non_binned_l, n_lags_, resolution_ms_, fs_, duration_ms_=None):
    """
    Calculate ACF based on non binned spiking data using sttc.

    :param sua_non_binned_l: list
    list of spike trains (elements of the list are rows from csv file)
    every row: animal_id, age, unit_id, channel_id, bin1, ..., bin_n
    first bin: idx 4
    :param n_lags_: int, number of time lags to calculate ACF on
    :param resolution_ms_: int, shift step
    :param fs_: int, sampling frequency, Hz
    :param duration_ms_: int, recording portion to use for calculation, [0, duration_ms_]
    :return: acf_dict: dict, dict with all animal/unit meta info and calculated ACF
    """
    acf_dict = {}

    shift_ms_l = np.linspace(resolution_ms_+1, resolution_ms_ * n_lags_ + 1, n_lags_).astype(int)

    for row_idx, row in enumerate(sua_non_binned_l):
        print('Processing unit_id {},  row_idx {}'.format(row[0], row_idx))
        spike_train = np.asarray(row[3:]).astype(float)
        # spike_train_ms = spike_train / fs_ * 1000
        spike_train_ms = spike_train * 1000  # csv is in sec
        spike_train_ms_int = spike_train_ms.astype(int)

        if duration_ms_ is not None:
            spike_train_ms_int = spike_train_ms_int[spike_train_ms_int <= duration_ms_]
            spike_train_bin = np.zeros(duration_ms_ + 1)
            print('Calculating for duration {} ms, last spike is at {} ms'.
                  format(duration_ms_, spike_train_ms_int[-1] if len(spike_train_ms_int) > 0 else 'none'))
        else:
            spike_train_bin = np.zeros(spike_train_ms_int[-1] + 1)

        spike_train_bin[spike_train_ms_int] = 1

        sttc_self_l = []

        # correlate with itself
        spike_train_neo = neo.SpikeTrain(spike_train_ms_int, units='ms', t_start=0, t_stop=len(spike_train_bin))
        sttc_no_shift = spike_time_tiling_coefficient(spike_train_neo, spike_train_neo, dt=resolution_ms_ * pq.ms)
        sttc_self_l.append(sttc_no_shift)

        # correlated shifted signal
        for shift_ms in shift_ms_l:
            spike_train_bin1 = spike_train_bin[:-1 - shift_ms + 1]
            spike_train_bin2 = spike_train_bin[shift_ms:]

            spike_train_bin1_idx = np.nonzero(spike_train_bin1)[0]
            spike_train_bin2_idx = np.nonzero(spike_train_bin2)[0]

            spike_train_neo_1 = neo.SpikeTrain(spike_train_bin1_idx, units='ms', t_start=0, t_stop=len(spike_train_bin1))
            spike_train_neo_2 = neo.SpikeTrain(spike_train_bin2_idx, units='ms', t_start=0, t_stop=len(spike_train_bin2))

            sttc_self = spike_time_tiling_coefficient(spike_train_neo_1, spike_train_neo_2, dt=resolution_ms_ * pq.ms)
            sttc_self_l.append(sttc_self)

        acf_ = np.asarray(sttc_self_l)
        acf_dict[row_idx] = {'unit_id': row[0],
                             'trial_id': row[1],
                             'condition_id': row[2],
                             'acf': acf_}

    return acf_dict


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


def func_exp_t(x, a, b, c):
    """
    Exponential function to fit the data.
    :param x: 1d array, independent variable
    :param a: float, parameter to fit
    :param b: float, parameter to fit
    :param c: float, parameter to fit
    :return: callable
    """
    return a * np.exp(-b * x) + c


def fit_exp_t(acf_dict_in_out_, n_lags_, start_idx_=1):
    """
    Fit function func_exp to data using non-linear least square.

    important point: Fit is done from the first ACF value (acf[0] is skipped, it is done like this in the papers,
    still not sure)

    :param acf_dict_in_out_: dict, dict with calculated ACF functions
    :param n_lags_: int, number of time lags used to calculate ACFs
    :return: acf_dict_in_out_: dict, dict same as input with new fields: popt, pcov, r-squared
    """
    t = np.linspace(start_idx_, n_lags_, n_lags_).astype(int)

    for k, v in acf_dict_in_out_.items():
        print('Processing unit_id: {}, trial_id: {}'.format(v['unit_id'], v['trial_id']))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                popt, pcov = curve_fit(func_exp_t, t, v['acf'][start_idx_:], maxfev=5000)
                v['popt'] = popt
                v['pcov'] = pcov
                # fit r-squared
                y_pred = func_exp_t(t, *popt)
                r_squared = r2_score(v['acf'][start_idx_:], y_pred)
                v['r_squared'] = r_squared
            except RuntimeError as e:
                print('unit_id: {}, trial_id: {}: RuntimeError: {}'. format(v['unit_id'], v['trial_id'], e))
                v['popt'] = np.nan
                v['pcov'] = np.nan
                v['r_squared'] = np.nan
            except OptimizeWarning as o:
                print('unit_id: {}, trial_id: {}: OptimizeWarning: {}'. format(v['unit_id'], v['trial_id'], o))
                v['popt'] = np.nan
                v['pcov'] = np.nan
                v['r_squared'] = np.nan
            except RuntimeWarning as re:
                print('unit_id: {}, trial_id: {}: RuntimeWarning: {}'. format(v['unit_id'], v['trial_id'], re))
                v['popt'] = np.nan
                v['pcov'] = np.nan
                v['r_squared'] = np.nan
            except ValueError as ve:
                print('unit_id: {}, trial_id: {}: ValueError: {}'. format(v['unit_id'], v['trial_id'], ve))
                print('Possible reason: acf contains NaNs, low spike count')
                v['popt'] = np.nan
                v['pcov'] = np.nan
                v['r_squared'] = np.nan

    return acf_dict_in_out_


def calculate_tau_t(acf_dict_in_out_, bin_size_):
    """
    Calculates tau (time constant) of the ACF.

    :param acf_dict_in_out_: dict, dict with calculated ACF functions
    :param bin_size_: int, bin size in ms
    :return: dict, dict same as input with new fields: tau and tau_ms
    """
    for k, v in acf_dict_in_out_.items():
        if not np.any(np.isnan(v['popt'])):
            tau = 1 / v['popt'][1]
            tau_ms = tau * bin_size_
        else:
            tau = np.nan
            tau_ms = np.nan

        v['tau'] = tau
        v['tau_ms'] = tau_ms

    return acf_dict_in_out_


def calculate_area_sttc(sua_data_folder_, results_data_folder_, resolution_suffix_, n_lags_,
                        resolution_ms_, fs_, duration_ms_=None):
    # sua_list = load_csv(sua_data_folder_, area_file_prefix_)
    csv_data_file = sua_data_folder_ + 'data_pfdl_fixon_1500ms_fixation.csv'

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

    csv_data_file = sua_data_folder_ + 'data_pfdl_fixon_1500ms_fixation_binned_50ms.csv'
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

    # acf_df['rec_length_ms'] = acf_df['rec_length'] / (fs_ / 1000)  # todo move to another place
    if duration_ms_ is not None:
        output_filename = results_data_folder_ + 'binned_pearson\\' + 'tau_bsl_sua_pearson_' \
                          + bin_size_suffix_ + '_' + str(n_lags_) + 'lags_' \
                          + str(np.round(duration_ms_/1000/60).astype(int)) + 'min_df.pkl'
    else:
        output_filename = results_data_folder_ + 'binned_pearson\\' + 'tau_bsl_sua_pearson_' \
                          + bin_size_suffix_ + '_' + str(n_lags_) + 'lags_df.pkl'
    acf_df.to_pickle(output_filename)


if __name__ == "__main__":
    fs = 1000
    duration_ms = None

    # todo import from cfg_global is not working, fix later
    dataset_folder = 'Q:\\Personal\\Irina\\projects\\isttc\\results\\monkey\\'
    isttc_results_folder_path = 'Q:\\Personal\\Irina\\projects\\isttc\\results\\'

    monkey_results_folder = isttc_results_folder_path + 'monkey\\fixation_period_1500ms\\'

    params_dict = {'sttc_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'sttc',
                                  'calc': True},

                   'pear_50_20': {'bin_size': 50, 'n_lags': 20, 'bin_size_suffix': '50ms', 'metric': 'pearson',
                                  'calc': False}
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
                calculate_area_pearson(dataset_folder,
                                       monkey_results_folder,bin_size_suffix,
                                       n_lags,
                                       bin_size, fs, duration_ms)

            if calc_sttc:
                calculate_area_sttc(dataset_folder, monkey_results_folder, bin_size_suffix,
                                    n_lags,
                                    bin_size, fs, duration_ms)
