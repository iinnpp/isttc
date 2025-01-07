"""
Script loads monkey spiking data.
"""

import pickle
import csv
import numpy as np
import pandas as pd
import os
from scripts.cfg_global import project_folder_path
from scripts.spike_train_utils import bin_spike_train_fixed_len


def prep_data_for_csv(units_l, n_units):
    unit_id_l, trial_id_l, condition_id_l, spike_trains_l = [], [], [], []

    for n_unit in range(n_units):
        unit = units_l[n_unit]
        trial_id = 0

        # loop over conditions
        n_trials_all_conditions = 0
        for n_condition in range(len(unit)):
            condition = unit[n_condition]
            n_trials_in_condition = len(condition)
            n_trials_all_conditions = n_trials_all_conditions + n_trials_in_condition

            # loop over trials within a condition
            for idx, trial in enumerate(condition):
                spike_times = trial[0]

                if type(spike_times) != np.ndarray:
                    spike_times = np.expand_dims(spike_times, axis=0)

                spike_trains_l.append(spike_times)
                unit_id_l.append(n_unit)
                trial_id_l.append(trial_id)
                condition_id_l.append(n_condition)

                trial_id = trial_id + 1

    return unit_id_l, trial_id_l, condition_id_l, spike_trains_l


def get_n_trials_per_neuron(unit_all_trial, n_neuron):
    n_trials_all_conditions = 0
    for i in range(len(unit_all_trial[n_neuron])):
        # print('N trials for task conditions {}: {}'.format(i+1, len(unit_all_trial[n_neuron][i])))
        n_trials_all_conditions = n_trials_all_conditions + len(unit_all_trial[n_neuron][i])
    return n_trials_all_conditions


def get_spikes_per_interval(sua_list, area_name, interval=None, include_empty_trials=False, verbose=False):
    unit_id_l, trial_id_l, condition_id_l, spike_trains_l = [], [], [], []
    for trial in sua_list:
        if interval is None:
            unit_id_l.append(trial[0])
            trial_id_l.append(trial[1])
            condition_id_l.append(trial[2])
            spike_trains_l.append(trial[3:])
        else:
            spike_train = trial[3:]
            spike_train_interval = [spike for spike in spike_train if
                                    int(spike) >= interval[0] and int(spike) <= interval[1]]
            if len(spike_train_interval) >= 1:
                unit_id_l.append(trial[0])
                trial_id_l.append(trial[1])
                condition_id_l.append(trial[2])
                spike_trains_l.append(spike_train_interval)
            else:
                if include_empty_trials:
                    unit_id_l.append(trial[0])
                    trial_id_l.append(trial[1])
                    condition_id_l.append(trial[2])
                    spike_trains_l.append([])
                if verbose:
                    print('Unit {}, trial {}: NO spikes in this interval'.format(trial[0], trial[1]))

    summary_df = pd.DataFrame(np.vstack((unit_id_l, trial_id_l, condition_id_l)).T,
                              columns=['unit_id', 'trial_id', 'condition_id'])
    summary_df['area'] = area_name
    return summary_df, spike_trains_l


def write_csv(output_filename, unit_id_l, trial_id_l, condition_id_l, spike_trains_l, convert_to_list=False,
              verbose=True):
    """
    Write spike train data to a CSV file.

    Parameters:
    - output_filename: The name of the output CSV file. If file exists then it is deleted.
    - unit_id_l: List of unit IDs.
    - trial_id_l: List of trial IDs.
    - condition_id_l: List of condition IDs.
    - spike_trains_l: List of spike trains.
    - verbose: Whether to print progress information.
    - convert_to_list: Whether to convert each spike_train to a list.
    """
    # Check if the file exists and delete it if it does
    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for unit_row_n, spike_train in enumerate(spike_trains_l):
            if verbose:
                print(f'Writing unit {unit_id_l[unit_row_n]}')
            spikes_l = spike_train.tolist() if convert_to_list else spike_train
            row = [unit_id_l[unit_row_n]] + [trial_id_l[unit_row_n]] + [condition_id_l[unit_row_n]] + spikes_l
            writer.writerow(row)


if __name__ == "__main__":
    data_folder = project_folder_path + 'monkey_dataset\\Irina_data\\'
    results_folder = project_folder_path + 'results\\monkey\\'

    load_data = False

    preprocess_data = False
    cut_interval = [0, 1000]

    bin_data = True
    bin_size_ms = 50
    signal_len_fs = 1000  # in signal sampling frequency
    fs = 1000
    input_file_suffix = '1000ms_with_empty_fixation'
    output_file_suffix = '1000ms_with_empty_fixation_binned_50ms'

    if load_data:
        print('Loading data ...')
        # get data
        with open(data_folder + 'data_PFdl_fixON.pkl', 'rb') as f:
            data_PFdl_fix_on = pickle.load(f)
        n_units_pfdl = len(data_PFdl_fix_on)
        print('N neurons in PFdl fixON {}'.format(n_units_pfdl))

        with open(data_folder + 'data_PFp_fixON.pkl', 'rb') as f:
            data_PFp_fix_on = pickle.load(f)
        n_units_pfp = len(data_PFp_fix_on)
        print('N neurons in PFP fixON {}'.format(n_units_pfp))

        # prepare for csv
        unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_trains_pfdl_l = prep_data_for_csv(data_PFdl_fix_on,
                                                                                                      n_units_pfdl)
        unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_trains_pfp_l = prep_data_for_csv(data_PFp_fix_on,
                                                                                                  n_units_pfp)
        # write in csv file
        output_filename_pfdl = results_folder + 'data_pfdl_fixon.csv'
        write_csv(output_filename_pfdl, unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_trains_pfdl_l,
                  convert_to_list=True, verbose=True)

        output_filename_pfp = results_folder + 'data_pfp_fixon.csv'
        write_csv(output_filename_pfp, unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_trains_pfp_l,
                  convert_to_list=True, verbose=True)

        # save number of trials in df
        n_trials_per_neuron_pfdl_fix_on_l = []
        for i in range(len(data_PFdl_fix_on)):
            n_trails = get_n_trials_per_neuron(data_PFdl_fix_on, i)
            n_trials_per_neuron_pfdl_fix_on_l.append(n_trails)
        pfdl_summary_df = pd.DataFrame(n_trials_per_neuron_pfdl_fix_on_l, columns=['n_trials'])
        pfdl_summary_df.reset_index(inplace=True, drop=False)
        pfdl_summary_df.rename(columns={'index': 'unit_id'}, inplace=True)

        n_trials_per_neuron_pfp_fix_on_l = []
        for i in range(len(data_PFp_fix_on)):
            n_trails = get_n_trials_per_neuron(data_PFp_fix_on, i)
            n_trials_per_neuron_pfp_fix_on_l.append(n_trails)
        pfp_summary_df = pd.DataFrame(n_trials_per_neuron_pfp_fix_on_l, columns=['n_trials'])
        pfp_summary_df.reset_index(inplace=True, drop=False)
        pfp_summary_df.rename(columns={'index': 'unit_id'}, inplace=True)

        pfdl_summary_df.to_pickle(results_folder + 'pfdl_n_trials_per_unit_df.pkl')
        pfp_summary_df.to_pickle(results_folder + 'pfp_n_trials_per_unit_df.pkl')

    if preprocess_data:
        print(f'Preprocessing data, interval to cut {cut_interval} ms ...')
        # get csv
        csv_data_file_pfdl = results_folder + 'data_pfdl_fixon.csv'
        with open(csv_data_file_pfdl, newline='') as f:
            reader = csv.reader(f)
            sua_list_pfdl = list(reader)

        n_spike_trains_pfdl = len(sua_list_pfdl)
        print('N spike_trains in PFdl fixON {}'.format(n_spike_trains_pfdl))

        csv_data_file_pfp = results_folder + 'data_pfp_fixon.csv'
        with open(csv_data_file_pfp, newline='') as f:
            reader = csv.reader(f)
            sua_list_pfp = list(reader)

        n_spike_trains_pfp = len(sua_list_pfp)
        print('N spike_trains in PFp fixON {}'.format(n_spike_trains_pfp))

        # cut the data
        pfdl_no_empty_df, pfdl_no_empty_spike_trains_l = get_spikes_per_interval(sua_list_pfdl, 'pfdl',
                                                                                 interval=cut_interval,
                                                                                 include_empty_trials=False)
        pfp_no_empty_df, pfp_no_empty_spike_trains_l = get_spikes_per_interval(sua_list_pfp, 'pfp',
                                                                               interval=cut_interval,
                                                                               include_empty_trials=False)

        pfdl_with_empty_df, pfdl_with_empty_spike_trains_l = get_spikes_per_interval(sua_list_pfdl, 'pfdl',
                                                                                     interval=cut_interval,
                                                                                     include_empty_trials=True)
        pfp_with_empty_df, pfp_with_empty_spike_trains_l = get_spikes_per_interval(sua_list_pfp, 'pfp',
                                                                                   interval=cut_interval,
                                                                                   include_empty_trials=True)

        # write in csv files
        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + str(cut_interval[1]) + 'ms_no_empty_fixation.csv'
        write_csv(output_filename_pfdl, pfdl_no_empty_df['unit_id'].values,
                  pfdl_no_empty_df['trial_id'].values, pfdl_no_empty_df['condition_id'].values,
                  pfdl_no_empty_spike_trains_l, convert_to_list=False, verbose=False)

        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + str(cut_interval[1]) + 'ms_with_empty_fixation.csv'
        write_csv(output_filename_pfdl, pfdl_with_empty_df['unit_id'].values,
                  pfdl_with_empty_df['trial_id'].values, pfdl_with_empty_df['condition_id'].values,
                  pfdl_with_empty_spike_trains_l, convert_to_list=False, verbose=False)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + str(cut_interval[1]) + 'ms_no_empty_fixation.csv'
        write_csv(output_filename_pfp, pfp_no_empty_df['unit_id'].values,
                  pfp_no_empty_df['trial_id'].values, pfp_no_empty_df['condition_id'].values,
                  pfp_no_empty_spike_trains_l, convert_to_list=False, verbose=False)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + str(cut_interval[1]) + 'ms_with_empty_fixation.csv'
        write_csv(output_filename_pfp, pfp_with_empty_df['unit_id'].values,
                  pfp_with_empty_df['trial_id'].values, pfp_with_empty_df['condition_id'].values,
                  pfp_with_empty_spike_trains_l, convert_to_list=False, verbose=False)

        # save number of trials in df
        pfdl_n_trials_per_unit = pfdl_with_empty_df.groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfdl_n_trials_per_unit.rename(columns={'trial_id': 'n_trials_with_empty'}, inplace=True)

        pfdl_n_trials_per_unit_fix = pfdl_no_empty_df.groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfdl_n_trials_per_unit_fix.rename(columns={'trial_id': 'n_trials_no_empty'}, inplace=True)

        pfdl_n_trials_per_unit_merged = pfdl_n_trials_per_unit.merge(pfdl_n_trials_per_unit_fix, on=['unit_id', 'area'],
                                                                     how='left')

        pfp_n_trials_per_unit = pfp_with_empty_df.groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfp_n_trials_per_unit.rename(columns={'trial_id': 'n_trials_with_empty'}, inplace=True)

        pfp_n_trials_per_unit_fix = pfp_no_empty_df.groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfp_n_trials_per_unit_fix.rename(columns={'trial_id': 'n_trials_no_empty'}, inplace=True)

        pfp_n_trials_per_unit_merged = pfp_n_trials_per_unit.merge(pfp_n_trials_per_unit_fix, on=['unit_id', 'area'],
                                                                   how='left')

        pfdl_n_trials_per_unit_merged.to_pickle(results_folder + 'pfdl_n_trials_per_unit_fixation_'
                                                + str(cut_interval[1]) + '.pkl')
        pfp_n_trials_per_unit_merged.to_pickle(results_folder + 'pfp_n_trials_per_unit_fixation_'
                                               + str(cut_interval[1]) + '.pkl')

    if bin_data:
        print(f'Binning spikes, bin size {bin_size_ms} ms ...')

        # load data
        csv_data_file_pfdl = results_folder + 'data_pfdl_fixon_' + input_file_suffix + '.csv'
        with open(csv_data_file_pfdl, newline='') as f:
            reader = csv.reader(f)
            sua_list_pfdl = list(reader)
        n_spike_trains_pfdl = len(sua_list_pfdl)
        print('N spike_trains in PFdl fixON {}'.format(n_spike_trains_pfdl))

        csv_data_file_pfp = results_folder + 'data_pfp_fixon_' + input_file_suffix + '.csv'
        with open(csv_data_file_pfp, newline='') as f:
            reader = csv.reader(f)
            sua_list_pfp = list(reader)
        n_spike_trains_pfp = len(sua_list_pfp)
        print('N spike_trains in PFp fixON {}'.format(n_spike_trains_pfp))

        # bin
        unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_binned_pfdl_l = [], [], [], []
        for unit in sua_list_pfdl:
            unit_id_pfdl_l.append(unit[0])
            trial_id_pfdl_l.append(unit[1])
            condition_id_pfdl_l.append(unit[2])
            spike_train = list(map(int, unit[3:]))
            binned_spike_train = bin_spike_train_fixed_len(spike_train, bin_size_ms, fs, signal_len_fs,
                                                           verbose_=False)
            spike_binned_pfdl_l.append(binned_spike_train)

        unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_binned_pfp_l = [], [], [], []
        for unit in sua_list_pfp:
            unit_id_pfp_l.append(unit[0])
            trial_id_pfp_l.append(unit[1])
            condition_id_pfp_l.append(unit[2])
            spike_train = list(map(int, unit[3:]))
            binned_spike_train = bin_spike_train_fixed_len(spike_train, bin_size_ms, fs, signal_len_fs,
                                                           verbose_=False)
            spike_binned_pfp_l.append(binned_spike_train)

        # save in csv
        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + output_file_suffix + '.csv'
        write_csv(output_filename_pfdl, unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_binned_pfdl_l,
                  convert_to_list=True, verbose=True)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + output_file_suffix + '.csv'
        write_csv(output_filename_pfp, unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_binned_pfp_l,
                  convert_to_list=True, verbose=True)