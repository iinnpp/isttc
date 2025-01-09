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


def get_spikes_per_interval(sua_list_, interval_, sampling_freq_, include_empty_trials_=False, verbose_=False):
    """
        Extracts spike trains within a specified time interval, calculates spike counts and firing rates,
        and organizes the data into a dictionary.

        Parameters:
            sua_list_ (list): List of single-unit activity data, each entry containing unit ID, trial ID,
                            condition ID, and spike times.
            interval_ (tuple): Time interval (start, end) in which spikes are counted.
            sampling_freq_ (float): Sampling frequency in Hz.
            include_empty_trials_ (bool): Whether to include trials with no spikes in the interval.
            verbose_ (bool): Whether to print detailed information about empty trials.

    Returns:
        dict: A dictionary containing the following keys:
            - 'unit_ids': List of unit IDs.
            - 'trial_ids': List of trial IDs.
            - 'condition_ids': List of condition IDs.
            - 'spike_counts': List of spike counts per trial.
            - 'firing_rates': List of firing rates (Hz) per trial.
            - 'spike_trains': List of spike trains (spikes within the interval) per trial.
        """
    unit_id_l, trial_id_l, condition_id_l, spike_count_l, fr_hz_l, spike_trains_l = [], [], [], [], [], []
    interval_duration = (interval_[1] - interval_[0]) / sampling_freq_
    for trial in sua_list_:
        spike_train = trial[3:]
        spike_train_interval = [spike for spike in spike_train if interval_[0] <= int(spike) <= interval_[1]]
        if spike_train_interval or include_empty_trials_:
            unit_id_l.append(trial[0])
            trial_id_l.append(trial[1])
            condition_id_l.append(trial[2])
            spike_count_l.append(len(spike_train_interval))
            fr_hz_l.append(len(spike_train_interval) / interval_duration)
            spike_trains_l.append(spike_train_interval if spike_train_interval else [])
        if verbose_ and not spike_train_interval:
            print(f"Unit {trial[0]}, trial {trial[1]}: NO spikes in this interval")

    return {
        'unit_ids': unit_id_l,
        'trial_ids': trial_id_l,
        'condition_ids': condition_id_l,
        'spike_counts': spike_count_l,
        'firing_rates': fr_hz_l,
        'spike_trains': spike_trains_l
    }


def write_csv(output_filename, unit_id_l, trial_id_l, condition_id_l, spike_count_l, firing_rate_l,
              spike_trains_l, convert_to_list=False,
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
            if spike_count_l:
                row = [unit_id_l[unit_row_n]] + [trial_id_l[unit_row_n]] + [condition_id_l[unit_row_n]] \
                      + [spike_count_l[unit_row_n]] + [firing_rate_l[unit_row_n]] + spikes_l
            else:
                row = [unit_id_l[unit_row_n]] + [trial_id_l[unit_row_n]] + [condition_id_l[unit_row_n]] + spikes_l
            writer.writerow(row)


if __name__ == "__main__":
    data_folder = project_folder_path + 'monkey_dataset\\Irina_data\\'
    results_folder = project_folder_path + 'results\\monkey\\'

    load_data = False

    preprocess_data = False
    cut_interval = (0, 1500)  # in ms (should be in spikes time)

    bin_data = True
    bin_size_ms = 50
    signal_len_fs = 1500  # in signal sampling frequency
    fs = 1000
    input_file_suffix = '1500ms_with_empty_fixation'
    output_file_suffix = '1500ms_with_empty_fixation_binned_50ms'

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
        write_csv(output_filename_pfdl, unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, [], [],
                  spike_trains_pfdl_l,
                  convert_to_list=True, verbose=True)

        output_filename_pfp = results_folder + 'data_pfp_fixon.csv'
        write_csv(output_filename_pfp, unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, [], [], spike_trains_pfp_l,
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
        pfdl_no_empty = get_spikes_per_interval(sua_list_pfdl, cut_interval, fs, include_empty_trials_=False)
        pfp_no_empty = get_spikes_per_interval(sua_list_pfp, cut_interval, fs, include_empty_trials_=False)

        pfdl_with_empty = get_spikes_per_interval(sua_list_pfdl, cut_interval, fs, include_empty_trials_=True)
        pfp_with_empty = get_spikes_per_interval(sua_list_pfp, cut_interval, fs, include_empty_trials_=True)

        # write in csv files
        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + str(cut_interval[1]) + 'ms_no_empty_fixation.csv'
        write_csv(output_filename_pfdl, pfdl_no_empty['unit_ids'],
                  pfdl_no_empty['trial_ids'], pfdl_no_empty['condition_ids'],
                  pfdl_no_empty['spike_counts'], pfdl_no_empty['firing_rates'],
                  pfdl_no_empty['spike_trains'], convert_to_list=False, verbose=False)

        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + str(cut_interval[1]) + 'ms_with_empty_fixation.csv'
        write_csv(output_filename_pfdl, pfdl_with_empty['unit_ids'],
                  pfdl_with_empty['trial_ids'], pfdl_with_empty['condition_ids'],
                  pfdl_with_empty['spike_counts'], pfdl_with_empty['firing_rates'],
                  pfdl_with_empty['spike_trains'], convert_to_list=False, verbose=False)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + str(cut_interval[1]) + 'ms_no_empty_fixation.csv'
        write_csv(output_filename_pfp, pfp_no_empty['unit_ids'],
                  pfp_no_empty['trial_ids'], pfp_no_empty['condition_ids'],
                  pfp_no_empty['spike_counts'], pfp_no_empty['firing_rates'],
                  pfp_no_empty['spike_trains'], convert_to_list=False, verbose=False)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + str(cut_interval[1]) + 'ms_with_empty_fixation.csv'
        write_csv(output_filename_pfp, pfp_with_empty['unit_ids'],
                  pfp_with_empty['trial_ids'], pfp_with_empty['condition_ids'],
                  pfp_with_empty['spike_counts'], pfp_with_empty['firing_rates'],
                  pfp_with_empty['spike_trains'], convert_to_list=False, verbose=False)

        # save number of trials in df
        pfdl_n_trials_per_unit_with_empty = pd.DataFrame({
            'unit_id': pfdl_with_empty['unit_ids'],
            'area': ['pfdl'] * len(pfdl_with_empty['unit_ids']),
            'trial_id': pfdl_with_empty['trial_ids']
        }).groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfdl_n_trials_per_unit_with_empty.rename(columns={'trial_id': 'n_trials_with_empty'}, inplace=True)

        pfdl_n_trials_per_unit_no_empty = pd.DataFrame({
            'unit_id': pfdl_no_empty['unit_ids'],
            'area': ['pfdl'] * len(pfdl_no_empty['unit_ids']),
            'trial_id': pfdl_no_empty['trial_ids']
        }).groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfdl_n_trials_per_unit_no_empty.rename(columns={'trial_id': 'n_trials_no_empty'}, inplace=True)

        pfdl_n_trials_per_unit_merged = pfdl_n_trials_per_unit_with_empty.merge(pfdl_n_trials_per_unit_no_empty,
                                                                                on=['unit_id', 'area'], how='left')

        pfp_n_trials_per_unit_with_empty = pd.DataFrame({
            'unit_id': pfp_with_empty['unit_ids'],
            'area': ['pfdl'] * len(pfp_with_empty['unit_ids']),
            'trial_id': pfp_with_empty['trial_ids']
        }).groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfp_n_trials_per_unit_with_empty.rename(columns={'trial_id': 'n_trials_with_empty'}, inplace=True)

        pfp_n_trials_per_unit_no_empty = pd.DataFrame({
            'unit_id': pfp_no_empty['unit_ids'],
            'area': ['pfdl'] * len(pfp_no_empty['unit_ids']),
            'trial_id': pfp_no_empty['trial_ids']
        }).groupby(by=['unit_id', 'area'], as_index=False)['trial_id'].count()
        pfp_n_trials_per_unit_no_empty.rename(columns={'trial_id': 'n_trials_no_empty'}, inplace=True)

        pfp_n_trials_per_unit_merged = pfp_n_trials_per_unit_with_empty.merge(pfp_n_trials_per_unit_no_empty,
                                                                              on=['unit_id', 'area'], how='left')

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
        unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_count_pfdl_l, firing_rate_pfdl_l, spike_binned_pfdl_l = [], [], [], [], [], []
        for unit in sua_list_pfdl:
            unit_id_pfdl_l.append(unit[0])
            trial_id_pfdl_l.append(unit[1])
            condition_id_pfdl_l.append(unit[2])
            spike_count_pfdl_l.append(unit[3])
            firing_rate_pfdl_l.append(unit[4])
            spike_train = list(map(int, unit[5:]))
            binned_spike_train = bin_spike_train_fixed_len(spike_train, bin_size_ms, fs, signal_len_fs,
                                                           verbose_=False)
            spike_binned_pfdl_l.append(binned_spike_train)

        unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_count_pfp_l, firing_rate_pfp_l, spike_binned_pfp_l = [], [], [], [], [], []
        for unit in sua_list_pfp:
            unit_id_pfp_l.append(unit[0])
            trial_id_pfp_l.append(unit[1])
            condition_id_pfp_l.append(unit[2])
            spike_count_pfp_l.append(unit[3])
            firing_rate_pfp_l.append(unit[4])
            spike_train = list(map(int, unit[5:]))
            binned_spike_train = bin_spike_train_fixed_len(spike_train, bin_size_ms, fs, signal_len_fs,
                                                           verbose_=False)
            spike_binned_pfp_l.append(binned_spike_train)

        # save in csv
        output_filename_pfdl = results_folder + 'data_pfdl_fixon_' + output_file_suffix + '.csv'
        write_csv(output_filename_pfdl, unit_id_pfdl_l, trial_id_pfdl_l, condition_id_pfdl_l, spike_count_pfdl_l,
                  firing_rate_pfdl_l, spike_binned_pfdl_l,
                  convert_to_list=True, verbose=True)

        output_filename_pfp = results_folder + 'data_pfp_fixon_' + output_file_suffix + '.csv'
        write_csv(output_filename_pfp, unit_id_pfp_l, trial_id_pfp_l, condition_id_pfp_l, spike_count_pfp_l,
                  firing_rate_pfp_l, spike_binned_pfp_l,
                  convert_to_list=True, verbose=True)
