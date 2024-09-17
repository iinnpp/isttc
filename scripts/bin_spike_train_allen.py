"""
Script calculates and fits ACF based on spiking data.
"""

import pandas as pd
import numpy as np
import pickle
import csv
import os

# from scripts.utils.binned_spike_train_utils import bin_spike_train, get_non_zero_bin_ratio, calculate_bins_stats
# from scripts.utils.data_load_utils import load_csv, write_binned_csv
# from scripts.utils.spike_train_utils import get_firing_rate, get_rpv
# from scripts.cfg_global import timescales_results_folder_path, dataset_folder_path


# def get_firing_rate(spike_train_int_l_, fs_):
#     """
#     Calculate firing rate for a single unit.
#
#     :param spike_train_int_l_: list, list of spike times (int), sampling frequency fs_
#     :param fs_: int, sampling frequency in Hz
#     :return: float, firing rate, Hz
#     """
#     n_spikes = len(spike_train_int_l_)
#     last_spike_ts_s = spike_train_int_l_[-1] / fs_
#     fr = n_spikes / last_spike_ts_s
#
#     return fr


# def prepare_binned_dict(sua_l_, fs_, bin_length_, verbose_=False):
#     animal_id_l = []
#     session_id_l = []
#     unit_id_l = []
#     area_id_l = []
#     fr_l = []
#     # rpv_l = []
#     rec_length_l = []
#     binned_spikes_l = []
#     # non_zero_bins_ratio_l = []
#
#     for unit_ in sua_l_:
#         spikes_ = unit_[4:]
#         spikes = np.array(spikes_).astype(float)
#         spike_train_fs = spikes * fs_  # csv is in sec
#         spike_train_fs_int = list(spike_train_fs.astype(int))
#
#         unit_fr = get_firing_rate(spike_train_fs_int, fs_)
#         # unit_rpv = get_rpv(spikes, fs_)
#         # get rec length (as last spike time point)
#         rec_length = spike_train_fs_int[-1]
#         # get bins
#         bins_spike_counts = bin_spike_train(spike_train_fs_int, bin_length_, fs_, verbose_=verbose_)
#         # non_zero_bins_ratio = get_non_zero_bin_ratio(bins_spike_counts, verbose_=verbose_)
#
#         animal_id_l.append(unit_[0])
#         session_id_l.append(unit_[1])
#         unit_id_l.append(unit_[2])
#         area_id_l.append(unit_[3])
#         fr_l.append(unit_fr)
#         # rpv_l.append(unit_rpv)
#         rec_length_l.append(rec_length)
#         binned_spikes_l.append(bins_spike_counts)
#         # non_zero_bins_ratio_l.append(non_zero_bins_ratio)
#
#     # bins_stats = calculate_bins_stats(binned_spikes_l)
#
#     output_dict = {'animal_id_l': animal_id_l,
#                    'session_id_l': session_id_l,
#                    'unit_id_l': unit_id_l,
#                    'area_id_l': area_id_l,
#                    'fr_l': fr_l,
#                    # 'rpv_l': rpv_l,
#                    'rec_length_l': rec_length_l,
#                    'binned_spikes_l': binned_spikes_l,
#                    # 'non_zero_bins_ratio_l': non_zero_bins_ratio_l,
#                    # 'non_zero_seqs_list_l': bins_stats['non_zero_seqs_list_l'],
#                    # 'non_zero_seqs_len_list_l': bins_stats['non_zero_seqs_len_list_l'],
#                    # 'max_non_zero_seq_len_l': bins_stats['max_non_zero_seq_len_l'],
#                    # 'non_zero_seq_len_10_count_l': bins_stats['non_zero_seq_len_10_count_l'],
#                    # 'non_zero_seq_len_5_count_l': bins_stats['non_zero_seq_len_5_count_l']
#                    }
#
#     return output_dict


# def write_bins_stats_df(results_folder_, bin_size_suffix_, binned_data_dict_):
#     output_filename = results_folder_ + 'bsl_sua_binned_' + bin_size_suffix_ \
#                       + '_df.pkl'
#     bins_df = pd.DataFrame(np.vstack((binned_data_dict_['animal_id_l'],
#                                       binned_data_dict_['session_id_l'],
#                                       binned_data_dict_['unit_id_l'],
#                                       binned_data_dict_['area_id_l'],
#                                       # binned_data_dict_['non_zero_bins_ratio_l'],
#                                       binned_data_dict_['fr_l'],
#                                       # binned_data_dict_['rpv_l'],
#                                       binned_data_dict_['rec_length_l'],
#                                       # binned_data_dict_['max_non_zero_seq_len_l'],
#                                       # binned_data_dict_['non_zero_seq_len_5_count_l'],
#                                       # binned_data_dict_['non_zero_seq_len_10_count_l']
#                                       )).T,
#                            columns=['animal_id', 'session_id', 'unit_id', 'area_id', 'fr_hz',
#                                     'rec_length'])
#
#     for col in ['animal_id', 'session_id', 'unit_id', 'rec_length']:
#         bins_df[col] = bins_df[col].astype(int)
#
#     for col in ['fr_hz']:
#         bins_df[col] = bins_df[col].astype(float)
#
#     bins_df.to_pickle(output_filename)


def write_binned_csv(csv_data_folder_, bin_size_suffix_, binned_data_dict_, verbose_=False):
    """
    Write binned spike trains in csv file and stores it on the disk.
    Each row is one unit: animal_id, age, unit_id, channel_id, non_zero_bins_ratio, fr, rpv, rec_length, bin1, ..., bin_n

    :param csv_data_folder_: string, folder path
    :param area_file_prefix_: string, brain area label
    :param bin_size_suffix_: string, bin size label (ms)
    :param binned_data_dict_: dict, data to write
    :param verbose_: bool, default False, diagnostic printout if True, silent otherwise
    """
    output_filename = csv_data_folder_ + 'bsl_sua_binned_' + bin_size_suffix_ + '.csv'
    try:
        os.remove(output_filename)
        print('file {} removed'.format(output_filename))
    except FileNotFoundError:
        print('file {} did not exist, nothing to remove'.format(output_filename))

    animal_id_l = binned_data_dict_['animal_id_l']
    session_id_l = binned_data_dict_['session_id_l']
    unit_id_l = binned_data_dict_['unit_id_l']
    area_id_l = binned_data_dict_['area_id_l']
    # non_zero_bins_ratio_l = binned_data_dict_['non_zero_bins_ratio_l']
    fr_l = binned_data_dict_['fr_l']
    # rpv_l = binned_data_dict_['rpv_l']
    rec_length_l = binned_data_dict_['rec_length_l']
    binned_spikes_l = binned_data_dict_['binned_spikes_l']

    with open(output_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for unit_row_n, spike_train in enumerate(binned_spikes_l):
            if verbose_:
                print('Writing unit {}'.format(unit_row_n))
            spike_train_l = spike_train.tolist()
            row = [animal_id_l[unit_row_n]] + [session_id_l[unit_row_n]] + [unit_id_l[unit_row_n]] \
                + [area_id_l[unit_row_n]] + [fr_l[unit_row_n]] + [rec_length_l[unit_row_n]] + spike_train_l
            writer.writerow(row)


def process_one_area(csv_data_folder_, results_folder_, fs_, bin_length_, bin_size_suffix_):
    # sua_list = load_csv(csv_data_folder_, area_file_prefix_)

    csv_data_file = csv_data_folder_ + 'allen_test_full_v2.csv'
    with open(csv_data_file, newline='') as f:
        reader = csv.reader(f)
        sua_list = list(reader)
    print('Loaded N units {}'.format(len(sua_list)))

    binned_data_dict = prepare_binned_dict(sua_list, fs_, bin_length_)

    ##### SAVE #####
    # csv, similar to spiking data, each row is: animal_id, age, unit_id, channel_id, non_zero_bin_ratio, fr, rpv,
    # rec_length, bin1, ...
    write_binned_csv(results_folder_, bin_size_suffix_, binned_data_dict)

    # # dict with everything, also include non zero bin sequences in case I want to calculate acf only on those
    # with open(results_folder_ + area_file_prefix_ + '_bsl_sua_binned_' + bin_size_suffix_
    #           + '_dict.pkl', 'wb') as handle:
    #     pickle.dump(binned_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # df with bin stats
    write_bins_stats_df(results_folder_, bin_size_suffix_, binned_data_dict)


if __name__ == "__main__":
    # bin_size_suffix = '50ms'
    # bin_size = 50  # in ms, also resolution param for sttc

    fs_np = 30000

    # todo import from cfg_global is not working, fix later
    dataset_folder = 'Q:\\Personal\\Irina\\projects\\isttc\\'
    isttc_results_folder_path = 'Q:\\Personal\\Irina\\projects\\isttc\\results\\'

    allen_suffix = 'allen_mice'
    allen_results_folder = isttc_results_folder_path + allen_suffix + '\\'

    params_dict = {'25ms': {'bin_size': 25, 'bin_size_suffix': '25ms', 'calc': True},
                   '40ms': {'bin_size': 40, 'bin_size_suffix': '40ms', 'calc': True},
                   '50ms': {'bin_size': 50, 'bin_size_suffix': '50ms', 'calc': True},
                   '60ms': {'bin_size': 60, 'bin_size_suffix': '60ms', 'calc': True},
                   '75ms': {'bin_size': 75, 'bin_size_suffix': '75ms', 'calc': True},
                   '100ms': {'bin_size': 100, 'bin_size_suffix': '100ms', 'calc': True}
                   }

    for k, v in params_dict.items():
        print('Run {}, params {}'.format(k, v))
        if v['calc']:
            print('Calculating...')

            process_one_area(dataset_folder,
                             isttc_results_folder_path + allen_suffix + '\\binned_pearson\\binned_spikes\\',
                             fs_np, v['bin_size'], v['bin_size_suffix'])


