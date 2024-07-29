"""
Functions to calculate autocorrelation function.
"""

import numpy as np
from statsmodels.tsa.stattools import acf
import neo
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient


def calculate_acf_pearson(sua_binned_, n_lags_):
    """
    Calculate ACF using Pearson autocorrelation.

    :param sua_binned_:
    :param n_lags_:
    :return:
    """
    pass


def calculate_acf_isttc(sua_, n_lags_):
    """
    Calculate ACF using iSTTC autocorrelation.

    :param sua_:
    :param n_lags_:
    :return:
    """
    pass


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
        binned_spike_train = row[8:]

        if duration_bins_ is not None:
            binned_spike_train = binned_spike_train[:duration_bins_+1]
            print('Calculating for duration {} bins, last bin is at {}'.format(duration_bins_, len(binned_spike_train)))

        acf_ = acf(binned_spike_train, nlags=n_lags_)
        acf_dict[row_idx] = {'animal_id': row[0],
                             'age': row[1],
                             'unit_id': row[2],
                             'channel_id': row[3],
                             'non_zero_bin_ratio': row[4],
                             'fr_hz': row[5],
                             'rpv': row[6],
                             'rec_length': row[7],
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
        print('Processing animal_id {}, unit_id {},  row_idx {}'.format(row[0], row[2], row_idx))
        spike_train = np.asarray(row[4:]).astype(int)
        spike_train_ms = spike_train / fs_ * 1000
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
        acf_dict[row_idx] = {'animal_id': row[0],
                             'age': row[1],
                             'unit_id': row[2],
                             'channel_id': row[3],
                             'acf': acf_}

    return acf_dict
