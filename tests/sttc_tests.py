"""
Compares results from elephant sttc with my implementation of sttc:
* for sttc,
* for ACF calculated with sttc.
"""
import numpy as np
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient
import neo

from scripts.calculate_acf import sttc, acf_sttc


def calc_sttc_elephant(spike_train_1, spike_train_2, t_start_, t_stop_, dt_):
    spike_train_neo_1 = neo.SpikeTrain(spike_train_1, units='ms', t_start=t_start_, t_stop=t_stop_)
    spike_train_neo_2 = neo.SpikeTrain(spike_train_2, units='ms', t_start=t_start_, t_stop=t_stop_)
    sttc_no_shift = spike_time_tiling_coefficient(spike_train_neo_1, spike_train_neo_2, dt=dt_ * pq.ms)
    return sttc_no_shift


def acf_sttc_elephant(spike_train, n_lags_, acf_lag_ms_, sttc_lag_ms_, rec_length_, verbose=True):
    if acf_lag_ms_ * n_lags_ == rec_length_:
        shift_ms_l = np.linspace(acf_lag_ms_, acf_lag_ms_ * (n_lags_ - 1), n_lags_ - 1).astype(int)
    else:
        shift_ms_l = np.linspace(acf_lag_ms_, acf_lag_ms_ * n_lags_, n_lags_).astype(int)
    if verbose:
        print('shift_ms_l {}'.format(shift_ms_l))

    spike_train_bin = np.zeros(rec_length_)
    spike_train_bin[spike_train] = 1
    if verbose:
        print(spike_train_bin.shape)

    sttc_self_l = []
    # correlate with itself
    spike_train_neo = neo.SpikeTrain(spike_train, units='ms', t_start=0, t_stop=len(spike_train_bin))
    sttc_no_shift = spike_time_tiling_coefficient(spike_train_neo, spike_train_neo, dt=sttc_lag_ms_ * pq.ms)
    sttc_self_l.append(sttc_no_shift)

    # correlated shifted signal
    for shift_ms in shift_ms_l:

        spike_train_bin1 = spike_train_bin[shift_ms:]
        spike_train_bin2 = spike_train_bin[:- shift_ms]
        if verbose:
            print('spike_train_bin1 {}, spike_train_bin2 {}'.format(spike_train_bin1.shape, spike_train_bin2.shape))

        spike_train_bin1_idx = np.nonzero(spike_train_bin1)[0]
        spike_train_bin2_idx = np.nonzero(spike_train_bin2)[0]
        if verbose:
            print('spike_train_bin1_idx {}'.format(spike_train_bin1_idx))
            print('spike_train_bin2_idx {}'.format(spike_train_bin2_idx))

        spike_train_neo_1 = neo.SpikeTrain(spike_train_bin1_idx, units='ms', t_start=0, t_stop=len(spike_train_bin1))
        spike_train_neo_2 = neo.SpikeTrain(spike_train_bin2_idx, units='ms', t_start=0, t_stop=len(spike_train_bin2))
        if verbose:
            print(spike_train_neo_1)
            print(spike_train_neo_2)

        sttc_self = spike_time_tiling_coefficient(spike_train_neo_1, spike_train_neo_2, dt=sttc_lag_ms_ * pq.ms)
        sttc_self_l.append(sttc_self)

    return sttc_self_l


if __name__ == "__main__":
    # generate data
    spike_trains_l = []
    n_trains = 10
    train_len = 1000
    for i in range(n_trains):
        poisson = np.random.poisson(.05, train_len)
        bounded_poisson = np.clip(poisson, a_min=0, a_max=1)
        spike_trains_l.append(bounded_poisson)

    print('Running 1st test: my sttc vs elephant sttc:...')
    # calculate sttc
    for i in range(n_trains - 1):
        print('#############')
        print('i {}'.format(i))
        spike_times_1 = np.where(spike_trains_l[i] == 1)[0]
        spike_times_2 = np.where(spike_trains_l[i + 1] == 1)[0]

        my_sttc = sttc(spike_times_1, spike_times_2, t_start_=0, t_stop_=train_len, dt_=10, verbose_=False)
        print(my_sttc)

        elephant_sttc = calc_sttc_elephant(spike_times_1, spike_times_2, t_start_=0, t_stop_=train_len, dt_=10)
        print(elephant_sttc)

        print('my_sttc - elephant_sttc: {}'.format(my_sttc - elephant_sttc))
        assert np.isclose(np.asarray(my_sttc), np.asarray(elephant_sttc)).any(), 'STTC-my differs from STTC-elephant'

    print('Running 2nd test: my sttc-acf vs elephant sttc-acf:...')
    # calculate acf
    n_lags = 10
    for i in range(n_trains):
        print('#############')
        print('i {}'.format(i))
        spike_times = np.where(spike_trains_l[i] == 1)[0]

        my_acf_sttc = acf_sttc(spike_times, n_lags, lag_shift_=50, sttc_dt_=25, signal_length_=1000, verbose_=False)
        print(my_acf_sttc)

        elephant_acf_sttc = acf_sttc_elephant(spike_times, n_lags, acf_lag_ms_=50, sttc_lag_ms_=25,
                                              rec_length_=1000, verbose=False)
        print(elephant_acf_sttc)

        print('my_acf_sttc - elephant_acf_sttc: {}'.format(np.asarray(my_acf_sttc) - np.asarray(elephant_acf_sttc)))
        assert np.isclose(np.asarray(my_acf_sttc), np.asarray(elephant_acf_sttc)).any(), \
            'ACF-my differs from ACF-elephant'

