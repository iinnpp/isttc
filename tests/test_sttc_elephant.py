import numpy as np
import pytest

from isttc.scripts.calculate_acf import acf_sttc, sttc
from isttc.spike_utils import simulate_hawkes_thinning

neo = pytest.importorskip("neo")
pq = pytest.importorskip("quantities")
elephant_stc = pytest.importorskip("elephant.spike_train_correlation")


TRAIN_LEN_MS = 1_000
STTC_DT_MS = 10
N_LAGS = 10
ACF_LAG_MS = 50
ACF_STTC_DT_MS = 25


def _synthetic_spike_trains(n_trains=10, train_len_ms=TRAIN_LEN_MS, seed=123):
    return [
        np.unique(
            simulate_hawkes_thinning(
                fr_hz_=50.0,
                tau_ms_=100.0,
                alpha_=0.3,
                duration_ms_=train_len_ms,
                seed_=seed + idx,
            ).astype(int)
        ).astype(np.float64)
        for idx in range(n_trains)
    ]


def _elephant_sttc(spike_train_1, spike_train_2, t_start_ms, t_stop_ms, dt_ms):
    spike_train_neo_1 = neo.SpikeTrain(
        spike_train_1,
        units="ms",
        t_start=t_start_ms,
        t_stop=t_stop_ms,
    )
    spike_train_neo_2 = neo.SpikeTrain(
        spike_train_2,
        units="ms",
        t_start=t_start_ms,
        t_stop=t_stop_ms,
    )
    return elephant_stc.spike_time_tiling_coefficient(
        spike_train_neo_1,
        spike_train_neo_2,
        dt=dt_ms * pq.ms,
    )


def _elephant_acf_sttc(spikes, n_lags, acf_lag_ms, sttc_lag_ms, recording_length_ms):
    if acf_lag_ms * n_lags == recording_length_ms:
        shifts = np.linspace(acf_lag_ms, acf_lag_ms * (n_lags - 1), n_lags - 1).astype(int)
    else:
        shifts = np.linspace(acf_lag_ms, acf_lag_ms * n_lags, n_lags).astype(int)

    spike_train_bin = np.zeros(recording_length_ms)
    spike_train_bin[spikes.astype(int)] = 1

    acf_values = [
        _elephant_sttc(
            spikes,
            spikes,
            t_start_ms=0,
            t_stop_ms=recording_length_ms,
            dt_ms=sttc_lag_ms,
        )
    ]

    for shift_ms in shifts:
        spike_train_1 = np.flatnonzero(spike_train_bin[shift_ms:])
        spike_train_2 = np.flatnonzero(spike_train_bin[:-shift_ms])
        acf_values.append(
            _elephant_sttc(
                spike_train_1,
                spike_train_2,
                t_start_ms=0,
                t_stop_ms=recording_length_ms - shift_ms,
                dt_ms=sttc_lag_ms,
            )
        )

    return np.asarray(acf_values, dtype=np.float64)


@pytest.fixture(scope="module")
def synthetic_spike_trains():
    return _synthetic_spike_trains()


def test_sttc_matches_elephant_for_adjacent_synthetic_spike_trains(synthetic_spike_trains):
    for spike_train_1, spike_train_2 in zip(synthetic_spike_trains, synthetic_spike_trains[1:]):
        actual = sttc(
            spike_train_1,
            spike_train_2,
            t_start_=0,
            t_stop_=TRAIN_LEN_MS,
            dt_=STTC_DT_MS,
            verbose_=False,
        )
        expected = _elephant_sttc(
            spike_train_1,
            spike_train_2,
            t_start_ms=0,
            t_stop_ms=TRAIN_LEN_MS,
            dt_ms=STTC_DT_MS,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_acf_sttc_matches_elephant_for_synthetic_spike_trains(synthetic_spike_trains):
    for spikes in synthetic_spike_trains:
        actual = np.asarray(
            acf_sttc(
                spikes,
                N_LAGS,
                lag_shift_=ACF_LAG_MS,
                sttc_dt_=ACF_STTC_DT_MS,
                signal_length_=TRAIN_LEN_MS,
                verbose_=False,
            ),
            dtype=np.float64,
        )
        expected = _elephant_acf_sttc(
            spikes,
            n_lags=N_LAGS,
            acf_lag_ms=ACF_LAG_MS,
            sttc_lag_ms=ACF_STTC_DT_MS,
            recording_length_ms=TRAIN_LEN_MS,
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
