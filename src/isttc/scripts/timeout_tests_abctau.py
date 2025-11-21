# timeout_abctau_runner.py
import os
import sys
import time
import pickle
import numpy as np
import multiprocessing as mp
from scipy import stats

from isttc.scripts.cfg_global import project_folder_path

# Add the path to the abcTau package (re-applied in child via spawn)
sys.path.append(r"C:\Users\ipochino\AppData\Local\anaconda3\envs\isttc\Lib\site-packages\abcTau")
import abcTau

# ========== Parameters ==========
# data parameters
summStat_metric = "comp_cc"
ifNorm = True
deltaT = 1
binSize = 50
disp = None  # global default; per-trial dispersion computed below
maxTimeLag = 1000

# generative model & distance
generativeModel = "oneTauOU_gammaSpikes"
distFunc = "linear_distance"

# uniform prior on tau
t_min, t_max = 0.0, 400.0
priorDist = [stats.uniform(loc=t_min, scale=t_max - t_min)]

# fitting parameters
epsilon_0 = 1
min_samples = 50
steps = 60
minAccRate = 0.01
# force nested parallel OFF inside the worker for reliability
# parallel = False; n_procs = 1 inside the worker

# File paths
dataset_folder = project_folder_path + "synthetic_dataset\\"
results_folder_abctau = r"D:\all_abctau_dst_gamma_0_01_50_comp_time\\"
inter_save_direc = results_folder_abctau + "interim_results\\"
datasave_path = results_folder_abctau + "final_results\\"

# Model
class MyModel(abcTau.Model):
    def __init__(self):
        pass

    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    def generate_data(self, theta):
        # uses globals: deltaT, binSize, T, numTrials, data_mean, data_var, disp_local
        if disp is None:
            syn_data, numBinData = eval(
                f"abcTau.generative_models.{generativeModel}(theta, deltaT, binSize, T, numTrials, data_mean, data_var)"
            )
        else:
            syn_data, numBinData = eval(
                f"abcTau.generative_models.{generativeModel}(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp)"
            )
        syn_sumStat = abcTau.summary_stats.comp_sumStat(syn_data, summStat_metric, ifNorm, deltaT, binSize, T, numBinData, maxTimeLag)
        return syn_sumStat

    def summary_stats(self, data):
        return data

    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0:
            return 10**4
        return eval(f"abcTau.distance_functions.{distFunc}(data, synth_data)")

# Minimal timeout helper
def _wrap_fn(q, fn, arg):
    try:
        q.put(("ok", fn(arg)))
    except Exception as e:
        q.put(("err", repr(e)))

def run_with_timeout(fn, arg, timeout_s=600):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_wrap_fn, args=(q, fn, arg))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"### TIMED OUT after {timeout_s} sec.")
        return None
    status, payload = q.get()
    if status == "ok":
        return payload
    else:
        raise RuntimeError(payload)

# Per-unit worker
def fit_one_unit(args):
    """
    Args is a dict of primitives + numpy arrays only (picklable).
    Returns a small dict with 'k' and 'final_step'.
    """
    # Keep native libs from over-threading inside child
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Unpack
    k = args["k"]
    spike_binned = args["spike_binned"]
    numTrials = args["numTrials"]
    T = args["T"]
    binSize = args["binSize"]
    deltaT = args["deltaT"]
    maxTimeLag = args["maxTimeLag"]
    summStat_metric = args["summStat_metric"]
    ifNorm = args["ifNorm"]
    inter_save_direc = args["inter_save_direc"]
    datasave_path = args["datasave_path"]
    epsilon_0 = args["epsilon_0"]
    min_samples = args["min_samples"]
    steps = args["steps"]
    minAccRate = args["minAccRate"]

    # Compute summary stats & dispersion
    numBinData = spike_binned.shape[1]
    data_mean = float(np.mean(spike_binned))
    data_var = float(abcTau.preprocessing.comp_cc(spike_binned, spike_binned, 1, binSize, numBinData)[0])

    if data_mean == 0.0 or data_var == 0.0:
        return {"k": k, "skipped": True, "reason": "no spikes or zero variance"}

    disp = data_var / data_mean
    data_sumStat = abcTau.summary_stats.comp_sumStat(spike_binned, summStat_metric, ifNorm, deltaT, binSize, T, numBinData, maxTimeLag)

    print(
        f"unit {k}: sumStat {data_sumStat.shape}, mean {data_mean:.5g}, var {data_var:.5g}, disp {disp:.5g}, "
        f"T {T}, nTrials {numTrials}, nBins {numBinData}"
    )

    # Prepare filenames
    filenameSave = f"spike_train_{k}"
    inter_filename = f"spike_train_interim_{k}"

    # Make globals visible to MyModel.generate_data (abcTau expects these names)
    globals()["T"] = T
    globals()["numTrials"] = numTrials
    globals()["data_mean"] = data_mean
    globals()["data_var"] = data_var
    globals()["disp"] = disp  # local per-trial dispersion

    # Run ABC (disable nested parallelism)

    start = time.perf_counter()
    abc_results, final_step = abcTau.fit.fit_withABC(
        MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,
        datasave_path, filenameSave, epsilon_0, min_samples,
        steps, minAccRate, False, 1,  # parallel=False, n_procs=1
        disp
    )
    elapsed_time = time.perf_counter() - start
    return {"k": k, "final_step": int(final_step), "skipped": False, "elapsed_time": elapsed_time}

# ========== Main ==========
if __name__ == "__main__":
    mp.freeze_support()  # required on Windows when spawning

    with open(dataset_folder + "trials40_binned.pkl", "rb") as f:
        data_binned = pickle.load(f)

    trial_dict_binned = data_binned["trial_dict"]
    n_trials_binned = data_binned["n_trials"]
    trial_lens_binned = data_binned["trial_lens"]

    print(f"n spike trains {len(trial_dict_binned)}, trial_lens {trial_lens_binned[0]} ms")

    # timeout (seconds)
    TIMEOUT_S = 15 * 60

    # Iterate
    for k, v in list(trial_dict_binned.items())[:1000]:
        spike_binned = v[0]
        numTrials = n_trials_binned[k]
        T = trial_lens_binned[k]

        args = {
            "k": k,
            "spike_binned": spike_binned,
            "numTrials": numTrials,
            "T": T,
            "binSize": binSize,
            "deltaT": deltaT,
            "maxTimeLag": maxTimeLag,
            "summStat_metric": summStat_metric,
            "ifNorm": ifNorm,
            "inter_save_direc": inter_save_direc,
            "datasave_path": datasave_path,
            "epsilon_0": epsilon_0,
            "min_samples": min_samples,
            "steps": steps,
            "minAccRate": minAccRate,
        }

        try:
            res = run_with_timeout(fit_one_unit, args, timeout_s=TIMEOUT_S)
        except RuntimeError as e:
            print(f"### unit {k} failed: {e}")
            continue

        if res is None:
            print(f"### unit {k} timed out after {TIMEOUT_S//60} min — skipping.")
            continue

        if res.get("skipped"):
            print(f"### unit {k} skipped: {res.get('reason')}")
            continue

        print(f"### unit {k} finished (final_step={res['final_step']}, elapsed time = {res['elapsed_time']})")

        unit_time_dict = {'unit_id': k,
                        'elapsed_time': res['elapsed_time']}
        with open(results_folder_abctau + f'comp_time\\elapsed_time_spike_train_{k}.pkl', "wb") as f:
            pickle.dump(unit_time_dict, f)

