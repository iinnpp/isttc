import numpy as np
from itertools import islice
import pickle
from datetime import datetime
from scipy import stats

from isttc.scripts.cfg_global import project_folder_path

# add the path to the abcTau package
import sys
#sys.path.append('./abcTau')
sys.path.append('C:\\Users\\ipochino\\AppData\\Local\\anaconda3\\envs\\isttc\\Lib\\site-packages\\abcTau')
import abcTau


# ========== Parameters ==========
# data parameters
summStat_metric = 'comp_cc'
ifNorm = True # if normalize the autocorrelation or PSD
deltaT = 1 # temporal resolution of data.
binSize = 50 #  bin-size for binning data and computing the autocorrelation.
disp = None # put the dispersion parameter if computed with grid-search
maxTimeLag = 1000 # only used when using autocorrelation for summary statistics
#lm = round(maxTimeLag/binSize) # maximum bin for autocorrelation computation

# desired generative model from the list of 'generative_models.py' and the distance function from 'diatance_functions.py'
generativeModel = 'oneTauOU'
distFunc = 'linear_distance'

# Define a uniform prior distribution over the given range
# for a uniform prior: stats.uniform(loc=x_min,scale=x_max-x_min)
t_min = 0.0 # first timescale
t_max = 400.0
priorDist = [stats.uniform(loc= t_min, scale = t_max - t_min)]

# aABC fitting parameters
epsilon_0 = 1  # initial error threshold
min_samples = 100 # min samples from the posterior
steps = 60 # max number of iterations
minAccRate = 0.01 # minimum acceptance rate to stop the iterations
parallel = False # if parallel processing
n_procs = 1 # number of processor for parallel processing (set to 1 if there is no parallel processing)

# File paths
dataset_folder = project_folder_path + 'synthetic_dataset\\'
results_folder = project_folder_path + 'results\\synthetic\\results\\param_fr_alpha_tau\\'

results_folder_abctau = project_folder_path + 'results\\synthetic\\results\\param_fr_alpha_tau_abctau\\'
# path and filename to save the intermediate results after running each step
inter_save_direc = results_folder_abctau + 'interim_results\\'
datasave_path = results_folder_abctau + 'final_results\\'

# Execution flags

# creating model object
class MyModel(abcTau.Model):

    # This method initializes the model object.
    def __init__(self):
        pass

    # draw samples from the prior.
    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    # Choose the generative model (from generative_models)
    # Choose autocorrelation computation method (from basic_functions)
    def generate_data(self, theta):
        # generate synthetic data
        if disp == None:
            syn_data, numBinData = eval('abcTau.generative_models.' + generativeModel + \
                                        '(theta, deltaT, binSize, T, numTrials, data_mean, data_var)')
        else:
            syn_data, numBinData = eval('abcTau.generative_models.' + generativeModel + \
                                        '(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp)')

        # compute the summary statistics
        syn_sumStat = abcTau.summary_stats.comp_sumStat(syn_data, summStat_metric, ifNorm, deltaT, binSize, T,
                                                        numBinData, maxTimeLag)
        return syn_sumStat

    # Computes the summary statistics
    def summary_stats(self, data):
        sum_stat = data
        return sum_stat

    # Choose the method for computing distance (from basic_functions)
    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0:  # in case of all nans return large d to reject the sample
            d = 10 ** 4
        else:
            d = eval('abcTau.distance_functions.' + distFunc + '(data, synth_data)')
        return d

# ========== Main ==========
if __name__ == "__main__":

    with open(dataset_folder + 'trials40_binned.pkl', 'rb') as f:
        data_binned = pickle.load(f)

    trial_dict_binned = data_binned['trial_dict']
    alphas_binned = data_binned['alphas']
    fr_values_binned = data_binned['fr_values']
    taus_ms_binned = data_binned['tau_ms']
    n_trials_binned = data_binned['n_trials']
    trial_lens_binned = data_binned['trial_lens']

    print(f'n spike trains {len(trial_dict_binned)}, trial_lens {trial_lens_binned[0]} ms')

    for k, v in list(trial_dict_binned.items())[100:1000]:
        spike_binned = v[0]
        numTrials = n_trials_binned[k]
        T = trial_lens_binned[k]

        numBinData = spike_binned.shape[1]
        data_mean = np.mean(spike_binned)
        data_var = abcTau.preprocessing.comp_cc(spike_binned, spike_binned, 1, binSize, numBinData)[0]
        data_sumStat = abcTau.summary_stats.comp_sumStat(spike_binned, summStat_metric, ifNorm, deltaT, binSize, T,
                                                         numBinData, maxTimeLag)

        # data_sumStat, data_mean, data_var, T, numTrials = abcTau.preprocessing.extract_stats(binary_train, deltaT, binSize,\
        #                                                                                   summStat_metric, ifNorm, maxTimeLag)
        print(
            f'sumStat len {data_sumStat.shape}, data_mean {data_mean}, data_var {data_var}, T {T}, numTrials {numTrials}, numBinData {numBinData}')

        # Run the aABC algorithm and save the results
        filenameSave = 'spike_train_' + str(k)
        inter_filename = 'spike_train_interim_' + str(k)
        abc_results, final_step = abcTau.fit.fit_withABC(MyModel, data_sumStat, priorDist, inter_save_direc,
                                                         inter_filename, \
                                                         datasave_path, filenameSave, epsilon_0, min_samples, \
                                                         steps, minAccRate, parallel, n_procs, disp)


