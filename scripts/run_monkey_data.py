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


if __name__ == "__main__":
    fs = 1000



