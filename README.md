<img src="iSTTC_logo.png" align="left" width="320" alt="iSTTC logo" />

<h3> iSTTC: a robust method for accurate estimation of intrinsic neural timescales from single-unit recordings </h3>

Preprint on biorxiv ([here][isttc_biorxiv]).
Published in PLoS Computational Biology ([here][isttc_compbio]).

[isttc_biorxiv]:https://www.biorxiv.org/content/10.1101/2025.08.01.668071v1
[isttc_compbio]:https://doi.org/10.1371/journal.pcbi.1013385

The repository includes tools to compute iSTTC-based autocorrelation functions, fit timescales on continuous or epoched spiking data, and replicate analyses from the associated paper.
<br clear="left" />
### Installation

You can install the latest version of **isttc** directly from the GitHub repository:

```bash
pip install git+https://github.com/iinnpp/isttc.git
```
To install with optional plotting and notebook dependencies:

```bash
pip install "git+https://github.com/iinnpp/isttc.git#egg=isttc[plot,notebook]"
```

If you want to modify the code, clone the repository and install it in editable mode:

```bash
git clone https://github.com/iinnpp/isttc.git
cd isttc
pip install -e .
```

### How to run

Examples of how to use the iSTTC are in the Examples folder:

* generate_syn_data notebook contains code to generate test dataset (10 min long spike trains and sets of 40 x 1000 ms trails)
* estimate_tau notebook contains code to estimate intrinsic timescale using four methods: classic ACF and iSTTC on long spike trains and PearsonR and iSTTC on trials. Notebook generates plot of estimated intrinsic timescale and REE (relative estimation error).
* spike_trains.pkl, trials.pkl and trials_binned.pkl are sample datasets generated with generate_syn_data

The details of the method are explained in:
Pochinok, I., Hanganu-Opatz, I. L., & Chini, M. (2026). iSTTC: A robust method for accurate estimation of intrinsic neural timescales from single-unit recordings. PLoS Computational Biology, 22(3), e1013385. https://doi.org/10.1371/journal.pcbi.1013385

### Functions

#### `acf_sttc()`

Computes an **autocorrelation function (ACF)** for a spike train using the modified version of **Spike Time Tiling Coefficient (STTC)** ([Cutts and Eglen 2014][isttc]). This method quantifies temporal dependencies while remaining robust to firing rate variability and sparse spiking.  

[isttc]:https://pubmed.ncbi.nlm.nih.gov/25339742/

```python
acf_sttc(signal_, n_lags_, lag_shift_, sttc_dt_, signal_length_, verbose_=False)
```
[go to implementation][acf_sttc]

[acf_sttc]:https://github.com/iinnpp/isttc/blob/master/scripts/calculate_acf.py#L345

#### `acf_sttc_fast()`

Computes the same **autocorrelation function (ACF)** as `acf_sttc()`, but uses a faster STTC implementation with `np.searchsorted` for spike-window matching. It is intended as a drop-in replacement for long spike trains and returns values matching `acf_sttc()` within numerical precision.

```python
acf_sttc_fast(signal_, n_lags_, lag_shift_, sttc_dt_, signal_length_, verbose_=False)
```

Benchmarking on 1000 synthetic Hawkes spike trains spanning firing rates from 0.01 to 10 Hz, timescales from 50 to 300 ms, and excitation strengths from 0.1 to 0.9 showed:

* average speedup across all 1000 spike trains: 17.69x
* median speedup across all 1000 spike trains: 12.73x
* average speedup for spike trains with at least 100 spikes: 27.64x
* median speedup for spike trains with at least 100 spikes: 27.80x

To reproduce the benchmark:

```bash
python benchmarks/benchmark_acf_sttc_fast.py --num-settings 1000 --repeats 1
```

[go to implementation][acf_sttc_fast]

[acf_sttc_fast]:https://github.com/iinnpp/isttc/blob/master/src/isttc/scripts/calculate_acf.py

#### `acf_sttc_trial_concat()`

Computes the **autocorrelation function (ACF)** across multiple spike train trials using a modified **Spike Time Tiling Coefficient (STTC)**. Each trial is **zero-padded** before concatenation to prevent artificial correlations across trial boundaries. 

```python
acf_sttc_trial_concat(spike_train_l_, n_lags_, lag_shift_, sttc_dt_, trial_len_, zero_padding_len_, verbose_=False)
```

[go to implementation][acf_sttc_trial_concat]

[acf_sttc_trial_concat]:https://github.com/iinnpp/isttc/blob/master/scripts/calculate_acf.py#L398
