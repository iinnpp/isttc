# iSTTC

This repository relates to **iSTTC: a robust method for accurate estimation of intrinsic neural timescales from single-unit recordings** 

Preprint on biorxiv ([here][isttc_biorxiv]).

[isttc_biorxiv]:https://www.biorxiv.org/content/10.1101/2025.08.01.668071v1

It includes tools to compute iSTTC-based autocorrelation functions, fit timescales on continuous or epoched spiking data, and replicate analyses from the associated paper.

#### How-to run

Examples of how-to use the iSTTC are in the Examples folder:

* generate_syn_data notebook contains code to generate test dataset (10 min long spike trains and sets of 40 x 1000 ms trails)
* estimate_tau notebook contains code to estimate intrinsic timescale using four methods: classic ACF and iSTTC on long spike trains and PearsonR and iSTTC on trials. Notebook generates plot of estimated intrinsic timescale and REE (relative estimation error).
* spike_trains.pkl, trials.pkl and trials_binned.pkl are sample datasets generated with generate_syn_data


