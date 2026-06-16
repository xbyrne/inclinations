# The true masses of radial-velocity exoplanets constrained by stability


## Introduction

RV observations of exoplanets only give minimum masses. However, in compact systems, they can't be *too* large, as this would make the system dynamically unstable.

This work makes use of stability likelihoods from [`SPOCK`](https://github.com/dtamayo/spock) to constrain the inclination -- and hence true masses -- of planets in compact RV systems. Our Bayesian framework has been named CINEMAS (Constraining INclinations of Exoplanets and their MAsses by Stability); a dedicated python package, `cinemas`, is available [here](https://github.com/xbyrne/cinemas), and on `pip`.


## Scripts

The python scripts here use the packages in `requirements.txt`.

The relevant data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) are stored in `data/exoplanet_catalogue.csv`. These can be redownloaded using `cinemas.download_multiplanet_systems` if desired.


### Running Bayesian inference with `cinemas`

The `scripts/run_cinemas.py` script runs the `cinemas` Bayesian inference on six compact RV systems, whose inclination is *a priori* unconstrained.

The MCMC routine (powered by the [`emcee`](https://github.com/dfm/emcee) package) runs for 20 000 steps, using the default `cinemas` parameters.

The results are saved to `results/mcmc_results/<system>.npz`.
To avoid bloating this repo, the results folder are available on zenodo at ...[TODO!]


### Using `cinemas` to constrain unseen planets

The `scripts/search_for_hd2151512f.py` uses `cinemas` to constrain the properties of a hypothetical planet, HD 215152 $\~{\rm f}$, located in a dynamically tight region between planets d and e of that system.

The results of this scan are saved to `results/search_hd215152f/hd215152f.npz`. Again, these are available on zenodo at ...[TODO!]


## Analysis

The figures and tables are generated from the results file by `scripts/plot_figures.py` and `scripts/tabulate_tables.py`, which call scripts in the `scripts/figs` and `scripts/tables` folders. The figures rely on some tabulated data, so run `tabulate_tables.py` before `plot_figures.py`.

The figures and tables are saved to `tex/figs` and `tex/tables`.
