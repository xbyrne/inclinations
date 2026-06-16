# The true masses of radial-velocity exoplanets constrained by stability

## Introduction

RV observations of exoplanets only give minimum masses. However, in compact systems, they can't be *too* large, as this would make the system dynamically unstable.

This work makes use of stability likelihoods from [`SPOCK`](https://github.com/dtamayo/spock) to constrain the inclination -- and hence true masses -- of planets in compact RV systems. Our Bayesian framework has been named CINEMAS (Constraining INclinations of Exoplanets and their MAsses by Stability); a dedicated python package, `cinemas`, is available [here](https://github.com/xbyrne/cinemas), and on `pip`.

## Scripts

The python scripts here use the packages in `requirements.txt`.

The relevant data from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) are stored in `data/exoplanet_catalogue.csv`. These can be redownloaded using `cinemas.download_multiplanet_systems` if desired.
