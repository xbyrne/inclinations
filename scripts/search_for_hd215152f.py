"""
search_for_hd215152f.py
========================
Using CINEMAS to get posterior distributions on an undetected planet in the HD 215152
system.
"""

from pathlib import Path

import cinemas
import numpy as np
import pandas as pd
from cinemas import observation_classes as obs

ROOT_DIR = Path(__file__).resolve().parents[2]


def create_hypothetical_planet_obs() -> obs.SystemObservations:
    """
    Create observations for a hypothetical planet in between HD 215152 d and e, which
    can be roughly Hill-stable within a certain parameter space.
    """

    minimum_mass_obs = obs.Observation(
        distribution="uniform",
        bounds=(0.05, 5.0),  # Earth masses
    )  # 0.05 ~ Mercury; NB anything above 2.5 would have a K larger than planet c
    period_obs = obs.Observation(
        distribution="uniform",
        bounds=(12, 24),  # days
    )  # Outside this range would be <10 R_H from d or e (at minimum masses)
    eccentricities = obs.Observation(
        distribution="uniform",
        bounds=(0.0, 0.3),
    )  # Seems a decent guess!

    hypothetical_planet_obs = obs.PlanetObservations(
        name="HD 215152 f",
        minimum_mass=minimum_mass_obs,
        period=period_obs,
        eccentricity=eccentricities,
    )

    return hypothetical_planet_obs


def main():

    EXOPLANET_CATALOGUE = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    SYSTEM = "HD 215152"

    NSTEPS = 20000

    hd_215152_obs = cinemas.load_system_observations(SYSTEM, EXOPLANET_CATALOGUE)

    hypothetical_planet_obs = create_hypothetical_planet_obs()

    hd_215152_obs.add_planet(hypothetical_planet_obs)

    np.random.seed(42)

    print("=" * 50)
    print("In CINEMAS now: HD 215152 with hypothetical planet f...")
    print("=" * 50)

    samples, log_probs, tau, acf = cinemas.run_mcmc_sampling(
        hd_215152_obs, nsteps=NSTEPS, nwalkers=100
    )  # Use 4*n_params for nwalkers ----------^^^ as this one struggled a bit

    RESULTS_DIR = ROOT_DIR / "results" / "search_hd215152f"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULTS_DIR / "hd215152f.npz",
        samples=samples,
        log_probs=log_probs,
        tau=tau,
        acceptance_fraction=acf,
    )


if __name__ == "__main__":
    main()
