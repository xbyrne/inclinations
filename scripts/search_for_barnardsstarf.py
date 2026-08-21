"""
search_for_barnardsstarf.py
===========================
Using CINEMAS to get posterior distributions on an outer planet in the Barnard's Star
system.
"""

from pathlib import Path

import cinemas
import numpy as np
import pandas as pd
from cinemas import observation_classes as obs

ROOT_DIR = Path(__file__).resolve().parents[1]


def create_hypothetical_planet_obs() -> obs.SystemObservations:
    """
    Create observations for a hypothetical planet beyond Barnard's Star e, which
    can be roughly Hill-stable within a certain parameter space.
    """

    minimum_mass_obs = obs.Observation(
        distribution="uniform",
        bounds=(50, 500),  # Earth masses
    )
    period_obs = obs.Observation(
        distribution="uniform",
        bounds=(1_000, 10_000),  # days
    )
    eccentricities = obs.Observation(
        distribution="uniform",
        bounds=(0.0, 0.3),
    )  # Seems a decent guess!

    hypothetical_planet_obs = obs.PlanetObservations(
        name="Barnard's star f",
        minimum_mass=minimum_mass_obs,
        period=period_obs,
        eccentricity=eccentricities,
    )

    return hypothetical_planet_obs


def main():

    EXOPLANET_CATALOGUE = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    SYSTEM = "Barnard's star"

    NSTEPS = 20000

    barnard_obs = cinemas.load_system_observations(SYSTEM, EXOPLANET_CATALOGUE)

    hypothetical_planet_obs = create_hypothetical_planet_obs()

    barnard_obs.add_planet(hypothetical_planet_obs)

    np.random.seed(42)

    print("=" * 50)
    print("In CINEMAS now: Barnard's Star with hypothetical outer planet f...")
    print("=" * 50)

    samples, log_probs, tau, acf = cinemas.run_mcmc_sampling(
        barnard_obs, nsteps=NSTEPS, nwalkers=100
    )

    RESULTS_DIR = ROOT_DIR / "results" / "search_barnardsstarf"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULTS_DIR / "barnardsstarf.npz",
        samples=samples,
        log_probs=log_probs,
        tau=tau,
        acceptance_fraction=acf,
    )


if __name__ == "__main__":
    main()
