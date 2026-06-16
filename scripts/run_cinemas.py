"""
run_cinemas.py
==============
Running CINEMAS on the six compact multi-planet RV systems.
"""

from pathlib import Path

import cinemas
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_DIR = ROOT_DIR / "data"
EXOPLANET_CATALOGUE_PATH = DATA_DIR / "exoplanet_catalogue.csv"

RESULTS_DIR = ROOT_DIR / "results" / "mcmc_results"


def main() -> None:

    exoplanet_catalogue = pd.read_csv(EXOPLANET_CATALOGUE_PATH)

    systems = [
        "HD 184010",
        "YZ Cet",
        "Barnard's star",
        "HD 158259",
        "HD 215152",
        "HD 28471",
    ]

    for system in systems:
        results_path = RESULTS_DIR / f"{system.lower().replace(' ', '_')}.npz"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        NSTEPS = 20000

        system_obs = cinemas.load_system_observations(system, exoplanet_catalogue)

        n_planets = len(exoplanet_catalogue[exoplanet_catalogue["hostname"] == system])
        print("=" * 50)
        print(f"In CINEMAS now: {system} ({n_planets} planets)...")
        print("=" * 50)

        np.random.seed(42)

        print(f"Running MCMC sampling for {system} with {NSTEPS} steps...")
        samples, log_probs, tau, acceptance_fraction = cinemas.run_mcmc_sampling(
            system_obs, nsteps=NSTEPS
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            results_path,
            samples=samples,
            log_probs=log_probs,
            tau=tau,
            acceptance_fraction=acceptance_fraction,
        )
        print(f"Results saved to {results_path.resolve()} .")
        print(f"Autocorrelation time: {tau}")
        print(f"Acceptance fraction: {acceptance_fraction}")


if __name__ == "__main__":
    main()
