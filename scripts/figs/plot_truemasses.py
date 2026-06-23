"""
plot_truemasses.py
==================
Plots the distributions of true mass for the planets in the YZ Ceti system.
I might not use this in the end; it doesn't get the point across very well.
"""

import sys
from pathlib import Path

import cinemas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def true_mass_function(minimum_mass_function: callable) -> callable:
    i = np.radians(np.linspace(0, 90, 1000))

    def integrand(M, i):
        return minimum_mass_function(M * np.sin(i)) * np.sin(i) ** 2

    def prior(M):
        return np.array([np.trapezoid(integrand(Mi, i), i) for Mi in M])

    return prior


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_truemasses.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data
    results_dir = ROOT_DIR / "results" / "mcmc_results"
    system = "YZ Cet"
    fl = np.load(results_dir / f"{system.lower().replace(' ', '_')}.npz")
    samples = fl["samples"]

    burn_in = 2000
    inclination_samples = np.degrees(np.arccos(samples[burn_in:, :, 0].flatten()))
    mmin_samples = samples[burn_in:, :, 2:5].reshape(-1, 3)
    true_mass_samples = mmin_samples / np.sin(np.radians(inclination_samples[:, None]))

    exoplanet_catalogue = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    yz_cet_obs = cinemas.dataloading.load_system_observations(
        "YZ Cet", exoplanet_catalogue
    )

    # ==================
    # Calculating priors
    bins = np.arange(0.3, 3.71, 0.05)

    x = np.linspace(bins[0], bins[-1], 1000)
    true_mass_prior_functions = [
        true_mass_function(
            lambda M, i=i: gaussian(
                M, yz_cet_obs.minimum_masses[i].mean, yz_cet_obs.minimum_masses[i].error
            )
        )
        for i in range(3)
    ]
    true_mass_priors = [f(x) for f in true_mass_prior_functions]

    true_mass_95 = np.percentile(true_mass_samples, 95, axis=0)

    x_long = np.linspace(bins[0], 20, 10000)
    prior_long = [f(x_long) for f in true_mass_prior_functions]
    true_mass_prior_95 = [
        x_long[np.where(np.cumsum(prior) / np.sum(prior) >= 0.95)[0][0]].item()
        for prior in prior_long
    ]

    # ========
    # Plotting
    fg, axs = plt.subplots(
        3, 1, figsize=(6, 5), sharex=True, gridspec_kw={"hspace": 0.0}
    )
    for i, (p, ax, c) in enumerate(zip(["b", "c", "d"], axs, ["r", "g", "b"])):
        ax.plot(x, true_mass_priors[i], c=c, ls=":", lw=2)
        ax.axvline(true_mass_prior_95[i], color=c, ls=":", lw=2)

        ax.hist(true_mass_samples[:, i], bins=bins, color=c, density=True, alpha=0.5)
        ax.axvline(true_mass_95[i], color=c)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.tick_params(length=6, direction="in", top=True, labelsize=12, width=1.5)
        ax.set_title(p, y=0.7, x=0.03, horizontalalignment="left", fontsize=16)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Labelling and legend
    ax = axs[0]
    ax.annotate(
        "95% post.",
        xy=(true_mass_95[0] - 0.15, 1.7),
        fontsize=12,
        c="r",
        rotation=90,
        va="center",
    )
    ax.annotate(
        "95% prior",
        xy=(true_mass_prior_95[0] - 0.15, 1.7),
        fontsize=12,
        c="r",
        rotation=90,
        va="center",
    )

    handles = [
        plt.Line2D([0], [0], color="k", ls=":", lw=2),
        plt.Rectangle((0, 0), 1, 1, color="k", alpha=0.5, lw=0),
    ]
    labels = ["Prior", "Posterior"]
    ax.legend(handles, labels, loc="center right", fontsize=12)

    axs[-1].set_xlabel("True Mass [M$_{{\\oplus}}$]", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
