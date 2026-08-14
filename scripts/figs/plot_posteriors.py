"""
plot_posteriors.py
==================
Plots the posterior distributions for certain parameters of the YZ Ceti system.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cinemas import dataloading as dl

ROOT_DIR = Path(__file__).resolve().parents[2]


def gaussian_from_obs(observation, x, x_min=0, x_max=np.inf):
    mean = observation.mean
    std = observation.error
    gaussian = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    raw_values = np.where((x >= x_min) & (x <= x_max), gaussian, 0)
    normalisation = np.trapezoid(raw_values, x)
    return raw_values / normalisation


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_posteriors.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data
    results_dir = ROOT_DIR / "results" / "mcmc_results"
    system = "YZ Cet"
    fl = np.load(results_dir / f"{system.lower().replace(' ', '_')}.npz")
    samples = fl["samples"]

    catalogue = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    yz_cet_obs = dl.load_system_observations(system, catalogue)
    burn_in = 2000

    # ========
    # Plotting
    fg, axs = plt.subplots(
        3, 3, figsize=(7, 4), gridspec_kw={"wspace": 0.05, "hspace": 0.0}
    )
    xlabelsize = 11

    # Inclination panel
    ax = axs[0, 0]
    cos_i_samples = samples[burn_in:, :, 0].flatten()
    inclination_samples = np.degrees(np.arccos(cos_i_samples))
    ax.hist(inclination_samples, bins=30, color="k", density=True, alpha=0.5)
    x = np.linspace(0, 90, 100)
    ax.plot(x, np.sin(np.radians(x)) * np.pi / 180, c="k", lw=1.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xticks([10, 20, 40, 50, 70, 80], minor=True)
    ax.set_xlabel("$i$ [deg]", fontsize=xlabelsize)
    ax.set_yticks([])

    # Stellar mass panel
    ax = axs[2, 0]
    ax.hist(
        samples[burn_in:, :, 1].flatten(),
        bins=30,
        color="k",
        density=True,
        alpha=0.5,
        label="Posterior",
    )
    x = np.linspace(0.08, 0.19, 100)
    ax.plot(x, gaussian_from_obs(yz_cet_obs.star_mass, x), c="k", lw=1.5, label="Prior")
    ax.set_xticks([0.1, 0.15])
    ax.set_xticks([0.125, 0.175], minor=True)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("$M_*$ [M$_\\odot$]", fontsize=xlabelsize)

    # Planet mass panels
    x = np.linspace(0.3, 1.7, 100)
    colours = ["r", "g", "b"]
    planets = ["b", "c", "d"]
    for i in range(3):
        ax = axs[i, 1]
        ax.hist(
            samples[burn_in:, :, 2 + i].flatten(),
            bins=30,
            color=colours[i],
            density=True,
            alpha=0.5,
        )
        prior = gaussian_from_obs(yz_cet_obs.planet_observations[i].minimum_mass, x)
        ax.plot(x, prior, c="k", lw=1.5)
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.set_xticks([0.25, 0.75, 1.25, 1.75], minor=True)
        ax.set_xlim(x[0], x[-1])
        if i < 2:
            ax.set_xticklabels([])
        ax.set_title(
            planets[i], y=0.71, x=0.05, horizontalalignment="left", fontsize=13
        )
    ax.set_xlabel("$M_{\\rm min}$ [M$_{\\oplus}$]", fontsize=xlabelsize)

    # Eccentricity panels
    x = np.linspace(0.0, 0.27, 100)
    for i in range(3):
        ax = axs[i, 2]
        ax.hist(
            samples[burn_in:, :, 8 + i].flatten(),
            bins=30,
            color=colours[i],
            density=True,
            alpha=0.5,
        )
        prior = gaussian_from_obs(yz_cet_obs.planet_observations[i].eccentricity, x)
        ax.plot(x, prior, c="k", lw=1.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks([0.0, 0.1, 0.2])
        ax.set_xticks([0.05, 0.15, 0.25], minor=True)
        if i < 2:
            ax.set_xticklabels([])
    ax.set_xlabel("$e$", fontsize=xlabelsize)

    # Neatening
    for ax in axs.flatten():
        ax.tick_params(labelsize=9, width=1, direction="in", top=True, which="both")
        ax.tick_params(length=4, which="major")
        ax.tick_params(length=2.5, which="minor")
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    # Legend
    axs[1, 0].axis("off")
    handles, labels = axs[2, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles[::-1], labels[::-1], loc="lower center", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
