"""
plot_masses_vs_periods.py
=========================
Plots a figure comparing the prior and posterior distributions of true masses,
accounting for stability.
"""

import sys
from pathlib import Path

import cinemas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEMS = [
    "HD 158259",
    "Barnard's star",
    "HD 215152",
    "HD 184010",
    "HD 28471",
    "YZ Cet",
]

MOSAIC = [
    # ["Example"] * 2 + ["HD 158259"] * 4,
    ["HD 158259"] * 4 + ["Example"] * 2,
    # ["HD 158259"] * 6,
    ["HD 215152"] * 6,
    ["Barnard's star"] * 3 + ["HD 184010"] * 3,
    # ["HD 184010"] * 3 + ["Barnard's star"] * 3,
    ["HD 28471"] * 3 + ["YZ Cet"] * 3,
]

COLOURS = {"b": "r", "c": "g", "d": "b", "e": "m", "f": "c"}


def truncate_samples(samples):
    """Truncate samples between 5th and 95th percentiles."""
    return samples[
        (samples >= np.percentile(samples, 0.1))
        & (samples <= np.percentile(samples, 95))
    ]


def collect_prior_samples(system_obses):
    prior_samples = {}

    inclination_draws = np.arccos(np.random.uniform(0, 1, size=10000))
    for system in SYSTEMS:
        system_obs = system_obses[system]
        prior_samples[system] = {}

        for planet_obs in system_obs.planet_observations:
            minimum_mass_draws = np.random.normal(
                planet_obs.minimum_mass.mean, planet_obs.minimum_mass.error, size=10000
            )
            true_mass_draws = minimum_mass_draws / np.sin(inclination_draws)
            prior_samples[system][planet_obs.name] = truncate_samples(true_mass_draws)

    return prior_samples


def collect_posterior_samples(system_obses, burn_in: int = 2000):
    posterior_samples = {}
    burn_in = 2000

    for system in SYSTEMS:
        system_obs = system_obses[system]

        posterior_filename = f"{system.lower().replace(' ', '_')}.npz"
        posterior_filepath = ROOT_DIR / "results" / "mcmc_results" / posterior_filename
        if not posterior_filepath.exists():
            print(f"   No posterior found for {system}, skipping...")
            continue

        posterior_samples[system] = {}
        mcmc_samples = np.load(posterior_filepath)["samples"]
        inclination_samples = np.arccos(mcmc_samples[burn_in:, :, 0].flatten())

        for i, planet_obs in enumerate(system_obs.planet_observations):
            minimum_mass_samples = mcmc_samples[burn_in:, :, 2 + i].flatten()
            true_mass_samples = minimum_mass_samples / np.sin(inclination_samples)
            posterior_samples[system][planet_obs.name] = truncate_samples(
                true_mass_samples
            )
    return posterior_samples


def min_period_gaps(system_obses):
    widths = {}
    for system in SYSTEMS:
        system_obs = system_obses[system]
        periods = sorted(
            [planet_obs.period.mean for planet_obs in system_obs.planet_observations]
        )
        period_gaps = np.diff(periods)
        widths[system] = 0.9 * np.min(period_gaps)

    return widths


def make_violin(samples, planet_obs, side, ax, colour, width=0.5, alpha=0.3):
    period = planet_obs.period.mean
    violin = ax.violinplot(
        samples,
        positions=[period],
        showextrema=False,
        quantiles=[1.0],
        side=side,
        widths=width,
    )

    # Colouring in
    notch = violin["cquantiles"]
    vertices = notch.get_paths()[0].vertices

    # Change the lengths of the lines
    if side == "low":
        vertices[0, 0] = vertices[1, 0] - 0.4 * width
    else:
        vertices[1, 0] = vertices[0, 0] + 0.4 * width
    notch.set_paths([vertices])

    notch.set_linewidth(1 if side == "low" else 1.7)
    notch.set_color(colour)

    violin["bodies"][0].set_color(colour)
    violin["bodies"][0].set_alpha(alpha)

    return violin


def annotate_example_subplot(col, ax):
    ax.annotate("Prior", xy=(5.7, 0.6), fontsize=11, color=col, va="top", ha="right")
    ax.annotate(
        "Post.",
        xy=(5.8, 0.6),
        fontsize=11,
        color=col,
        va="top",
        ha="left",
        weight="bold",
    )
    ax.annotate("95%", xy=(5.55, 6.31), fontsize=11, color=col)
    ax.annotate("95%", xy=(5.8, 3.9), fontsize=11, color=col, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(5.4, 6.15)
    ax.set_ylim(-0.5, 7.4)


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_masses_vs_periods.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data
    print("  Gathering exoplanet data...")

    # Catalogue
    exoplanet_catalogue = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    system_obses = {
        system: cinemas.dataloading.load_system_observations(
            system, exoplanet_catalogue
        )
        for system in SYSTEMS
    }

    # =================
    # Gathering samples
    # (generating prior samples and collecting posterior samples)
    print("   Collecting samples...")
    # Prior and posterior samples
    burn_in = 2000
    prior_samples = collect_prior_samples(system_obses)
    posterior_samples = collect_posterior_samples(system_obses, burn_in=burn_in)

    # Widths of violins
    widths = min_period_gaps(system_obses)

    # ========
    # Plotting
    print("  Plotting...")
    fg, axs = plt.subplot_mosaic(MOSAIC, figsize=(7, 10), gridspec_kw={"wspace": 0.7})

    for system, ax in axs.items():
        if system == "Example":
            eg_col = "#333333"
            eg_prior_samples = prior_samples["HD 215152"]["HD 215152 b"]
            eg_posterior_samples = posterior_samples["HD 215152"]["HD 215152 b"]
            eg_planet_obs = system_obses["HD 215152"].planet_observations[0]

            # Plot the violin of HD 215152 b as an example
            make_violin(
                eg_prior_samples, eg_planet_obs, "low", ax, eg_col, width=0.5, alpha=0.2
            )
            make_violin(
                eg_posterior_samples,
                eg_planet_obs,
                "high",
                ax,
                eg_col,
                width=0.5,
                alpha=0.4,
            )
            annotate_example_subplot(eg_col, ax)
            continue

        print(f"   Plotting {system}...")
        for planet_obs in system_obses[system].planet_observations:
            colour = COLOURS.get(planet_obs.name[-1], "k")
            width = widths[system]
            # Prior
            samples = prior_samples[system][planet_obs.name]
            make_violin(
                samples,
                planet_obs,
                side="low",
                ax=ax,
                colour=colour,
                width=width,
                alpha=0.2,
            )
            # Posterior
            if posterior_samples.get(system) is None:
                continue
            samples = posterior_samples[system][planet_obs.name]
            make_violin(
                samples,
                planet_obs,
                side="high",
                ax=ax,
                colour=colour,
                width=width,
                alpha=0.4,
            )

        title = system if system != "Barnard's star" else "Barnard's Star"
        if title == "HD 215152":
            x, y = 0.025, 0.78
        elif title == "HD 158259":
            x, y = 0.04, 0.78
        else:
            x, y = 0.055, 0.78
        ax.set_title(title, fontsize=14, x=x, y=y, horizontalalignment="left")

        ax.tick_params(length=5, labelsize=11, width=1, direction="in", right=True)

    # Axes tweaks
    axs["HD 158259"].set_ylim(0.0, 23.5)
    axs["HD 215152"].set_xlim(3.5, 27)
    axs["HD 215152"].set_ylim(0.0, 11.0)
    axs["HD 184010"].set_xlim(140, 980)
    axs["Barnard's star"].set_ylim(0.0, 1.4)
    # axs["Barnard's star"].set_xticks([0, 1], labels=["0", "1"])
    axs["YZ Cet"].set_ylim(0.0, 3.91)

    fg.supxlabel("Period [d]", fontsize=16, y=0.045)
    fg.supylabel("True Mass [M$_{\\oplus}$]", fontsize=16, x=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
