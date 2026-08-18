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

SYSTEMS = ["Barnard's star", "HD 215152", "HD 184010", "HD 28471", "YZ Cet"]

MOSAIC = [["HD 215152"] * 2, ["Barnard's star", "HD 184010"], ["HD 28471", "YZ Cet"]]

COLOURS = {"b": "r", "c": "g", "d": "b", "e": "m", "f": "c"}

XTICKS = {
    "HD 215152": {"major": np.arange(5, 29, 5), "minor": np.arange(5, 28, 1)},
    "Barnard's star": {"major": np.arange(2, 8, 1), "minor": np.arange(2, 8, 0.5)},
    "HD 184010": {
        "major": np.arange(200, 1100, 200),
        "minor": np.arange(200, 950, 100),
    },
    "HD 28471": {"major": np.arange(2, 14, 2), "minor": np.arange(2, 13, 1)},
    "YZ Cet": {"major": np.arange(1, 6, 1), "minor": np.arange(1, 5, 0.5)},
}

YTICKS = {
    "HD 215152": {"major": np.arange(0, 11, 2), "minor": np.arange(0, 10, 1)},
    "Barnard's star": {
        "major": np.arange(0, 1.5, 0.5),
        "minor": np.arange(0, 1.5, 0.1),
    },
    "HD 184010": {"major": np.arange(100, 500, 100), "minor": np.arange(50, 500, 50)},
    "HD 28471": {"major": np.arange(5, 20, 5), "minor": np.arange(2, 19, 1)},
    "YZ Cet": {"major": np.arange(1, 4, 1), "minor": np.arange(0.5, 4, 0.5)},
}


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
    fg, axs = plt.subplot_mosaic(MOSAIC, figsize=(7, 8), gridspec_kw={"wspace": 0.2})

    for system, ax in axs.items():
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
            x, y = 0.025, 0.8
        else:
            x, y = 0.054, 0.8
        ax.set_title(title, fontsize=14, x=x, y=y, horizontalalignment="left")

        # Ticks
        ax.set_xticks(XTICKS[system]["major"])
        ax.set_xticks(XTICKS[system]["minor"], minor=True)
        ax.set_yticks(YTICKS[system]["major"])
        ax.set_yticks(YTICKS[system]["minor"], minor=True)

        ax.tick_params(labelsize=11, width=1, direction="in", right=True, which="both")
        ax.tick_params(which="major", length=5, axis="both")
        ax.tick_params(which="minor", length=2.5, axis="both")
        ax.tick_params(length=5, labelsize=11, width=1, direction="in", right=True)

    # Axes tweaks
    axs["HD 215152"].set_xlim(3.5, 27)
    axs["HD 215152"].set_ylim(0.0, 11.0)
    axs["HD 184010"].set_xlim(140, 980)
    axs["Barnard's star"].set_ylim(0.0, 1.4)
    axs["YZ Cet"].set_ylim(0.0, 3.91)

    fg.supxlabel("Period [d]", fontsize=16, y=0.045)
    fg.supylabel("True Mass [M$_{\\oplus}$]", fontsize=16, x=0.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
