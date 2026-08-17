"""
plot_allinclinations.py
=======================
Plots the distributions of inclinations for all systems in the sample.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cinemas import dataloading

ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEMS = ["HD 215152", "Barnard's star", "HD 184010", "HD 28471"]


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_allinclinations.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data

    # Load catalogue
    catalogue = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    compact_multiplanet_rv_systems = dataloading.select_compact_multiplanet_rv_systems(
        catalogue
    )
    compact_multiplanet_rv_systems.sort_values(by="hostname", inplace=True)
    # Remove YZ Cet, already plotted
    compact_multiplanet_rv_systems = compact_multiplanet_rv_systems[
        compact_multiplanet_rv_systems["hostname"] != "YZ Cet"
    ]

    # Collect inclination samples
    results_path = ROOT_DIR / "results" / "mcmc_results"
    inclination_samples = {}
    burn_in = 2000
    for system in SYSTEMS:
        posterior_file = results_path / f"{system.lower().replace(' ', '_')}.npz"
        if not posterior_file.exists():
            print(f"No posterior found for {system}, skipping...")
            inclination_samples[system] = np.array([])
            continue
        posterior_samples = np.load(posterior_file)["samples"]
        inclination_samples[system] = np.degrees(
            np.arccos(posterior_samples[burn_in:, :, 0].flatten())
        )

    # ========
    # Plotting
    fg, axs = plt.subplots(
        2, 2, figsize=(3.3, 3), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )
    x = np.linspace(0, 90, 100)
    for system, ax in zip(SYSTEMS, axs.flatten()):
        display_name = system if system != "Barnard's star" else "Barnard's Star"
        ax.set_title(display_name, fontsize=9, x=0.055, y=0.75, ha="left")

        for spine in ax.spines.values():
            spine.set_linewidth(1)

        samples = inclination_samples[system]
        ax.set_yticks([])
        if len(samples) == 0:
            continue

        ax.hist(samples, bins=np.arange(0, 91, 3), color="k", density=True, alpha=0.4)
        ax.plot(x, np.sin(np.radians(x)) * np.pi / 180, c="k", lw=1)
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xticks([10, 20, 40, 50, 70, 80], minor=True)
        ax.set_xticklabels([])

    axs[-1, 0].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", ""], fontsize=14)
    axs[-1, 1].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", "90"], fontsize=14)

    for ax in axs.flatten():
        ax.tick_params(labelsize=8, width=1, direction="in", top=True, which="both")
        ax.tick_params(length=4, which="major")
        ax.tick_params(length=2, which="minor")

    ax = axs[0, 0]
    ax.annotate("Prior", xy=(15, 0.0075), fontsize=9, rotation=38)
    ax.annotate("Posterior", xy=(47, 0.004), fontsize=9)

    fg.supxlabel("$i$ [deg]", fontsize=10, y=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
