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


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_allinclinations.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data

    # Load catalogue and select systems
    catalogue = pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")
    compact_multiplanet_rv_systems = dataloading.select_compact_multiplanet_rv_systems(
        catalogue
    )
    compact_multiplanet_rv_systems.sort_values(by="hostname", inplace=True)
    system_list = (
        compact_multiplanet_rv_systems.groupby("hostname")
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    system_list = [s for s in system_list if s != "DMPP-1"]

    # Collect inclination samples
    results_path = ROOT_DIR / "results" / "mcmc_results"
    inclination_samples = {}
    burn_in = 2000
    for system in system_list:
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
        3, 2, figsize=(6, 7), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )
    x = np.linspace(0, 90, 100)
    for system, ax in zip(system_list, axs.flatten()):
        display_name = system if system != "Barnard's star" else "Barnard's Star"
        ax.set_title(
            display_name, x=0.055, y=0.78, fontsize=15, horizontalalignment="left"
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        samples = inclination_samples[system]
        ax.set_yticks([])
        if len(samples) == 0:
            continue

        ax.hist(samples, bins=np.arange(0, 91, 3), color="k", density=True, alpha=0.4)
        ax.plot(x, np.sin(np.radians(x)) * np.pi / 180, c="k", lw=2)
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xticklabels([])

    axs[-1, 0].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", ""], fontsize=14)
    axs[-1, 1].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", "90"], fontsize=14)

    for ax in axs.flatten():
        ax.tick_params(length=5, labelsize=14, width=1.5, direction="in", top=True)

    ax = axs[0, 0]
    ax.annotate("Prior", xy=(15, 0.007), fontsize=13, rotation=33)
    ax.annotate("Posterior", xy=(52, 0.004), fontsize=13)

    fg.supxlabel("$i$ [deg]", fontsize=16, y=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
