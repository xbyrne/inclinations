"""
plot_walkers.py
===============
Plots the MCMC walkers for all parameters, for the YZ Ceti system run.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

ROOT_DIR = Path(__file__).resolve().parents[2]

PLANETS = ["b", "c", "d"]
N_PLANETS = len(PLANETS)


def gather_samples():
    results_dir = ROOT_DIR / "results" / "mcmc_results"
    system = "YZ Cet"
    fl = np.load(results_dir / f"{system.lower().replace(' ', '_')}.npz")
    samples = fl["samples"]
    return samples


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_walkers.py <figure_path>")
        sys.exit(1)

    # ==============
    # Gathering data

    samples = gather_samples()

    burn_in = 2000
    parameters = ["$\\cos i$", "$M_*$ [$M_{\\odot}$]"] + PLANETS * 3 + PLANETS[1:] * 2

    n_params = N_PLANETS * 5

    fg, axs = plt.subplots(
        n_params,
        2,
        figsize=(12, 1 * n_params),
        gridspec_kw={"width_ratios": [5, 1], "wspace": 0.01, "hspace": 0.0},
    )

    for i in range(samples.shape[-1]):
        # =======
        # Walkers
        ax = axs[i, 0]
        ax.plot(samples[:burn_in, :, i], alpha=0.2, c="r")  # Burn-in in red
        ax.plot(
            np.arange(burn_in, samples.shape[0]),
            samples[burn_in:, :, i],
            alpha=0.2,
            c="k",
        )  # Post burn-in in black

        # x-axis labelling
        if i < n_params - 1:
            ax.set_xticklabels([])
        # y-axis labelling
        if i <= 1:
            ax.set_ylabel(parameters[i], fontsize=17, rotation=0, labelpad=45)
        else:
            ax.set_ylabel(parameters[i], fontsize=17, rotation=0, labelpad=20)

        ax.set_xlim(0, samples.shape[0])
        if i == 0:
            ax.set_ylim(0, 1)
        if 2 + 2 * N_PLANETS <= i < 2 + 3 * N_PLANETS:
            ax.set_ylim(0, ax.get_ylim()[1])
        if i >= 2 + 3 * N_PLANETS:
            ax.set_ylim(0, 360)
        ax.tick_params(
            axis="both",
            right=True,
            top=True,
            direction="in",
            labelsize=11,
            length=6,
        )
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # ==========
        # Histograms
        ax = axs[i, 1]
        ax.hist(
            samples[burn_in:, :, i].flatten(),
            bins=30,
            color="k",
            orientation="horizontal",
        )
        ax.set_xticks([])
        ax.set_yticks(axs[i, 0].get_yticks())
        ax.set_yticklabels([])
        ax.set_ylim(axs[i, 0].get_ylim())
        ax.tick_params(axis="both", right=True, top=True, direction="in", length=6)

    # Labelling (kinda complex, sorry ruff)
    for i, label, nspaces in zip(
        [3, 6, 9],
        ["$M_{\\rm{min}}$ [M$_{\\oplus}$]", "$P$ [d]", "$e$"],
        [1, 5, 7],
    ):
        plot_label = "┌─" + (" " * nspaces) + f"  {label}  " + (" " * nspaces) + "─┐"
        axs[i, 0].annotate(
            plot_label,
            xy=(-0.2, 0.5),
            xycoords="axes fraction",
            fontsize=18,
            rotation=90,
            va="center",
        )

    axs[-1, 0].annotate(
        "┌─ $\\Delta f_0$ [$^\\circ$] ─┐",
        xy=(-0.2, 1.0),
        xycoords="axes fraction",
        fontsize=18,
        rotation=90,
        va="center",
    )
    axs[-3, 0].annotate(
        "┌─ $\\Delta\\varpi$ [$^\\circ$] ─┐",
        xy=(-0.2, 1.0),
        xycoords="axes fraction",
        fontsize=18,
        rotation=90,
        va="center",
    )

    for i in range(2 + N_PLANETS * 2, 2 + N_PLANETS * 3):
        # Eccentricities
        axs[i, 0].set_ylim(0, 0.25)
        axs[i, 1].set_ylim(0, 0.25)
    for i in range(1, 5):
        axs[-i, 0].set_yticks([0, 180])

    axs[-1, 0].set_xlabel("MCMC Iteration", fontsize=18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
