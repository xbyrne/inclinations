"""
plot_truevmin.py
================
Plots the example figure showing the relationship between distributions on M_min and M.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def prior_mmin(mmin):
    mean = 1.0
    std = 0.2
    prior = np.exp(-0.5 * ((mmin - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return prior


def prior_mtrue(mass):
    i = np.radians(np.linspace(0.001, 90, 100))

    unnormalised_prior = np.array(
        [np.trapezoid(prior_mmin(m * np.sin(i)) * np.sin(i) ** 2, i) for m in mass]
    )
    return unnormalised_prior / np.trapezoid(unnormalised_prior, mass)


def main() -> None:
    # Get the figure path from the first command line argument
    if len(sys.argv) > 1:
        figure_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_truevmin.py <figure_path>")
        sys.exit(1)

    fg, ax = plt.subplots(figsize=(3.3, 1.5))
    masses = np.linspace(0.001, 3, 100)
    ax.plot(masses, prior_mmin(masses), c="k", ls="--")
    ax.plot(masses, prior_mtrue(masses), c="k")

    ax.set_xlabel("Mass [arbitrary]", fontsize=9)
    ax.set_xlim(0, np.max(masses))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticks([0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75], minor=True)
    ax.set_yticks([])
    ax.set_ylabel("Prob. density", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1])

    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(axis="x", labelsize=8, direction="in", which="both", width=1)
    ax.tick_params(length=4, which="major")
    ax.tick_params(length=2, which="minor")

    ax.annotate("$\\Pr(M_{\\rm min})$", xy=(1.2, 1.65), fontsize=10)
    ax.annotate("$\\Pr(M)$", xy=(1.8, 0.35), fontsize=10)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(figure_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
