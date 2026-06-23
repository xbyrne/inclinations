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

    fg, ax = plt.subplots(figsize=(6, 2.5))
    masses = np.linspace(0.001, 3, 100)
    ax.plot(masses, prior_mmin(masses), c="k", ls="--", lw=2)
    ax.plot(masses, prior_mtrue(masses), c="k", lw=2)

    ax.set_xlabel("Mass [arbitrary]", fontsize=14)
    ax.set_xlim(0, np.max(masses))
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([])
    ax.set_ylabel("Prob. density", fontsize=14)
    ax.set_ylim(0, ax.get_ylim()[1])

    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis="x", length=7, labelsize=14, direction="in", width=1.5)

    ax.annotate("$\\Pr(M_{\\rm min})$", xy=(0.25, 1.6), fontsize=16)
    ax.annotate("$\\Pr(M)$", xy=(1.8, 0.4), fontsize=16)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(figure_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
