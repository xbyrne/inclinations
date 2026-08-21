"""
plot_barnardsstarf.py
======================
Plots the Barnard's Star f search figure.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_FILE = ROOT_DIR / "results" / "search_barnardsstarf" / "barnardsstarf.npz"


def plot_prior(ax):
    ax.vlines(x=[1_000, 10_000], ymin=50, ymax=500, color="k", linestyle="--")
    ax.hlines(y=[50, 500], xmin=1_000, xmax=10_000, color="k", linestyle="--")

    ax.annotate("Prior", (5000, 600), fontsize=12, ha="right")


def detection_threshold(P):
    a = 0.07
    b = 1 / 3
    return a * (P**b)


def plot_2d_parameter_histogram(x_samples, y_samples, ax, colour, vmax=None):
    cmap = LinearSegmentedColormap.from_list(
        "white_to_colour", ["white", colour], N=256
    )

    x_range = [
        np.round(x_samples.min() - 0.1, 1),
        np.round(x_samples.max() + 0.1, 1),
    ]
    y_range = np.percentile(y_samples, [0.5, 99.5])
    ranges = [x_range, y_range]

    ax.hist2d(
        x_samples,
        y_samples,
        bins=[np.arange(*x_range, 0.1), np.linspace(*y_range, 50)],
        cmap=cmap,
        range=ranges,
        density=True,
        vmax=vmax,
    )


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_barnardsstarf.py <figure_path>")
        sys.exit(1)

    fl = np.load(RESULTS_FILE)
    samples = fl["samples"]

    fg, ax = plt.subplots(figsize=(7, 2.5))

    THIN = 50
    BURN = 2000
    X_MIN, X_MAX = 700, 11000

    period_samples = samples[BURN::THIN, :, 12].flatten()
    msini_samples = samples[BURN::THIN, :, 7].flatten()

    plot_2d_parameter_histogram(period_samples, msini_samples, ax, "k")

    # Add prior on f~'s parameters
    plot_prior(ax)

    # Add approximate detection threshold curve
    P_values = np.linspace(X_MIN, X_MAX, 100)
    M_min_thresh = detection_threshold(P_values)
    ax.plot(P_values, M_min_thresh, color="c", linestyle="-.", label="Detection limit")

    # Aesthetics
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 600)
    ax.set_xticks([2000, 4000, 6000, 8000, 10000])
    ax.set_xticks(np.arange(500, 12000, 500), minor=True)
    ax.set_yticks([200, 400, 600])
    ax.set_yticks(np.arange(0, 700, 100), minor=True)

    ax.tick_params(axis="both", which="both", direction="in", labelsize=12, width=1)
    ax.tick_params(length=6, which="major")
    ax.tick_params(length=3, which="minor")

    ax.set_xlabel("Period [d]", fontsize=14)
    ax.set_ylabel("$M_{\\rm min}$ [$M_\\oplus$]", fontsize=14)
    ax.legend(loc="upper left", fontsize=12)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
