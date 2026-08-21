"""
plot_hd215152f.py
=================
Plots the HD 215152 f search figure.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_FILE = ROOT_DIR / "results" / "search_hd215152f" / "hd215152f.npz"


def plot_prior(ax):
    ax.vlines(x=[12, 24], ymin=0, ymax=5.0, color="k", linestyle="--")
    ax.hlines(y=5.0, xmin=12, xmax=24, color="k", linestyle="--")

    ax.annotate("Prior", (24, 5.25), fontsize=12, ha="right")


def label_planets(ax):
    ax.annotate("b", (5, 1.7), fontsize=14, c="r")
    ax.annotate("c", (6.5, 1.7), fontsize=14, c="g")
    ax.annotate("d", (10, 1.7), fontsize=14, c="b")
    ax.annotate("e", (24.5, 2.2), fontsize=14, c="c")
    ax.annotate("$\\~{\\rm f}$", (18, 5.3), fontsize=18, c="k", ha="center")


def detection_threshold(P, reference_msini, reference_period):
    return reference_msini * (P / reference_period) ** (1 / 3)


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
        print("Usage: python plot_hd215152f.py <figure_path>")
        sys.exit(1)

    fl = np.load(RESULTS_FILE)
    samples = fl["samples"]

    fg, ax = plt.subplots(figsize=(7, 2.5))

    COLOURS = ["r", "g", "b", "c", "k"]
    THIN = 50
    BURN = 2000
    X_MIN, X_MAX = 3.5, 27

    for i, colour in enumerate(COLOURS):
        period_samples = samples[BURN::THIN, :, 7 + i].flatten()
        msini_samples = samples[BURN::THIN, :, 2 + i].flatten()

        vmax = 0.15 if i == 4 else None
        plot_2d_parameter_histogram(period_samples, msini_samples, ax, colour, vmax)

    # Labelling planets
    label_planets(ax)

    # Add prior on f~'s parameters
    plot_prior(ax)

    # Add approximate detection threshold curve
    P_values = np.linspace(X_MIN, X_MAX, 100)
    M_min_c = 1.526
    M_min_c_sigma = 0.561
    M_min_thresh = detection_threshold(P_values, M_min_c - M_min_c_sigma, 7.283)
    # -1sigma value for planet c's M_min --------^^^^^^^^^^^^^^^^^^^^^^^
    # (This is the planet with the smallest detected RV signal size)
    ax.plot(P_values, M_min_thresh, color="g", linestyle="-.", label="Detection limit")

    # Aesthetics
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 6.8)
    ax.set_xticks([5, 10, 15, 20, 25])
    ax.set_xticks(np.arange(4, 28, 1), minor=True)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticks(np.arange(0, 7, 1), minor=True)

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
