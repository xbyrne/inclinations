"""
plot_hillstability.py
=====================
Compares the 95% upper limits from CINEMAS to the stability criteria from Chambers+96,
which are in terms of mutual Hill radii.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase

M_SUN = 1.989e30  # kg
M_EARTH = 5.972e24  # kg
ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEMS = ["Barnard's star", "YZ Cet", "HD 28471", "HD 184010", "HD 215152"]

DELTA_CRIT = 10

EXOPLANET_CATALOGUE_PATH = ROOT_DIR / "data" / "exoplanet_catalogue.csv"

RESULTS_PATH = ROOT_DIR / "results" / "mcmc_results"
BURN = 2000


def Delta_factor(a1, a2, M1min, M2min, Mstar):
    """
    Calculate the Hill stability coefficient (for sin(i)^(-1/3) )
    """
    return 2 * ((a2 - a1) / (a1 + a2)) * (3 * Mstar / (M1min + M2min)) ** (1 / 3)


def collect_inclinations_5pc():
    """
    Collects the 5% inclination limits of all systems from the CINEMAS runs.
    """
    inclinations_5pc = {}
    for system in SYSTEMS:
        posterior_file = RESULTS_PATH / f"{system.lower().replace(' ', '_')}.npz"
        if not posterior_file.exists():
            print(f"No posterior found for {system}, skipping...")
            inclinations_5pc[system] = np.nan
            continue
        posterior_samples = np.load(posterior_file)["samples"]
        inclination_samples = np.degrees(
            np.arccos(posterior_samples[BURN:, :, 0].flatten())
        )
        inclinations_5pc[system] = np.percentile(inclination_samples, 5).item()

    return inclinations_5pc


class VerticalLineHandler(HandlerBase):
    def create_artists(
        self, legend, handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        x = width / 2
        return [
            plt.Line2D(
                [x, x],
                [-ydescent, height - ydescent],
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                transform=trans,
            )
        ]


def main():
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = ROOT_DIR / "tex" / "figs" / "fig7_hillstability.png"

    # Loading data
    print("  Loading data...")

    exoplanet_catalogue = pd.read_csv(EXOPLANET_CATALOGUE_PATH)
    planets_data = exoplanet_catalogue[
        exoplanet_catalogue["hostname"].isin(SYSTEMS)
    ].sort_values(["hostname", "pl_orbper"])

    inclinations_5pc = collect_inclinations_5pc()

    # Plotting
    print("  Plotting...")

    fg, axs = plt.subplots(
        3, 2, figsize=(3.3, 4), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )

    i_deg = np.linspace(0, 90, 1000)
    sini_1_3 = np.sin(np.radians(i_deg)) ** (1 / 3)
    Y_MAX = 24
    TITLES = ["Barnard's Star", "YZ Cet", "HD 28471", "HD 184010", "HD 215152"]
    TITLE_DICT = dict(zip(SYSTEMS, TITLES))

    for i in range(6):
        ax = axs.flatten()[i]
        if i == 3:
            ax.axis("off")

            continue
        else:
            system = SYSTEMS[i if i < 3 else i - 1]
            title = TITLE_DICT[system]

        system_data = planets_data[planets_data["hostname"] == system]

        m_star = system_data["st_mass"].values[0] * M_SUN
        m_min = system_data["pl_msinie"].values * M_EARTH

        p_day = system_data["pl_orbper"].values
        a_au = (p_day / 365.25) ** (2 / 3) * (m_star / M_SUN) ** (1 / 3)

        for n, ((a1, a2), (m1, m2)) in enumerate(
            zip(zip(a_au[:-1], a_au[1:]), zip(m_min[:-1], m_min[1:]))
        ):
            if n == 0:
                delta_min = np.inf
                closest_pair = (0, 1)
            Delta = Delta_factor(a1, a2, m1, m2, m_star)
            if Delta < delta_min:
                delta_min = Delta
                closest_pair = (n, n + 1)

        closest_planets = "".join(
            [system_data.iloc[p].pl_name[-1] for p in list(closest_pair)]
        )
        title_text = f"{title}\n$\\Delta_{{\\rm {closest_planets}}}$"

        ax.plot(i_deg, delta_min * sini_1_3, color="k", lw=1)

        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xticks([10, 20, 40, 50, 70, 80], minor=True)
        ax.set_xticklabels([])
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks([0, 10, 20])
        ax.set_yticks([5, 15], minor=True)

        ax.fill_between(i_deg, 0, 10, color="r", alpha=0.2)

        inclination_limit = inclinations_5pc[system]
        ax.vlines(inclination_limit, 8, 12, color="k", lw=1, alpha=0.5)

        ax.set_title(title_text, fontsize=9, ha="left", va="top", x=0.06, y=0.84)

    axs[-1, 0].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", ""], fontsize=8)
    axs[-1, 1].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", "90"], fontsize=8)
    for ax in axs[:, 1]:
        ax.set_yticklabels([])

    for ax in axs.flatten():
        ax.tick_params(labelsize=8, direction="in", top=True, right=True, which="both")
        ax.tick_params(length=3, which="major")
        ax.tick_params(length=1.5, which="minor")

    # Label Chambers+96 limit
    ax = axs[0, 0]
    ax.annotate(
        "$\\Delta < 10$", (50, 5), fontsize=9, color="r", alpha=0.7, va="center"
    )

    # Label prior on HD 28471
    axs[-2, 0].vlines(
        np.degrees(np.arccos(0.95)), 8, 12, color="k", ls=":", lw=1.0, alpha=0.5
    )

    # Add legend to blank plot

    legend_handles = [
        plt.Line2D([0], [0], linewidth=0.5, c="grey", ls=":", label="5% prior"),
        plt.Line2D([0], [0], linewidth=0.5, c="grey", ls="-", label="5% posterior"),
    ]
    axs[1, 1].legend(
        handles=legend_handles,
        fontsize=8,
        loc="center",
        handler_map={plt.Line2D: VerticalLineHandler()},
        handlelength=0.1,
        handleheight=1.7,
        labelspacing=1,
    )

    fg.supxlabel("$i$ [deg]", fontsize=11, y=0.02)

    fg.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
