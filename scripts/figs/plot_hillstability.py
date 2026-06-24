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

M_SUN = 1.989e30  # kg
M_EARTH = 5.972e24  # kg
ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEMS = [
    "HD 158259",
    "HD 215152",
    "Barnard's star",
    "HD 184010",
    "HD 28471",
    "YZ Cet",
]

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
    PRIOR_5PC = np.degrees(np.arccos(0.95))

    # Plotting
    print("  Plotting...")

    fg, axs = plt.subplots(
        3, 2, figsize=(5, 6), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )

    i_deg = np.linspace(0, 90, 1000)
    sini_1_3 = np.sin(np.radians(i_deg)) ** (1 / 3)
    Y_MAX = 24
    TITLES = [
        "HD 158259",
        "HD 215152",
        "Barnard's Star",
        "HD 184010",
        "HD 28471",
        "YZ Ceti",
    ]
    TITLE_DICT = dict(zip(SYSTEMS, TITLES))
    for i, system in enumerate(SYSTEMS):
        ax = axs.flatten()[i]
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
        title = TITLE_DICT[system] + "\n" + f"$\\Delta_{{\\rm {closest_planets}}}$"

        ax.plot(i_deg, delta_min * sini_1_3, color="k", lw=1.5)

        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xticklabels([])
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks([0, 10, 20])

        ax.fill_between(i_deg, 0, 10, color="r", alpha=0.2)

        inclination_limit = inclinations_5pc[system]
        ax.vlines(
            PRIOR_5PC, 8, 12, color="k", ls=":", lw=1.5, alpha=0.5, label="5% prior"
        )
        ax.vlines(
            inclination_limit, 8, 12, color="k", lw=1.5, alpha=0.5, label="5% post."
        )

        ax.set_title(title, fontsize=15, ha="left", va="top", x=0.06, y=0.87)

    axs[-1, 0].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", ""], fontsize=14)
    axs[-1, 1].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", "90"], fontsize=14)
    for ax in axs[:, 1]:
        ax.set_yticklabels([])

    for ax in axs.flatten():
        ax.tick_params(
            length=3, labelsize=13, width=1.5, direction="in", top=True, right=True
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Label CINEMAS limits
    axs[-1, -1].legend(
        loc=[0.47, 0.5],
        fontsize=10,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )

    # Label Chambers+96 limit
    axs[0, 0].annotate(
        "$\\Delta < 10$",
        (55, 4.8),
        fontsize=14,
        va="center",
        ha="center",
        color="r",
        alpha=0.7,
    )

    fg.supxlabel("$i$ [deg]", fontsize=16, y=0.02)

    fg.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
