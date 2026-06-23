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
        3, 2, figsize=(6, 7), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
    )

    i_deg = np.linspace(0, 90, 1000)
    sini_1_3 = np.sin(np.radians(i_deg)) ** (1 / 3)

    for i, system in enumerate(SYSTEMS):
        ax = axs.flatten()[i]
        system_data = planets_data[planets_data["hostname"] == system]

        m_star = system_data["st_mass"].values[0] * M_SUN
        m_min = system_data["pl_msinie"].values * M_EARTH

        p_day = system_data["pl_orbper"].values
        a_au = (p_day / 365.25) ** (2 / 3) * (m_star / M_SUN) ** (1 / 3)

        for (a1, a2), (m1, m2) in zip(
            zip(a_au[:-1], a_au[1:]), zip(m_min[:-1], m_min[1:])
        ):
            Delta = Delta_factor(a1, a2, m1, m2, m_star)
            ax.plot(i_deg, Delta * sini_1_3, color="k", lw=1.5)

        ax.set_xlim(0, 90)
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xticklabels([])
        ax.set_ylim(0, 27)
        ax.set_yticks([0, 10, 20])

        # ax.axhline(10, color="k", ls="--", lw=1.5)
        ax.fill_between(i_deg, 0, 10, color="r", alpha=0.3)

        inclination_limit = inclinations_5pc[system]
        ax.vlines(inclination_limit, 0, 27, color="k", lw=1.5, alpha=0.5)
        ax.vlines(PRIOR_5PC, 0, 27, color="k", ls="--", lw=1.5, alpha=0.5)

        title = system if system != "Barnard's star" else "Barnard"
        ax.set_title(title, fontsize=15, loc="right", x=0.97, y=0.78)

    axs[-1, 0].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", ""], fontsize=14)
    axs[-1, 1].set_xticks([0, 30, 60, 90], labels=["0", "30", "60", "90"], fontsize=14)
    for ax in axs[:, 1]:
        ax.set_yticklabels([])

    for ax in axs.flatten():
        ax.tick_params(
            length=5, labelsize=13, width=1.5, direction="in", top=True, right=True
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Label CINEMAS limits
    ax = axs[0, 0]
    ax.annotate(
        "Posterior", xy=(27, 19.5), va="center", fontsize=12, rotation=90, alpha=0.7
    )
    ax.annotate(
        "Prior", xy=(11, 19.5), va="center", fontsize=12, rotation=90, alpha=0.7
    )
    for y in [17, 19]:
        # Short rightwards arrow, from x=lower limit for HD 158259
        ax.arrow(
            inclinations_5pc["HD 158259"], y, 5, 0, head_width=0.8, head_length=1.4
        )

    fg.supxlabel("$i$ [deg]", fontsize=16, y=0.02)
    fg.supylabel("$\\Delta$", fontsize=18, x=0.01)

    fg.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
