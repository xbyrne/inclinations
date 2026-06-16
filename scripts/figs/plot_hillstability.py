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
    "Barnard's star",
    "HD 215152",
    "HD 184010",
    "HD 28471",
    "YZ Cet",
]

DELTA_CRIT = 10

MASSES95_PATH = ROOT_DIR / "tex" / "tables" / "masses95.csv"
EXOPLANET_CATALOGUE_PATH = ROOT_DIR / "data" / "exoplanet_catalogue.csv"

COLOURS = {"b": "r", "c": "g", "d": "b", "e": "m", "f": "c"}
SUBPLOT_MOSAIC = [
    ["HD 158259"] * 2,
    ["HD 215152"] * 2,
    ["Barnard's star", "HD 184010"],
    ["HD 28471", "YZ Cet"],
]


def critical_m2(m1, P1, P2, m_star, Delta_crit=10):
    """
    Calculate the critical mass for Hill stability.
    Masses should be in kg, and periods in days.
    """
    a1 = (P1 / 365.25) ** (2 / 3) * (m_star / M_SUN) ** (1 / 3)  # AU
    a2 = (P2 / 365.25) ** (2 / 3) * (m_star / M_SUN) ** (1 / 3)  # AU

    max_m1_plus_m2 = 3 * m_star * ((2 / Delta_crit) * ((a2 - a1) / (a1 + a2))) ** 3
    return max_m1_plus_m2 - m1


def get_critical_masses(m1, P1, m_star, Delta_crit=10):
    """
    Return values of P and corresponding values of critical_m2
    """

    P_low = np.linspace(P1 * 0.1, P1, 100)
    P_high = np.linspace(P1, 5 * P1, 100)
    critical_m2_low = critical_m2(m1, P_low, P1, m_star, Delta_crit)
    critical_m2_high = critical_m2(m1, P1, P_high, m_star, Delta_crit)

    P_values = np.concatenate([P_low, P_high])
    critical_m2_values = np.concatenate([critical_m2_low, critical_m2_high])

    return P_values, critical_m2_values


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

    masses95_df = pd.read_csv(MASSES95_PATH)

    # Plotting
    print(  "Plotting...")

    fg, axs = plt.subplot_mosaic(
        SUBPLOT_MOSAIC, figsize=(7, 9), gridspec_kw={"wspace": 0.2, "hspace": 0.4}
    )

    for system in SYSTEMS:
        ax = axs[system]
        system_data = planets_data[planets_data["hostname"] == system]

        m_star = system_data["st_mass"].values[0]
        m_min = system_data["pl_msinie"].values
        p_day = system_data["pl_orbper"].values

        mass95_data = masses95_df[masses95_df["hostname"] == system]
        posterior_masses95 = mass95_data["posterior_95"].values

        roof = 1.3 * max(posterior_masses95)

        for i in range(len(system_data)):
            m1 = m_min[i]
            m_max = posterior_masses95[i]
            P1 = p_day[i]

            planet_name = system_data["pl_name"].values[i][-1:]  # Just letter
            colour = COLOURS.get(planet_name, "k")

            P_values, critical_m2_values = get_critical_masses(
                m_max * M_EARTH, P1, m_star * M_SUN, DELTA_CRIT
            )

            ax.scatter(P1, m1, color=colour, marker="^", alpha=0.6)
            ax.scatter(P1, m_max, color=colour, marker="v")

            ax.fill_between(
                P_values, critical_m2_values / M_EARTH, roof, color=colour, alpha=0.3
            )

        ax.set_xlim(0.8 * min(p_day), 1.2 * max(p_day))
        ax.set_ylim(0, roof)

        ax.tick_params(length=5, labelsize=11, width=1, direction="in")

        title = system if system != "Barnard's star" else "Barnard's Star"
        ax.set_title(title, fontsize=14, loc="left")

    ax = axs["HD 158259"]
    ax.annotate("─ 95% limit", (12.5, 10.8), fontsize=13)
    ax.annotate("─ $M_{\\rm min}$", (12.5, 5.4), fontsize=13)
    ax.set_xlim(1.5, 18)
    ax.annotate("$\\Delta = 10$", (16.5, 2), rotation=61, fontsize=12, color="#007777")

    fg.supxlabel("Period [d]", fontsize=16, y=0.04)
    fg.supylabel("Mass [M$_{\\oplus}$]", fontsize=16, x=0.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
