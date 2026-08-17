"""
tabulate_systems.py
===================
Builds and saves the systems summary LaTeX table.
"""

import sys
from pathlib import Path

import pandas as pd
from cinemas.dataloading import select_compact_multiplanet_rv_systems

ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEMS = [
    "HD 158259",
    "HD 215152",
    "Barnard's star",
    "HD 184010",
    "HD 28471",
    "YZ Cet",
]


def load_catalogue() -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")


def filter_problematic_systems(catalogue: pd.DataFrame) -> pd.DataFrame:
    # Remove planets which have been refuted
    refuted_planets = ["GJ 667 C f", "GJ 667 C e", "GJ 667 C g", "DMPP-1 e"]
    filtered_catalogue = catalogue[~catalogue["pl_name"].isin(refuted_planets)].copy()

    # Remove HD 158259
    # Planet b transits, but planets c--f do not.
    # Its inclination is therefore already pretty well-constrained
    filtered_catalogue = filtered_catalogue[
        filtered_catalogue["hostname"] != "HD 158259"
    ]
    return filtered_catalogue


def build_system_summary_df(catalogue: pd.DataFrame) -> pd.DataFrame:
    compact_systems = select_compact_multiplanet_rv_systems(catalogue)
    compact_systems["hostname"] = pd.Categorical(
        compact_systems["hostname"], categories=SYSTEMS, ordered=True
    )
    compact_systems.sort_values(by=["hostname", "pl_orbper"], inplace=True)
    # Convert the Categorical back to a string type for the DataFrame
    compact_systems["hostname"] = compact_systems["hostname"].astype(str)

    systems_df = pd.DataFrame(
        columns=["star_name", "star_mass", "planet_name", "period", "mmin"]
    )
    systems_df["star_name"] = compact_systems["hostname"]
    systems_df["star_mass"] = compact_systems["st_mass"]
    systems_df["planet_name"] = compact_systems["pl_name"]
    systems_df["period"] = compact_systems["pl_orbper"]
    systems_df["mmin"] = compact_systems["pl_msinie"]

    return systems_df


def format_systems_table(systems_df: pd.DataFrame) -> str:
    latex_df = systems_df.copy()
    latex_df["planet_name"] = latex_df["planet_name"].apply(lambda x: x[-1])  # Letter
    latex_df["mmin"] = latex_df["mmin"].apply(lambda x: f"{x:.2f}")
    latex_df["period"] = latex_df["period"].apply(lambda x: f"{x:.2f}")
    latex_df.rename(
        columns={
            "planet_name": "Planet",
            "mmin": "$M_{\\rm min}$~[$M_{\\earth}$]",
            "period": "$P$~[d]",
        },
        inplace=True,
    )

    this_system = ""
    for i, row in latex_df.iterrows():
        if row["star_name"] != this_system:
            this_system = row["star_name"]
            latex_df.at[i, "star_name"] = f"\\hline\n{this_system}"
            stellar_mass = row["star_mass"]
            label_next_row_with_mass = True
        else:
            if label_next_row_with_mass:
                latex_df.at[i, "star_name"] = f"(${stellar_mass:.2f}~M_{{\\sun}}$)"
                label_next_row_with_mass = False
            else:
                if row["star_name"] == this_system:
                    latex_df.at[i, "star_name"] = ""

    latex_df.rename(columns={"star_name": "Star (mass)"}, inplace=True)
    latex_df = latex_df[
        [
            "Star (mass)",
            "Planet",
            "$P$~[d]",
            "$M_{\\rm min}$~[$M_{\\earth}$]",
        ]
    ]

    latex_table = latex_df.to_latex(column_format="rcrr", index=False)
    latex_table = (
        latex_table.replace("\\toprule", "\\hline")
        .replace("\\midrule", "")
        .replace("\\bottomrule", "\\hline")
        .replace("Barnard's star", "Barnard's Star")
        .replace("YZ Cet", "YZ Ceti")
    )

    return latex_table


def build_systems_table(catalogue: pd.DataFrame) -> str:
    filtered_catalogue = filter_problematic_systems(catalogue)
    systems_df = build_system_summary_df(filtered_catalogue)
    return format_systems_table(systems_df)


def main() -> None:
    output_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT_DIR / "tex" / "tables" / "systems.tex"
    )

    catalogue = load_catalogue()
    latex_table = build_systems_table(catalogue)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        file.write(latex_table)

    print(f"Saved systems table to {output_path}")


if __name__ == "__main__":
    main()
