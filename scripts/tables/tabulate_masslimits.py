"""
tabulate_masslimits.py
======================
Builds and saves the LaTeX table of 95% prior/posterior true-mass limits.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cinemas import dataloading as dl
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def true_mass_function(minimum_mass_function: callable) -> callable:
    i = np.radians(np.linspace(0, 90, 1000))

    def integrand(mass, inclination):
        return (
            minimum_mass_function(mass * np.sin(inclination)) * np.sin(inclination) ** 2
        )

    def prior(mass):
        return np.array([np.trapezoid(integrand(mi, i), i) for mi in mass])

    return prior


def load_catalogue() -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / "data" / "exoplanet_catalogue.csv")


def get_system_posterior_samples(
    results_dir: Path, system: str, burn_in: int
) -> np.ndarray | None:
    posterior_file = results_dir / f"{system.lower().replace(' ', '_')}.npz"
    if not posterior_file.exists():
        return None
    return np.load(posterior_file)["samples"][burn_in:, :, :]


def compute_prior_95(planet_obs) -> float:
    true_mass_prior_fn = true_mass_function(
        lambda mass: gaussian(
            mass, planet_obs.minimum_mass.mean, planet_obs.minimum_mass.error
        )
    )
    x_long = np.linspace(
        planet_obs.minimum_mass.mean * 0.1,
        planet_obs.minimum_mass.mean * 10,
        10000,
    )
    prior_values = true_mass_prior_fn(x_long)
    return x_long[
        np.where(np.cumsum(prior_values) / np.sum(prior_values) >= 0.95)[0][0]
    ].item()


def collect_mass_limit_maps(
    catalogue: pd.DataFrame,
    systems: list[str],
    results_dir: Path,
    burn_in: int,
) -> tuple[dict[str, float], dict[str, float]]:
    true_mass_prior_95s = {}
    true_mass_posterior_95s = {}

    for system in tqdm(systems, total=len(systems), desc="Collecting mass limits"):
        system_obs = dl.load_system_observations(system, catalogue)

        posterior_samples = get_system_posterior_samples(results_dir, system, burn_in)
        if posterior_samples is None:
            print(f"No posterior found for {system}, skipping...")
            for planet in system_obs.planet_observations:
                true_mass_prior_95s[planet.name] = np.nan
                true_mass_posterior_95s[planet.name] = np.nan
            continue

        inclination_samples = np.arccos(posterior_samples[:, :, 0].flatten())
        minimum_mass_samples = posterior_samples[
            :, :, 2 : 2 + system_obs.n_planets
        ].reshape(-1, system_obs.n_planets)

        for i, planet_obs in enumerate(system_obs.planet_observations):
            true_mass_prior_95s[planet_obs.name] = compute_prior_95(planet_obs)

            true_mass_posterior_samples = minimum_mass_samples[:, i] / np.sin(
                inclination_samples
            )
            true_mass_posterior_95s[planet_obs.name] = np.percentile(
                true_mass_posterior_samples, 95
            )

    return true_mass_prior_95s, true_mass_posterior_95s


def build_mass_limits_df(
    catalogue: pd.DataFrame,
    systems: list[str],
    true_mass_prior_95s: dict[str, float],
    true_mass_posterior_95s: dict[str, float],
) -> pd.DataFrame:
    df = catalogue[catalogue["hostname"].isin(systems)].sort_values(
        ["sy_pnum", "hostname", "pl_orbper"], ascending=[False, True, True]
    )

    m95_df = pd.DataFrame(
        columns=[
            "hostname",
            "pl_name",
            "pl_msinie",
            "prior_95",
            "posterior_95",
            "change%",
        ]
    )
    m95_df["hostname"] = df["hostname"]
    m95_df["pl_name"] = df["pl_name"]
    m95_df["pl_msinie"] = df["pl_msinie"]
    m95_df["prior_95"] = m95_df["pl_name"].map(
        lambda x: true_mass_prior_95s.get(x, np.nan)
    )
    m95_df["posterior_95"] = m95_df["pl_name"].map(
        lambda x: true_mass_posterior_95s.get(x, np.nan)
    )
    m95_df["change%"] = (
        100 * (m95_df["posterior_95"] - m95_df["prior_95"]) / m95_df["prior_95"]
    )

    for column in ["pl_msinie", "prior_95", "posterior_95"]:
        m95_df[column] = m95_df[column].map(lambda x: f"{x:.2f}")
    m95_df["change%"] = m95_df["change%"].map(lambda x: f"{x:+.1f}")

    duplicated_hosts = m95_df["hostname"].duplicated()
    m95_df.loc[duplicated_hosts, "pl_name"] = m95_df.loc[
        duplicated_hosts, "pl_name"
    ].map(lambda x: x[-1])

    return m95_df[
        ["hostname", "pl_name", "pl_msinie", "prior_95", "posterior_95", "change%"]
    ]


def format_mass_limits_table(m95_df: pd.DataFrame) -> str:
    latex_df = m95_df.copy()
    latex_df.set_index("hostname", inplace=True)
    latex_df.columns = pd.MultiIndex.from_tuples(
        [
            ("Planet", ""),
            ("$M_{\\rm min}$", "[$M_{\\earth}$]"),
            ("95\\% mass limit [$M_{\\earth}$]", "Prior"),
            ("95\\% mass limit [$M_{\\earth}$]", "Posterior"),
            ("Change", "[\\%]"),
        ]
    )

    m95_latex = (
        latex_df.to_latex(
            index=False,
            column_format="rrrrr",
            multicolumn=True,
            multicolumn_format="c",
        )
        .replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )

    for planet in m95_df["pl_name"].unique():
        if len(planet) > 1 and planet != "HD 158259 b":
            m95_latex = m95_latex.replace(planet, f"\\hline\n{planet}")

    return (
        m95_latex.replace("YZ Cet", "YZ Ceti")
        .replace("Barnard", "Barnard's Star")
        .replace("-", "$-$")
    )


def main() -> None:
    output_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT_DIR / "tex" / "tables" / "masses95.tex"
    )

    catalogue = load_catalogue()
    systems = [
        "HD 158259",
        "Barnard's star",
        "HD 215152",
        "HD 184010",
        "HD 28471",
        "YZ Cet",
    ]
    results_dir = ROOT_DIR / "results" / "mcmc_results"

    true_mass_prior_95s, true_mass_posterior_95s = collect_mass_limit_maps(
        catalogue, systems, results_dir, burn_in=2000
    )

    m95_df = build_mass_limits_df(
        catalogue, systems, true_mass_prior_95s, true_mass_posterior_95s
    )
    m95_df.to_csv(ROOT_DIR / "tex" / "tables" / "masses95.csv", index=False)

    m95_latex = format_mass_limits_table(m95_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        file.write(m95_latex)

    print(f"Saved mass-limits table to {output_path}")


if __name__ == "__main__":
    main()
