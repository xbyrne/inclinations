"""
plot_corner.py
==============
Plots a corner plot of the MCMC samples for the YZ Ceti system run.
"""

import sys
from pathlib import Path

import numpy as np
from corner import corner

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        print("Usage: python plot_corner.py <figure_path>")
        sys.exit(1)

    results_dir = ROOT_DIR / "results" / "mcmc_results"
    system = "YZ Cet"
    fl = np.load(results_dir / f"{system.lower().replace(' ', '_')}.npz")
    samples = fl["samples"]

    burn_in = 2000
    planets = ["b", "c", "d"]
    parameters = (
        ["$\\cos i$", "$M_*$ [$M_{\\odot}$]"]
        + ["$M_{{{\\rm min}, " + p + "}}$ [M$_{{\\oplus}}$]" for p in planets]
        + ["$P_{" + p + "}$ [d]" for p in planets]
        + ["$e_{" + p + "}$" for p in planets]
        + ["$\\Delta\\omega_{" + p + "}$ [$^\\circ$]" for p in planets[1:]]
        + ["$\\Delta f_{{0," + p + "}}$ [$^\\circ$]" for p in planets[1:]]
    )

    fg = corner(
        samples[burn_in:].reshape(-1, samples.shape[-1]),
        labels=parameters,
        label_kwargs={"fontsize": 18},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fg.savefig(output_path, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
