"""
plot_figures.py
===============
Orchestrator script to run the individual figure plotting scripts.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ROOT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = ROOT_DIR / "tex" / "figs"
FIGURE_SCRIPT_DIR = SCRIPT_DIR / "figs"

FIGURE_SCRIPTS = {
    "1": "plot_truevmin.py",
    "2": "plot_posteriors.py",
    "3": "plot_truemasses.py",
    "4": "plot_allinclinations.py",
    "5": "plot_masses_vs_periods.py",
    "6": "plot_hd215152f.py",
    "7": "plot_hillstability.py",
    "c1": "plot_walkers.py",
    "c2": "plot_corner.py",
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for figure_number, script_name in FIGURE_SCRIPTS.items():
        figure_name = f"fig{figure_number}"
        figure_name += f"_{script_name.split('.')[0].replace('plot_', '')}"
        figure_name += ".png"

        figure_path = FIGURES_DIR / figure_name

        print(f"Plotting {figure_name} using {script_name}...")

        subprocess.run(
            [sys.executable, str(FIGURE_SCRIPT_DIR / script_name), figure_path],
            check=True,
        )


if __name__ == "__main__":
    main()
