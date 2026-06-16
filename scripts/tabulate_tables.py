"""
tabulate_tables.py
==================
Orchestrator script to run the individual table-tabulation scripts.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TABLES_DIR = ROOT_DIR / "tex" / "tables"
TABLE_SCRIPT_DIR = SCRIPT_DIR / "tables"

TABLE_SCRIPTS = {
    "systems.tex": "tabulate_systems.py",
    "masses95.tex": "tabulate_masslimits.py",
}


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for table_name, script_name in TABLE_SCRIPTS.items():
        table_path = TABLES_DIR / table_name

        print(f"Tabulating {table_name} using {script_name}...")

        subprocess.run(
            [sys.executable, str(TABLE_SCRIPT_DIR / script_name), str(table_path)],
            check=True,
        )


if __name__ == "__main__":
    main()
