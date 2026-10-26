"""Re-run the whole analysis from the raw workbook.

Usage (from this folder):  python run_all.py
Executes the numbered notebooks in order, in place, with nbconvert, so that
every table in outputs/results/, figure in outputs/figures/ and clean table in data/processed/
is regenerated from data/raw/Observations 2012-2025.xlsx.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))

for nb in NOTEBOOKS:
    print(f"==> executing {nb.name}")
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", "--ExecutePreprocessor.timeout=1800", str(nb)],
                   cwd=nb.parent, check=True)
print("All notebooks executed.")
