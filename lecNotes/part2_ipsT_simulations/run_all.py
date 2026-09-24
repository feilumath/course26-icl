#!/usr/bin/env python3
"""Regenerate the Part II demonstrations and the TikZ inputs used by the note."""

from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    for script in ("sensitivity.py", "integrators.py", "dynamics.py"):
        subprocess.run([sys.executable, str(root / script)], check=True, cwd=root)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for name in ("sensitivity.tex", "dynamics.tex"):
        shutil.copyfile(root / "results" / name, figures / name)
    print("All demonstrations regenerated; lecture TikZ inputs refreshed in figures/.")


if __name__ == "__main__":
    main()
