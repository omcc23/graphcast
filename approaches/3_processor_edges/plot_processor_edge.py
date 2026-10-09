"""Power-law panels for the processor-edge energies."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "outputs" / "graphcast_small_processor_edge_entropy.csv"
OUT_PATH = ROOT / "outputs" / "graphcast_processor_edge_entropy.png"

SHOW = ("encoder", "processor_8", "processor_16")


def main() -> None:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["step_name"]: row for row in rows}
    length_km = np.array([np.pi * 6371.0 / (2 * 2**level) for level in range(6)])
    wavenumber = 1.0 / length_km

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.8))
    for axis, name in zip(axes, SHOW):
        row = by_name[name]
        energy = np.array([float(row[f"E_M{level}"]) for level in range(6)])
        slope = float(row["slope"])
        fitted = float(row["prefactor"]) * wavenumber**slope
        kolmogorov = energy[0] * (wavenumber / wavenumber[0]) ** (-5.0 / 3.0)
        axis.loglog(wavenumber, energy, "o", color="C0", label="measured")
        axis.loglog(wavenumber, fitted, "-", color="C3", label=f"fit {slope:.2f}")
        axis.loglog(wavenumber, kolmogorov, "--", color="0.35", label=r"$k^{-5/3}$")
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel(r"$k = 1/L$ (km$^{-1}$)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"$E(k)$")
    figure.tight_layout()
    figure.savefig(OUT_PATH, dpi=160)
    plt.close(figure)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
