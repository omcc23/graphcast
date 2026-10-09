"""One panel per checkpoint and stage, with the fitted power law and k^{-5/3}."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "outputs" / "graphcast_weight_entropy.csv"
OUT_PATH = ROOT / "outputs" / "graphcast_weight_powerlaw.png"

MODELS = ("small", "operational", "full")
STAGES = (
    ("E_linear_0", "linear_0"),
    ("E_mlp", "both linear maps"),
    ("E_encoder", "after layer norm"),
)


def main() -> None:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(3, 3, figsize=(11.4, 9.4))
    for row_index, model in enumerate(MODELS):
        model_rows = [row for row in rows if row["model"] == model]
        wavenumber = np.array([float(row["k_per_km"]) for row in model_rows])
        for column_index, (key, stage) in enumerate(STAGES):
            axis = axes[row_index, column_index]
            energy = np.array([float(row[key]) for row in model_rows])
            slope, intercept, *_ = linregress(np.log(wavenumber), np.log(energy))
            fitted = np.exp(intercept) * wavenumber**slope
            kolmogorov = energy[0] * (wavenumber / wavenumber[0]) ** (-5.0 / 3.0)
            axis.loglog(wavenumber, energy, "o", color="C0", label="measured")
            axis.loglog(wavenumber, fitted, "-", color="C3", label=f"fit {slope:.2f}")
            axis.loglog(wavenumber, kolmogorov, "--", color="0.35", label=r"$k^{-5/3}$")
            axis.set_title(f"{model}, {stage}")
            axis.grid(True, which="both", alpha=0.3)
            legend_at = "lower left" if key == "E_encoder" else "upper right"
            axis.legend(fontsize=8, loc=legend_at)
            if row_index == len(MODELS) - 1:
                axis.set_xlabel(r"$k = 1/L$ (km$^{-1}$)")
            if column_index == 0:
                axis.set_ylabel(r"$E(k)$")

    figure.tight_layout()
    figure.savefig(OUT_PATH, dpi=160)
    plt.close(figure)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
