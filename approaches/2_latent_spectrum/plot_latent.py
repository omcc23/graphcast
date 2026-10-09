import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

src = Path(__file__).resolve().parent / "kaggle_out" / "graphcast_small_latent_entropy.csv"
rows = list(csv.DictReader(src.open(encoding="utf-8")))
radius_km = 6371.0
length_km = np.array([np.pi * radius_km / (2 * 2**level) for level in range(6)])
wavenumber = 1.0 / length_km
ln2 = np.log(2)

steps, entropy, normalized, share = [], [], [], []
for row in rows:
    energy = np.array([float(row[f"E_M{level}"]) for level in range(6)])
    slope, *_ = linregress(np.log(wavenumber), np.log(energy))
    steps.append(int(row["step_index"]))
    entropy.append(float(row["S_H_bits"]) * ln2)
    normalized.append(float(row["S_H_over_S_max"]))
    share.append(float(row["dominant_fraction"]))
    print(
        row["step_name"],
        f"H={entropy[-1]:.3f}",
        f"Hn={normalized[-1]:.3f}",
        f"M0={100 * share[-1]:.1f}",
        f"band={slope:.3f}",
        f"deg={float(row['slope']):.3f}",
    )

figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
axes[0].plot(steps, entropy, "o-", color="darkorange")
axes[0].set_xlabel("processor step (0 = encoder)")
axes[0].set_ylabel("H (nats)")
axes[0].set_xticks([0, 4, 8, 12, 16])
axes[0].grid(True, alpha=0.3)

styles = {0: ("o", "encoder"), 8: ("s", "step 8"), 16: ("D", "step 16")}
for row in rows:
    index = int(row["step_index"])
    if index not in styles:
        continue
    energy = np.array([float(row[f"E_M{level}"]) for level in range(6)])
    marker, label = styles[index]
    axes[1].loglog(wavenumber, energy, marker=marker, label=label)
reference = np.array([float(rows[0][f"E_M{level}"]) for level in range(6)])
axes[1].loglog(
    wavenumber,
    reference[0] * (wavenumber / wavenumber[0]) ** (-5.0 / 3.0),
    "--",
    color="0.4",
    label=r"k$^{-5/3}$",
)
axes[1].set_xlabel(r"k = 1/L (km$^{-1}$)")
axes[1].set_ylabel("band energy")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)
figure.tight_layout()
destination = Path(__file__).resolve().parent / "outputs" / "graphcast_latent_spectrum.png"
figure.savefig(destination, dpi=140)
print("wrote", destination)
