"""Weight energy of GraphCast on the mesh lengths the checkpoint actually has.

GraphCast stores one mesh-edge encoder and applies it to every multimesh edge.
There is no separate weight matrix for M0, M1, ... The energy at level m is the
mean squared output of those shared weights on the edges of that level. Levels
finer than model_config.mesh_size are not invented.

H = -sum p_i ln p_i is reported in nats. The wavenumber is k = 1/L, with L the
physical length of the level in kilometres.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.stats import linregress

R_KM = 6371.0
PREFIX = "params:mesh_gnn/~_networks_builder/encoder_edges_mesh"
BUCKET = "https://storage.googleapis.com/dm_graphcast/graphcast/params/"

MODELS = (
    (
        "small",
        "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
    ),
    (
        "operational",
        "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
    ),
    (
        "full",
        "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
    ),
)


def output_dir() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    path = Path(__file__).resolve().parent / "outputs"
    path.mkdir(exist_ok=True)
    return path


def cache_dir() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        path = kaggle / "checkpoints"
        path.mkdir(exist_ok=True)
        return path
    path = Path(__file__).resolve().parents[1] / "_cache"
    path.mkdir(exist_ok=True)
    return path


def load_mesh():
    path = Path(__file__).resolve().parent / "icosahedral_mesh.py"
    spec = importlib.util.spec_from_file_location("icosahedral_mesh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(filename: str) -> Path:
    path = cache_dir() / filename
    if not path.is_file():
        print(f"downloading {filename}")
        urlretrieve(BUCKET + quote(filename), path)
    return path


def edge_geometry(mesh_module, splits: int):
    meshes = mesh_module.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    merged = mesh_module.merge_meshes(meshes)
    senders, receivers = mesh_module.faces_to_edges(merged.faces)
    face_levels = np.concatenate(
        [
            np.full(mesh.faces.shape[0], level, dtype=np.int32)
            for level, mesh in enumerate(meshes)
        ]
    )
    # faces_to_edges groups the three directed edges of every face, so the
    # level tags are the face levels repeated three times, not interleaved.
    edge_levels = np.tile(face_levels, 3)
    difference = merged.vertices[receivers] - merged.vertices[senders]
    length = np.linalg.norm(difference, axis=-1, keepdims=True)
    features = np.concatenate([difference, length], axis=-1).astype(np.float32)
    return edge_levels, features


def encoder_outputs(features: np.ndarray, checkpoint: np.lib.npyio.NpzFile):
    """Return linear_0, the second linear map, and the layer-norm embedding."""
    linear0 = features @ checkpoint[f"{PREFIX}_mlp/~/linear_0:w"]
    linear0 = linear0 + checkpoint[f"{PREFIX}_mlp/~/linear_0:b"]
    hidden = linear0 * expit(linear0)
    pre_norm = hidden @ checkpoint[f"{PREFIX}_mlp/~/linear_1:w"]
    pre_norm = pre_norm + checkpoint[f"{PREFIX}_mlp/~/linear_1:b"]
    mean = pre_norm.mean(axis=-1, keepdims=True)
    variance = pre_norm.var(axis=-1, keepdims=True)
    normalized = (pre_norm - mean) / np.sqrt(variance + 1e-5)
    embedding = normalized * checkpoint[f"{PREFIX}_layer_norm:scale"]
    embedding = embedding + checkpoint[f"{PREFIX}_layer_norm:offset"]
    return linear0, pre_norm, embedding


def mean_energy(values: np.ndarray, edge_levels: np.ndarray, n_levels: int) -> np.ndarray:
    energy = np.empty(n_levels, dtype=np.float64)
    counts = np.empty(n_levels, dtype=np.int64)
    squared = np.sum(values.astype(np.float64) ** 2, axis=-1)
    for level in range(n_levels):
        mask = edge_levels == level
        counts[level] = int(mask.sum())
        energy[level] = float(squared[mask].mean())
    return energy, counts


def entropy(energy: np.ndarray) -> tuple[float, float, np.ndarray]:
    probability = energy / energy.sum()
    spectral = float(-np.sum(probability * np.log(probability)))
    return spectral, spectral / float(np.log(len(energy))), probability


def power_law(energy: np.ndarray, wavenumber: np.ndarray) -> tuple[float, float, float]:
    slope, intercept, correlation, *_ = linregress(np.log(wavenumber), np.log(energy))
    return float(slope), float(np.exp(intercept)), float(correlation**2)


def analyze(name: str, filename: str, mesh_module) -> tuple[list[dict], dict]:
    with np.load(checkpoint_path(filename)) as checkpoint:
        splits = int(checkpoint["model_config:mesh_size"])
        hidden_layers = int(checkpoint["model_config:hidden_layers"])
        if hidden_layers != 1:
            raise RuntimeError(f"{name}: expected one hidden layer, found {hidden_layers}")
        edge_levels, features = edge_geometry(mesh_module, splits)
        outputs = encoder_outputs(features, checkpoint)
        weight_energy = {
            "linear_0": float(np.sum(checkpoint[f"{PREFIX}_mlp/~/linear_0:w"] ** 2)),
            "linear_1": float(np.sum(checkpoint[f"{PREFIX}_mlp/~/linear_1:w"] ** 2)),
        }
    n_levels = splits + 1
    length_km = np.array([np.pi * R_KM / (2 * 2**level) for level in range(n_levels)])
    wavenumber = 1.0 / length_km
    stage_names = ("linear_0", "mlp", "encoder")
    energies = {}
    counts = None
    for stage, values in zip(stage_names, outputs):
        energies[stage], counts = mean_energy(values, edge_levels, n_levels)
        del values

    rows = []
    summary = {
        "model": name,
        "mesh_size": splits,
        "n_levels": n_levels,
        "sum_w2_linear_0": weight_energy["linear_0"],
        "sum_w2_linear_1": weight_energy["linear_1"],
    }
    probabilities = {}
    for stage in stage_names:
        spectral, normalized, probability = entropy(energies[stage])
        slope, prefactor, r2 = power_law(energies[stage], wavenumber)
        probabilities[stage] = probability
        summary[f"H_{stage}"] = spectral
        summary[f"Hn_{stage}"] = normalized
        summary[f"slope_{stage}"] = slope
        summary[f"prefactor_{stage}"] = prefactor
        summary[f"r2_{stage}"] = r2
        summary[f"m0_{stage}"] = float(probability[0])
    for level in range(n_levels):
        row = {
            "model": name,
            "level": level,
            "L_km": float(length_km[level]),
            "k_per_km": float(wavenumber[level]),
            "n_edges": int(counts[level]),
        }
        for stage in stage_names:
            row[f"E_{stage}"] = float(energies[stage][level])
            row[f"p_{stage}"] = float(probabilities[stage][level])
        rows.append(row)
    print_model(summary, rows)
    return rows, summary


def print_model(summary: dict, rows: list[dict]) -> None:
    print(
        f"\n{summary['model']}: mesh_size={summary['mesh_size']} "
        f"levels M0-M{summary['mesh_size']} "
        f"(no finer rung is in this checkpoint)"
    )
    print(
        f"shared sum of squared weights: "
        f"linear_0={summary['sum_w2_linear_0']:.4f}  "
        f"linear_1={summary['sum_w2_linear_1']:.4f}"
    )
    header = f"{'level':<6}{'L_km':>10}{'n_edges':>10}{'E_linear0':>14}{'E_mlp':>14}{'E_encoder':>14}"
    print(header)
    for row in rows:
        print(
            f"M{row['level']:<5}{row['L_km']:10.1f}{row['n_edges']:10d}"
            f"{row['E_linear_0']:14.6e}{row['E_mlp']:14.6e}{row['E_encoder']:14.6e}"
        )
    for stage in ("linear_0", "mlp", "encoder"):
        print(
            f"{stage:<10} H={summary[f'H_{stage}']:.4f} nats  "
            f"H_n={summary[f'Hn_{stage}']:.4f}  "
            f"E(k)= {summary[f'prefactor_{stage}']:.4e} k^{summary[f'slope_{stage}']:.4f}  "
            f"R2={summary[f'r2_{stage}']:.4f}  "
            f"M0={100 * summary[f'm0_{stage}']:.1f}%"
        )


def write_tables(rows: list[dict], summaries: list[dict], destination: Path) -> None:
    level_path = destination / "graphcast_weight_entropy.csv"
    with level_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = destination / "graphcast_weight_entropy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nwrote {level_path}")
    print(f"wrote {summary_path}")


def write_figure(rows: list[dict], destination: Path) -> None:
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])
    figure, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.2), sharey=False)
    if len(models) == 1:
        axes = [axes]
    styles = (
        ("E_linear_0", "o", "linear_0 only"),
        ("E_mlp", "s", "both linear maps"),
        ("E_encoder", "D", "after layer norm"),
    )
    for axis, model in zip(axes, models):
        model_rows = [row for row in rows if row["model"] == model]
        wavenumber = np.array([row["k_per_km"] for row in model_rows])
        for key, marker, label in styles:
            energy = np.array([row[key] for row in model_rows])
            axis.loglog(wavenumber, energy, marker=marker, label=label)
        reference = np.array([row["E_linear_0"] for row in model_rows])
        kolmogorov = reference[0] * (wavenumber / wavenumber[0]) ** (-5.0 / 3.0)
        axis.loglog(wavenumber, kolmogorov, "--", color="0.4", label=r"k$^{-5/3}$")
        axis.set_title(model)
        axis.set_xlabel(r"k = 1/L  (km$^{-1}$)")
        axis.set_ylabel(r"E(k)")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    path = destination / "graphcast_weight_entropy.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    mesh_module = load_mesh()
    rows: list[dict] = []
    summaries: list[dict] = []
    for name, filename in MODELS:
        model_rows, summary = analyze(name, filename, mesh_module)
        rows.extend(model_rows)
        summaries.append(summary)
    destination = output_dir()
    write_tables(rows, summaries, destination)
    write_figure(rows, destination)


if __name__ == "__main__":
    main()
