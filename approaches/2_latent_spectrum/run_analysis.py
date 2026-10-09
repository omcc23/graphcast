"""Spherical-harmonic energy and hydrodynamic entropy of a mesh latent field.

E(l) is the power of the latent node field at spherical-harmonic degree l,
summed over order m and over latent channels. Octave bands group those degrees
onto the GraphCast mesh levels. Hydrodynamic entropy is the Shannon entropy of
that band distribution, in bits.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def characteristic_degrees(mesh_size: int) -> np.ndarray:
    """Spherical-harmonic degree matched to each mesh level.

    Mesh level m has edge length L(m) = pi R / (2 * 2^m). A spherical harmonic
    of degree l has wavelength 2 pi R / l. Setting those equal gives l(m) = 2^(m+2).
    """
    return 2 ** (np.arange(mesh_size + 1) + 2)


def node_nyquist_lmax(n_nodes: int) -> int:
    """Largest degree a set of mesh nodes can support.

    There are (lmax + 1)^2 real spherical-harmonic coefficients up to lmax.
    """
    return int(np.floor(np.sqrt(n_nodes) - 1))


def power_per_degree(cilm: np.ndarray) -> np.ndarray:
    """Sum of squared orthonormal coefficients at each degree.

    cilm has shape (2, lmax + 1, lmax + 1): index 0 is cosine, index 1 is sine.
    """
    lmax = cilm.shape[1] - 1
    energy = np.empty(lmax + 1, dtype=np.float64)
    energy[0] = cilm[0, 0, 0] ** 2
    for degree in range(1, lmax + 1):
        energy[degree] = (
            cilm[0, degree, 0] ** 2
            + np.sum(cilm[0, degree, 1 : degree + 1] ** 2)
            + np.sum(cilm[1, degree, 1 : degree + 1] ** 2)
        )
    return energy


def band_slices(mesh_size: int, lmax: int) -> list[tuple[int, int, int, bool]]:
    """Octave bands aligned with mesh levels.

    Each item is (level, l_start, l_stop, partial). Degrees run from l_start
    through l_stop inclusive. l = 0, the spatial mean, is left out. partial is
    true when the mesh level's characteristic degree sits above lmax, so the
    band is cut off at the node Nyquist.
    """
    chars = characteristic_degrees(mesh_size)
    lower = 1
    bands = []
    for level, char in enumerate(chars):
        upper = min(int(char), lmax)
        if lower > lmax:
            break
        if lower <= upper:
            bands.append((level, lower, upper, int(char) > lmax))
            lower = upper + 1
    return bands


def entropy_from_band_energy(band_energy: np.ndarray) -> dict[str, float]:
    """Shannon entropy in bits of a band-energy distribution."""
    total = float(np.sum(band_energy))
    if total <= 0:
        raise ValueError("Band energy sums to zero.")
    probability = band_energy / total
    positive = probability > 0
    entropy = float(
        -np.sum(probability[positive] * np.log2(probability[positive]))
    )
    n_bands = int(band_energy.shape[0])
    entropy_max = float(np.log2(n_bands))
    dominant = int(np.argmax(probability))
    return {
        "S_H_bits": entropy,
        "S_max_bits": entropy_max,
        "S_H_over_S_max": entropy / entropy_max,
        "dominant_index": dominant,
        "dominant_fraction": float(probability[dominant]),
    }


def degree_slope(degree_energy: np.ndarray, l_min: int = 2) -> tuple[float, float]:
    """Log-log slope of E(l) and its coefficient of determination."""
    degrees = np.arange(l_min, degree_energy.shape[0])
    energy = degree_energy[l_min:]
    keep = energy > 0
    log_l = np.log(degrees[keep].astype(np.float64))
    log_e = np.log(energy[keep].astype(np.float64))
    slope, intercept = np.polyfit(log_l, log_e, 1)
    fitted = slope * log_l + intercept
    residual = np.sum((log_e - fitted) ** 2)
    total = np.sum((log_e - np.mean(log_e)) ** 2)
    r_squared = float(1.0 - residual / total) if total > 0 else float("nan")
    return float(slope), r_squared


def mesh_to_gaussian_grid(
    values: np.ndarray,
    vertices: np.ndarray,
    lmax: int,
    lat_lon_to_xyz,
    k: int = 8,
) -> np.ndarray:
    """Interpolate node features onto the Gauss-Legendre grid used by the SHT.

    lat_lon_to_xyz(lat_deg, lon_deg) must return (x, y, z) on the unit sphere
    in the same frame as `vertices`.
    """
    import pyshtools

    lat, lon = pyshtools.expand.GLQGridCoord(lmax)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y, z = lat_lon_to_xyz(lat_grid, lon_grid)
    grid_xyz = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=-1)
    grid_xyz = grid_xyz.reshape(-1, 3)

    tree = cKDTree(np.asarray(vertices, dtype=np.float64))
    distance, index = tree.query(grid_xyz, k=k)
    distance = np.maximum(distance, 1e-8)
    weight = 1.0 / distance
    weight /= weight.sum(axis=1, keepdims=True)

    neighbours = np.asarray(values, dtype=np.float64)[index]
    interpolated = np.einsum("nk,nkc->nc", weight, neighbours)
    nlat, nlon = lat_grid.shape
    return interpolated.reshape(nlat, nlon, values.shape[-1])


def spectrum_of_grid(grid: np.ndarray) -> np.ndarray:
    """Orthonormal power spectrum of a lat-lon grid, summed over channels.

    grid has shape (nlat, nlon, channels) and must be the GLQ grid for its lmax.
    """
    import pyshtools

    lmax = grid.shape[0] - 1
    zeros, weights = pyshtools.expand.SHGLQ(lmax)
    total = np.zeros(lmax + 1, dtype=np.float64)
    for channel in range(grid.shape[-1]):
        coefficients = pyshtools.expand.SHExpandGLQ(
            grid[:, :, channel], weights, zeros, norm=4
        )
        total += power_per_degree(np.asarray(coefficients, dtype=np.float64))
    return total


def summarize_spectrum(
    degree_energy: np.ndarray,
    mesh_size: int,
) -> dict:
    """Band energies, entropy, and the per-degree slope."""
    lmax = int(degree_energy.shape[0] - 1)
    bands = band_slices(mesh_size, lmax)
    band_energy = np.array(
        [float(np.sum(degree_energy[start : stop + 1])) for _, start, stop, _ in bands],
        dtype=np.float64,
    )
    summary = entropy_from_band_energy(band_energy)
    slope, r_squared = degree_slope(degree_energy)
    summary.update(
        {
            "slope": slope,
            "slope_r2": r_squared,
            "lmax": lmax,
            "band_levels": [level for level, _, _, _ in bands],
            "band_l_start": [start for _, start, _, _ in bands],
            "band_l_stop": [stop for _, _, stop, _ in bands],
            "band_partial": [partial for _, _, _, partial in bands],
            "band_energy": band_energy,
            "l0_power": float(degree_energy[0]),
            "resolved_power": float(np.sum(degree_energy[1:])),
        }
    )
    return summary


import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_search_roots = [Path("/kaggle/working"), Path.cwd()]
if "__file__" in globals():
    _search_roots.append(Path(__file__).resolve().parent)
for candidate in _search_roots:
    if (candidate / "spectrum.py").is_file():
        sys.path.insert(0, str(candidate))
        break

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import xarray
from google.cloud import storage

from weathernext.utils import casting
from weathernext.utils import checkpoint
from weathernext.utils import data_utils
from weathernext.utils import icosahedral_mesh
from weathernext.utils import model_utils
from weathernext.utils import normalization
from weathernext.weathernext1_graph import graphcast


PARAMS_NAME = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DATASET_NAME = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
BUCKET_PREFIX = "graphcast/"

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def download_blob(bucket, blob_name, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"already present: {destination.name}")
        return
    print(f"downloading {blob_name}")
    bucket.blob(blob_name).download_to_filename(str(destination))
    print(f"  {destination.stat().st_size / 1e6:.1f} MB")


def node_matrix(array, n_nodes: int) -> np.ndarray:
    """Return latent features as (n_nodes, channels)."""
    array = np.asarray(jax.device_get(array), dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a node-by-channel matrix, got {array.shape}")
    if array.shape[0] == n_nodes:
        return array
    if array.shape[1] == n_nodes:
        return array.T
    raise ValueError(f"No axis of {array.shape} matches {n_nodes} mesh nodes")


def main() -> None:
    import pyshtools

    print("python", sys.executable)
    print("pyshtools", getattr(pyshtools, "__version__", "unknown"), pyshtools.__file__)
    print("JAX devices:", jax.devices())

    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.bucket("dm_graphcast")
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    params_path = cache / "graphcast_small.npz"
    data_path = cache / DATASET_NAME
    download_blob(bucket, BUCKET_PREFIX + "params/" + PARAMS_NAME, params_path)
    download_blob(bucket, BUCKET_PREFIX + "dataset/" + DATASET_NAME, data_path)
    for stats_name in (
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc",
    ):
        download_blob(bucket, BUCKET_PREFIX + "stats/" + stats_name, cache / stats_name)

    with params_path.open("rb") as handle:
        ckpt = checkpoint.load(handle, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("model:", ckpt.description)
    print(
        f"mesh_size={model_config.mesh_size}  "
        f"latent_size={model_config.latent_size}  "
        f"gnn_msg_steps={model_config.gnn_msg_steps}"
    )

    example_batch = xarray.load_dataset(data_path).compute()
    print("example batch:", dict(example_batch.sizes))
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    print("inputs", dict(inputs.sizes))
    print("targets", dict(targets.sizes))

    diffs_stddev_by_level = xarray.load_dataset(cache / "diffs_stddev_by_level.nc").compute()
    mean_by_level = xarray.load_dataset(cache / "mean_by_level.nc").compute()
    stddev_by_level = xarray.load_dataset(cache / "stddev_by_level.nc").compute()

    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=model_config.mesh_size
    )
    vertices = np.asarray(meshes[-1].vertices, dtype=np.float64)
    n_nodes = vertices.shape[0]
    lmax = node_nyquist_lmax(n_nodes)
    print(f"mesh nodes={n_nodes}  spherical-harmonic lmax={lmax}")

    # Confirm the geographic xyz frame used for the Gauss grid matches the mesh.
    lat, lon = model_utils.cartesian_to_lat_lon(
        vertices[:, 0], vertices[:, 1], vertices[:, 2]
    )
    rebuilt = np.stack(model_utils.lat_lon_to_cartesian(lat, lon), axis=-1)
    roundtrip = float(np.max(np.linalg.norm(rebuilt - vertices, axis=-1)))
    print(f"lat/lon round-trip error: {roundtrip:.3e}")
    if roundtrip > 1e-5:
        raise RuntimeError("Mesh vertex frame does not match lat/lon conversion.")

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = graphcast.GraphCast(model_config, task_config)
        mesh_gnn = predictor._mesh_gnn
        collected = []

        def process_and_record(latent_graph_0, processor_networks):
            latent_graph = latent_graph_0
            collected.append(
                jnp.asarray(
                    latent_graph.nodes["mesh_nodes"].features, dtype=jnp.float32
                )
            )
            for _ in range(mesh_gnn._num_processor_repetitions):
                for processor_network in processor_networks:
                    latent_graph = mesh_gnn._process_step(
                        processor_network, latent_graph
                    )
                    collected.append(
                        jnp.asarray(
                            latent_graph.nodes["mesh_nodes"].features,
                            dtype=jnp.float32,
                        )
                    )
            return latent_graph

        object.__setattr__(mesh_gnn, "_process", process_and_record)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        prediction = predictor(
            inputs, targets_template=targets_template, forcings=forcings
        )
        return prediction, collected

    print("compiling and running GraphCast Small...")
    (prediction, latents), _next_state = jax.jit(run_forward.apply)(
        params,
        state,
        jax.random.PRNGKey(0),
        inputs,
        targets * np.nan,
        forcings,
    )

    temperature = np.asarray(jax.device_get(prediction["2m_temperature"].data))
    finite = float(np.isfinite(temperature).mean())
    print(
        f"prediction 2m_temperature shape={temperature.shape} "
        f"mean={float(np.nanmean(temperature)):.3f} finite={finite:.3f}"
    )
    if finite < 0.99:
        raise RuntimeError("Prediction is not finite. The forward pass did not run.")

    expected_steps = 1 + int(model_config.gnn_msg_steps)
    print(f"saved latent fields: {len(latents)} (expected {expected_steps})")
    if len(latents) != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} latent fields, saved {len(latents)}."
        )

    step_names = ["encoder"] + [
        f"processor_{step}" for step in range(1, len(latents))
    ]
    rows = []
    spectra = []
    for step_index, (name, latent) in enumerate(zip(step_names, latents)):
        nodes = node_matrix(latent, n_nodes)
        print(f"{name}: latent {nodes.shape}  std {nodes.std():.4f}")
        grid = mesh_to_gaussian_grid(
            nodes, vertices, lmax, model_utils.lat_lon_to_cartesian
        )
        degree_energy = spectrum_of_grid(grid)
        summary = summarize_spectrum(degree_energy, model_config.mesh_size)
        spectra.append(degree_energy)
        partial_levels = [
            level
            for level, partial in zip(summary["band_levels"], summary["band_partial"])
            if partial
        ]
        row = {
            "step_index": step_index,
            "step_name": name,
            "S_H_bits": summary["S_H_bits"],
            "S_max_bits": summary["S_max_bits"],
            "S_H_over_S_max": summary["S_H_over_S_max"],
            "dominant_level": summary["band_levels"][summary["dominant_index"]],
            "dominant_fraction": summary["dominant_fraction"],
            "slope": summary["slope"],
            "slope_r2": summary["slope_r2"],
            "lmax": summary["lmax"],
            "l0_fraction": summary["l0_power"]
            / (summary["l0_power"] + summary["resolved_power"]),
            "partial_levels": ",".join(str(level) for level in partial_levels),
        }
        for level, energy in zip(summary["band_levels"], summary["band_energy"]):
            row[f"E_M{level}"] = float(energy)
        rows.append(row)
        print(
            f"  S_H={row['S_H_bits']:.4f} bits  "
            f"S_H/S_max={row['S_H_over_S_max']:.3f}  "
            f"slope={row['slope']:.3f}  "
            f"dominant=M{row['dominant_level']} ({100 * row['dominant_fraction']:.1f}%)"
        )

    import pandas as pd

    table = pd.DataFrame(rows)
    table_path = OUT / "graphcast_small_latent_entropy.csv"
    table.to_csv(table_path, index=False)
    np.savez(
        OUT / "graphcast_small_degree_spectra.npz",
        degree_energy=np.stack(spectra, axis=0),
        step_names=np.array(step_names),
    )
    print(table.to_string(index=False))
    print(f"wrote {table_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(table["step_index"], table["S_H_bits"], marker="o", color="#1f4e79")
    axes[0].set_xticks(table["step_index"])
    axes[0].set_xticklabels(table["step_name"], rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("S_H (bits)")
    axes[0].set_title("Hydrodynamic entropy of the mesh latent")
    axes[0].axhline(table["S_max_bits"].iloc[0], color="#9a3412", ls="--", lw=1)
    axes[0].text(
        0.02,
        table["S_max_bits"].iloc[0],
        "  equal energy across bands",
        color="#9a3412",
        va="bottom",
        fontsize=8,
    )

    degrees = np.arange(1, lmax + 1)
    for index, label in ((0, "encoder"), (len(spectra) - 1, step_names[-1])):
        axes[1].loglog(degrees, spectra[index][1:], label=label)
    anchor_degree = 8
    anchor = float(spectra[-1][anchor_degree])
    if anchor > 0:
        reference = anchor * (degrees / anchor_degree) ** (-5.0 / 3.0)
        axes[1].loglog(
            degrees, reference, color="0.45", ls="--", label="slope -5/3"
        )
    axes[1].set_xlabel("spherical-harmonic degree l")
    axes[1].set_ylabel("E(l)")
    axes[1].set_title("Latent power spectrum")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure_path = OUT / "graphcast_small_latent_entropy.png"
    figure.savefig(figure_path, dpi=140)
    print(f"wrote {figure_path}")


main()
