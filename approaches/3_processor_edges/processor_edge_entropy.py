"""Mean squared processor-edge output of GraphCast Small, by mesh level.

One 6-hour forecast. At each processor step the edge network writes a
512-vector on every multimesh edge, before the residual is added. The energy
of level m is the mean of the squared Euclidean norm of those vectors over
the edges created at that level. The same quantity is reported for the
mesh-edge embedding produced before the processor, which is the encoder.
"""

import dataclasses
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import xarray
from scipy.stats import linregress

from weathernext.utils import casting
from weathernext.utils import checkpoint
from weathernext.utils import data_utils
from weathernext.utils import icosahedral_mesh
from weathernext.weathernext1_graph import graphcast
from weathernext.utils import normalization

PARAMS_NAME = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DATASET_NAME = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
BUCKET = "https://storage.googleapis.com/dm_graphcast/graphcast/"
R_KM = 6371.0

# Result 1, GraphCast Small, mesh-edge encoder after layer norm.
RESULT1_ENCODER = np.array(
    [4155.69459134796, 4132.91781598949, 4098.492850568822,
     3977.1267105128327, 3666.0157883931447, 3343.576796815721],
    dtype=np.float64,
)

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"already present: {destination.name}", flush=True)
        return
    print(f"downloading {destination.name}", flush=True)
    urlretrieve(url, destination)
    print(f"  {destination.stat().st_size / 1e6:.1f} MB", flush=True)


def edge_levels_for(mesh_size: int) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=mesh_size
    )
    merged = icosahedral_mesh.merge_meshes(meshes)
    face_levels = np.concatenate(
        [
            np.full(mesh.faces.shape[0], level, dtype=np.int32)
            for level, mesh in enumerate(meshes)
        ]
    )
    # faces_to_edges groups the three directed edges of every face, so the
    # level tags are the face levels repeated three times, not interleaved.
    levels = np.tile(face_levels, 3)
    counts = [(levels == level).sum() for level in range(mesh_size + 1)]
    print(f"edge counts by level: {counts}", flush=True)
    return levels


def edge_features(graph):
    if len(graph.edges) != 1:
        raise RuntimeError(f"expected one edge set, found {list(graph.edges)}")
    features = jnp.squeeze(next(iter(graph.edges.values())).features)
    if features.ndim != 2:
        raise ValueError(f"expected (edges, channels), got {features.shape}")
    return features


def mean_energy(features, levels, n_levels: int):
    if features.shape[0] != levels.shape[0]:
        raise ValueError(
            f"edge count {features.shape[0]} does not match tags {levels.shape[0]}"
        )
    squared = jnp.sum(jnp.square(features.astype(jnp.float32)), axis=-1)
    sums = jnp.zeros(n_levels, dtype=jnp.float32).at[levels].add(squared)
    counts = jnp.zeros(n_levels, dtype=jnp.float32).at[levels].add(jnp.ones_like(squared))
    return sums / counts


def entropy_and_slope(energy: np.ndarray, wavenumber: np.ndarray) -> dict:
    probability = energy / energy.sum()
    spectral = float(-np.sum(probability * np.log(probability)))
    slope, intercept, correlation, *_ = linregress(
        np.log(wavenumber), np.log(energy)
    )
    return {
        "H_nats": spectral,
        "H_n": spectral / float(np.log(len(energy))),
        "slope": float(slope),
        "slope_r2": float(correlation**2),
        "prefactor": float(np.exp(intercept)),
        "m0_share": float(probability[0]),
        "probability": probability,
    }


def main() -> None:
    print("python", sys.executable, flush=True)
    print("JAX devices:", jax.devices(), flush=True)

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    params_path = cache / "graphcast_small.npz"
    data_path = cache / DATASET_NAME
    download(BUCKET + "params/" + quote(PARAMS_NAME), params_path)
    download(BUCKET + "dataset/" + DATASET_NAME, data_path)
    for stats_name in (
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc",
    ):
        download(BUCKET + "stats/" + stats_name, cache / stats_name)

    with params_path.open("rb") as handle:
        ckpt = checkpoint.load(handle, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("model:", ckpt.description, flush=True)
    print(
        f"mesh_size={model_config.mesh_size}  "
        f"latent_size={model_config.latent_size}  "
        f"gnn_msg_steps={model_config.gnn_msg_steps}",
        flush=True,
    )

    levels_np = edge_levels_for(int(model_config.mesh_size))
    n_levels = int(model_config.mesh_size) + 1
    levels = jnp.asarray(levels_np)

    example_batch = xarray.load_dataset(data_path).compute()
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    diffs_stddev_by_level = xarray.load_dataset(
        cache / "diffs_stddev_by_level.nc"
    ).compute()
    mean_by_level = xarray.load_dataset(cache / "mean_by_level.nc").compute()
    stddev_by_level = xarray.load_dataset(cache / "stddev_by_level.nc").compute()

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = graphcast.GraphCast(model_config, task_config)
        mesh_gnn = predictor._mesh_gnn
        collected = []

        def process_and_record(latent_graph_0, processor_networks):
            latent_graph = latent_graph_0
            # Mesh-edge encoder embedding, before any processor step.
            collected.append(mean_energy(edge_features(latent_graph), levels, n_levels))
            for _ in range(mesh_gnn._num_processor_repetitions):
                for processor_network in processor_networks:
                    proposed = processor_network(latent_graph)
                    collected.append(
                        mean_energy(edge_features(proposed), levels, n_levels)
                    )
                    latent_graph = mesh_gnn._process_step(
                        processor_network, latent_graph
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
        return prediction, jnp.stack(collected)

    print("compiling and running GraphCast Small...", flush=True)
    (prediction, energies), _next_state = jax.jit(run_forward.apply)(
        params,
        state,
        jax.random.PRNGKey(0),
        inputs,
        targets * np.nan,
        forcings,
    )
    energies = np.asarray(jax.device_get(energies), dtype=np.float64)
    temperature = np.asarray(jax.device_get(prediction["2m_temperature"].data))
    finite = float(np.isfinite(temperature).mean())
    print(
        f"prediction 2m_temperature shape={temperature.shape} "
        f"mean={float(np.nanmean(temperature)):.3f} finite={finite:.3f}",
        flush=True,
    )
    if finite < 0.99:
        raise RuntimeError("Prediction is not finite. The forward pass did not run.")

    # One encoder row, then one row per unshared processor step.
    # GraphCast uses a single pass over those steps.
    expected = 1 + int(model_config.gnn_msg_steps)
    print(f"saved edge-energy rows: {energies.shape[0]} (expected {expected})", flush=True)
    if energies.shape != (expected, n_levels):
        raise RuntimeError(
            f"Expected energies {(expected, n_levels)}, got {energies.shape}."
        )

    length_km = np.array(
        [np.pi * R_KM / (2 * 2**level) for level in range(n_levels)]
    )
    wavenumber = 1.0 / length_km
    names = ["encoder"] + [f"processor_{step}" for step in range(1, expected)]

    encoder_ratio = energies[0] / RESULT1_ENCODER
    print(
        "encoder energy / Result 1 after layer norm: "
        + " ".join(f"{value:.4f}" for value in encoder_ratio),
        flush=True,
    )

    rows = []
    for index, name in enumerate(names):
        summary = entropy_and_slope(energies[index], wavenumber)
        row = {
            "step_index": index,
            "step_name": name,
            "H_nats": summary["H_nats"],
            "H_n": summary["H_n"],
            "slope": summary["slope"],
            "slope_r2": summary["slope_r2"],
            "prefactor": summary["prefactor"],
            "m0_share": summary["m0_share"],
        }
        for level, energy in enumerate(energies[index]):
            row[f"E_M{level}"] = float(energy)
        rows.append(row)
        print(
            f"{name}: H={summary['H_nats']:.4f} nats  "
            f"H_n={summary['H_n']:.3f}  "
            f"slope={summary['slope']:.3f}  "
            f"R2={summary['slope_r2']:.3f}  "
            f"M0={100 * summary['m0_share']:.1f}%",
            flush=True,
        )

    import csv

    table_path = OUT / "graphcast_small_processor_edge_entropy.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {table_path}", flush=True)

    shutil.rmtree(cache, ignore_errors=True)
    print("removed download cache", flush=True)


if __name__ == "__main__":
    main()
