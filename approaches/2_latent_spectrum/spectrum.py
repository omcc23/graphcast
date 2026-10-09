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
