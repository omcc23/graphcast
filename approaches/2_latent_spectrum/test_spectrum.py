"""Checks the band edges, Parseval-style coefficient sum, and entropy."""

import numpy as np

from spectrum import (
    band_slices,
    characteristic_degrees,
    entropy_from_band_energy,
    node_nyquist_lmax,
    power_per_degree,
)


def test_characteristic_degrees():
    degrees = characteristic_degrees(5)
    assert list(degrees) == [4, 8, 16, 32, 64, 128]


def test_nyquist_of_small_mesh():
    # 5 refinements of an icosahedron: 10 * 4**5 + 2 nodes.
    n_nodes = 10 * 4**5 + 2
    assert n_nodes == 10242
    assert node_nyquist_lmax(n_nodes) == 100


def test_bands_truncate_at_nyquist():
    bands = band_slices(mesh_size=5, lmax=100)
    assert [(level, start, stop) for level, start, stop, _ in bands] == [
        (0, 1, 4),
        (1, 5, 8),
        (2, 9, 16),
        (3, 17, 32),
        (4, 33, 64),
        (5, 65, 100),
    ]
    assert bands[-1][3] is True
    assert all(partial is False for _, _, _, partial in bands[:-1])


def test_power_per_degree_one_mode():
    cilm = np.zeros((2, 5, 5))
    cilm[0, 3, 2] = 2.0
    cilm[1, 3, 1] = 1.0
    energy = power_per_degree(cilm)
    assert energy.shape == (5,)
    assert energy[3] == 4.0 + 1.0
    assert np.sum(energy) == 5.0


def test_entropy_of_equal_bands():
    summary = entropy_from_band_energy(np.ones(4))
    assert abs(summary["S_H_bits"] - 2.0) < 1e-12
    assert abs(summary["S_H_over_S_max"] - 1.0) < 1e-12


def test_entropy_of_one_band():
    summary = entropy_from_band_energy(np.array([0.0, 3.0, 0.0]))
    assert summary["S_H_bits"] == 0.0
    assert summary["dominant_index"] == 1
    assert summary["dominant_fraction"] == 1.0


if __name__ == "__main__":
    test_characteristic_degrees()
    test_nyquist_of_small_mesh()
    test_bands_truncate_at_nyquist()
    test_power_per_degree_one_mode()
    test_entropy_of_equal_bands()
    test_entropy_of_one_band()
    print("spectrum checks passed")
