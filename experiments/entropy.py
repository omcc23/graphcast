"""Hydrodynamic entropy of a spatial field.

The definition is the one in Verma and Chatterjee, Phys. Rev. Fluids 7, 114608
(2022) and Verma, Stepanov, and Delache, Phys. Rev. E 110, 055106 (2024):

    E(k) = sum_c |f_hat_c(k)|^2
    p_k  = E(k) / sum_k E(k)
    S_H  = -sum_k p_k log2(p_k)

S_tilde = S_H / log2(M) removes the grid-size ceiling. The per-channel spatial
mean is removed before the transform so a ReLU offset does not dominate k = 0.
A Hann window is applied because activation maps are not periodic.
"""

from __future__ import annotations

import math

import torch


def hann2d(height: int, width: int, device, dtype) -> torch.Tensor:
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def _as_batch(field: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if field.ndim == 3:
        return field.unsqueeze(0), True
    if field.ndim != 4:
        raise ValueError(f"expected (C,H,W) or (B,C,H,W), got {tuple(field.shape)}")
    return field, False


def prepare_field(field: torch.Tensor, remove_mean: bool = True, window: bool = True):
    """Return the field ready for the FFT, and the DC energy fraction of the raw field.

    field: (B, C, H, W)
    dc_fraction: (B,) energy share of the spatial mean, before it is removed.
    """
    spatial_mean = field.mean(dim=(-2, -1), keepdim=True)
    height, width = field.shape[-2:]
    mean_energy = spatial_mean.flatten(1).pow(2).sum(dim=1) * height * width
    total_energy = field.flatten(1).pow(2).sum(dim=1).clamp_min(1e-12)
    dc_fraction = (mean_energy / total_energy).clamp(0, 1)

    prepared = field - spatial_mean if remove_mean else field
    if window:
        prepared = prepared * hann2d(height, width, field.device, field.dtype)
    return prepared, dc_fraction


def modal_energy(field: torch.Tensor) -> torch.Tensor:
    """Sum |FFT|^2 over channels. field is (B, C, H, W). Returns (B, H, W)."""
    spectrum = torch.fft.fft2(field, norm="ortho")
    return spectrum.abs().pow(2).sum(dim=1)


def entropy_from_energy(energy: torch.Tensor, eps: float = 1e-12):
    """Shannon entropy of a non-negative modal energy map.

    energy: (B, H, W)
    Returns S_H (B,), S_tilde (B,), and M.
    A map with no energy is defined as S_H = 0.
    """
    total = energy.sum(dim=(-2, -1))
    safe_total = total.clamp_min(eps)
    prob = energy / safe_total[:, None, None]
    entropy = -(prob * torch.log2(prob.clamp_min(eps))).sum(dim=(-2, -1))
    entropy = torch.where(total < eps, torch.zeros_like(entropy), entropy)
    modes = energy.shape[-1] * energy.shape[-2]
    return entropy, entropy / math.log2(modes), modes


def hydrodynamic_entropy(field: torch.Tensor, remove_mean: bool = True, window: bool = True):
    """Per-example hydrodynamic entropy of a spatial activation or image.

    Returns a dict with s_h, s_tilde, dc_fraction, centroid, M.
    s_h and s_tilde have shape (B,). Gradients flow through s_h and s_tilde.
    """
    batch, squeezed = _as_batch(field)
    prepared, dc_fraction = prepare_field(batch, remove_mean=remove_mean, window=window)
    energy = modal_energy(prepared)
    s_h, s_tilde, modes = entropy_from_energy(energy)
    centroid = spectral_centroid(energy)
    if squeezed:
        s_h, s_tilde, dc_fraction, centroid = (t.squeeze(0) for t in (s_h, s_tilde, dc_fraction, centroid))
    return {
        "s_h": s_h,
        "s_tilde": s_tilde,
        "dc_fraction": dc_fraction,
        "centroid": centroid,
        "M": modes,
        "energy": energy,
    }


def spectral_centroid(energy: torch.Tensor) -> torch.Tensor:
    """Mean radial frequency of E(k), in cycles per pixel. energy is (B, H, W)."""
    height, width = energy.shape[-2:]
    ky = torch.fft.fftfreq(height, device=energy.device, dtype=energy.dtype)
    kx = torch.fft.fftfreq(width, device=energy.device, dtype=energy.dtype)
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = torch.sqrt(yy.pow(2) + xx.pow(2))
    total = energy.sum(dim=(-2, -1)).clamp_min(1e-12)
    return (energy * radius).sum(dim=(-2, -1)) / total


def _mask_from_index(energy: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    chosen = index[:count]
    flat = torch.zeros(energy.numel(), dtype=torch.bool, device=energy.device)
    flat[chosen] = True
    return flat.view_as(energy)


def top_energy_mask(energy: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    """Keep the strongest modes until they hold `fraction` of the energy.

    energy is a single (H, W) map. Returns a boolean mask and the mode count.
    """
    flat = energy.reshape(-1)
    order = torch.argsort(flat, descending=True)
    cumulative = torch.cumsum(flat[order], 0)
    total = flat.sum().clamp_min(1e-12)
    count = int((cumulative / total < fraction).sum().item()) + 1
    count = max(1, min(count, flat.numel()))
    return _mask_from_index(energy, order, count), count


def lowpass_mask(energy: torch.Tensor, count: int) -> torch.Tensor:
    """Keep the `count` modes closest to k = 0."""
    height, width = energy.shape
    ky = torch.fft.fftfreq(height, device=energy.device)
    kx = torch.fft.fftfreq(width, device=energy.device)
    yy, xx = torch.meshgrid(ky, kx, indexing="ij")
    radius = (yy.pow(2) + xx.pow(2)).reshape(-1)
    order = torch.argsort(radius, descending=False)
    count = max(1, min(count, energy.numel()))
    return _mask_from_index(energy, order, count)


def random_energy_mask(energy: torch.Tensor, fraction: float, generator: torch.Generator) -> tuple[torch.Tensor, int]:
    """Random modes whose energy sums to `fraction` of the total."""
    flat = energy.reshape(-1)
    perm = torch.randperm(flat.numel(), generator=generator)
    if perm.device != flat.device:
        perm = perm.to(flat.device)
    cumulative = torch.cumsum(flat[perm], 0)
    total = flat.sum().clamp_min(1e-12)
    count = int((cumulative / total < fraction).sum().item()) + 1
    count = max(1, min(count, flat.numel()))
    return _mask_from_index(energy, perm, count), count


def apply_mode_mask(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero Fourier modes outside `mask`, then put the spatial mean back.

    field: (C, H, W) or (B, C, H, W). mask: (H, W) boolean.
    The mean is restored so the edit does not shift the activation offset.
    """
    batch, squeezed = _as_batch(field)
    spatial_mean = batch.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(batch - spatial_mean)
    edited = spectrum * mask.to(spectrum.dtype)
    restored = torch.fft.ifft2(edited).real + spatial_mean
    return restored.squeeze(0) if squeezed else restored


def run_unit_tests() -> None:
    torch.manual_seed(0)
    height = width = 32

    constant = torch.ones(4, height, width)
    raw = hydrodynamic_entropy(constant, remove_mean=False, window=False)
    assert float(raw["s_h"]) < 1e-5, raw["s_h"]
    assert float(raw["dc_fraction"]) > 0.99
    removed = hydrodynamic_entropy(constant, remove_mean=True, window=False)
    assert float(removed["s_h"]) == 0.0

    row = torch.arange(height, dtype=torch.float32)
    wave = torch.cos(2 * math.pi * 3 * row / height)
    sinusoid = wave[:, None].expand(height, width).unsqueeze(0).contiguous()
    tone = hydrodynamic_entropy(sinusoid, remove_mean=True, window=False)
    assert abs(float(tone["s_h"]) - 1.0) < 1e-3, tone["s_h"]

    noise = torch.randn(8, 64, 64)
    spread = hydrodynamic_entropy(noise, remove_mean=True, window=True)
    assert float(spread["s_tilde"]) > 0.85, spread["s_tilde"]

    energy = modal_energy(prepare_field(noise.unsqueeze(0), remove_mean=True, window=False)[0])[0]
    mask, count = top_energy_mask(energy, 0.9)
    kept = float(energy[mask].sum() / energy.sum())
    assert kept >= 0.9
    assert count < energy.numel()

    low = lowpass_mask(energy, count)
    assert int(low.sum()) == count
    generator = torch.Generator(device="cpu").manual_seed(1)
    random_mask, random_count = random_energy_mask(energy, 0.9, generator)
    assert int(random_mask.sum()) == random_count
    assert random_count > count

    print(
        "unit tests passed",
        f"noise s_tilde={float(spread['s_tilde']):.3f}",
        f"sinusoid s_h={float(tone['s_h']):.3f}",
        f"top90 modes={count}/{energy.numel()}",
    )


if __name__ == "__main__":
    run_unit_tests()
