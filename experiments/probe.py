"""Forward hooks, depth curves, and Fourier-mode edits."""

from __future__ import annotations

import torch
import torch.nn as nn

from entropy import apply_mode_mask, hydrodynamic_entropy, lowpass_mask, random_energy_mask, top_energy_mask


def _tensor(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output if torch.is_tensor(output) else None


def discover_stages(model: nn.Module, example: torch.Tensor):
    """Depth points in forward order.

    Consecutive convolutional modules at one resolution collapse to the stage
    output. Each module whose class name contains EncoderBlock is kept, so a
    ViT yields one point per block. Token sequences (B, 1 + S*S, D) are
    reshaped to a square grid with the class token set aside.
    """
    seen = []
    hooks = []

    def hook(name, module):
        def _hook(_module, _inputs, output):
            tensor = _tensor(output)
            if tensor is None or tensor.ndim not in (3, 4) or tensor.shape[0] != example.shape[0]:
                return
            if tensor.ndim == 4 and min(tensor.shape[-2:]) >= 4:
                seen.append({"name": name, "module": module, "kind": "map", "spatial": tuple(tensor.shape[-2:])})
            elif tensor.ndim == 3 and "EncoderBlock" in module.__class__.__name__ and tensor.shape[1] > 1:
                tokens = tensor.shape[1] - 1
                side = int(round(tokens ** 0.5))
                if side * side == tokens and side >= 4:
                    seen.append({"name": name, "module": module, "kind": "tokens", "spatial": (side, side)})
        return _hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook(name, module)))
    with torch.no_grad():
        model(example)
    for handle in hooks:
        handle.remove()

    # Consecutive convolutional modules at one resolution collapse to the last
    # of them, which is the stage output. Each transformer block is its own
    # depth point even though the grid size does not change.
    stages = []
    for item in seen:
        if item["kind"] == "tokens":
            stages.append(item)
            continue
        if stages and stages[-1]["kind"] == "map" and stages[-1]["spatial"] == item["spatial"]:
            stages[-1] = item
        else:
            stages.append(item)
    return stages


def _maps_from_output(output, kind: str) -> torch.Tensor:
    tensor = _tensor(output)
    if kind == "map":
        return tensor
    tokens = tensor[:, 1:, :]
    batch, count, dim = tokens.shape
    side = int(count ** 0.5)
    return tokens.transpose(1, 2).reshape(batch, dim, side, side)


def _output_from_maps(original, maps: torch.Tensor, kind: str):
    if kind == "map":
        return maps
    batch, dim, side, _ = maps.shape
    tokens = maps.reshape(batch, dim, side * side).transpose(1, 2)
    return torch.cat([original[:, :1, :], tokens], dim=1)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    """Per-image entropy at every discovered stage, plus logits.

    Returns a list of stage summaries and a list of per-image dicts at each stage.
    """
    model.eval()
    example = next(iter(loader))[0][:1].to(device)
    stages = discover_stages(model, example)
    rows = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        captured = {}
        hooks = []

        def make_hook(stage):
            def _hook(_module, _inputs, output):
                maps = _maps_from_output(output, stage["kind"])
                stats = hydrodynamic_entropy(maps)
                captured[stage["name"]] = {key: stats[key].detach() for key in ("s_h", "s_tilde", "dc_fraction", "centroid")}
                captured[stage["name"]]["norm"] = maps.flatten(1).pow(2).mean(dim=1)
            return _hook

        for stage in stages:
            hooks.append(stage["module"].register_forward_hook(make_hook(stage)))
        logits = model(images)
        for handle in hooks:
            handle.remove()
        prediction = logits.argmax(dim=1)
        probability = torch.softmax(logits, dim=1).max(dim=1).values
        input_stats = hydrodynamic_entropy(images)
        for index in range(images.shape[0]):
            item = {
                "target": int(targets[index]),
                "prediction": int(prediction[index]),
                "correct": bool(prediction[index] == targets[index]),
                "msp": float(probability[index]),
                "input_s_tilde": float(input_stats["s_tilde"][index]),
                "stages": {},
            }
            for stage in stages:
                stats = captured[stage["name"]]
                item["stages"][stage["name"]] = {
                    "spatial": list(stage["spatial"]),
                    "s_h": float(stats["s_h"][index]),
                    "s_tilde": float(stats["s_tilde"][index]),
                    "dc_fraction": float(stats["dc_fraction"][index]),
                    "centroid": float(stats["centroid"][index]),
                    "norm": float(stats["norm"][index]),
                }
            rows.append(item)
    return stages, rows


def summarize_depth(rows):
    """Mean normalized entropy by stage, split by correct and incorrect."""
    if not rows:
        return []
    names = list(rows[0]["stages"])
    summary = []
    for name in names:
        correct = [row["stages"][name]["s_tilde"] for row in rows if row["correct"]]
        wrong = [row["stages"][name]["s_tilde"] for row in rows if not row["correct"]]
        all_values = [row["stages"][name]["s_tilde"] for row in rows]
        gap = (sum(wrong) / len(wrong) if wrong else float("nan")) - (sum(correct) / len(correct) if correct else float("nan"))
        summary.append({
            "name": name,
            "spatial": rows[0]["stages"][name]["spatial"],
            "s_tilde_all": sum(all_values) / len(all_values),
            "s_tilde_correct": sum(correct) / len(correct) if correct else None,
            "s_tilde_wrong": sum(wrong) / len(wrong) if wrong else None,
            "gap_wrong_minus_correct": gap,
            "n_correct": len(correct),
            "n_wrong": len(wrong),
        })
    return summary


def collapse_stage(summary):
    """Stage with the largest entropy gap between wrong and correct predictions."""
    scored = [item for item in summary if item["s_tilde_correct"] is not None and item["s_tilde_wrong"] is not None]
    if not scored:
        return summary[len(summary) // 2]["name"] if summary else None
    return max(scored, key=lambda item: item["gap_wrong_minus_correct"])["name"]


def _mask_for(energy, fraction, mode, generator):
    """Boolean mask and the number of Fourier modes it keeps."""
    if mode == "top":
        return top_energy_mask(energy, fraction)
    if mode == "lowpass":
        _top, count = top_energy_mask(energy, fraction)
        return lowpass_mask(energy, count), count
    if mode == "random":
        return random_energy_mask(energy, fraction, generator)
    raise ValueError(mode)


def _edit_hook(kind, fraction, mode, generator):
    def _hook(_module, _inputs, output):
        tensor = _tensor(output)
        maps = _maps_from_output(tensor, kind)
        edited = []
        for image in maps:
            stats = hydrodynamic_entropy(image, remove_mean=True, window=False)
            energy = stats["energy"][0]
            mask, _count = _mask_for(energy, fraction, mode, generator)
            edited.append(apply_mode_mask(image, mask))
        restored = torch.stack(edited, dim=0)
        return _output_from_maps(tensor, restored, kind)
    return _hook


@torch.no_grad()
def mode_edit_accuracy(model, loader, stage, fraction, mode, device, seed=0):
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    handle = stage["module"].register_forward_hook(_edit_hook(stage["kind"], fraction, mode, generator))
    correct = 0
    total = 0
    mode_fraction_sum = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = model(images).argmax(dim=1)
        correct += int((prediction == targets).sum())
        total += int(targets.shape[0])
    handle.remove()
    # Mode count is image-dependent; report accuracy. A second pass would be
    # needed for the exact kept-mode fraction, computed on one batch below.
    sample = next(iter(loader))[0][:8].to(device)
    fractions = []
    captured = {}

    def capture(_module, _inputs, output):
        maps = _maps_from_output(_tensor(output), stage["kind"])
        captured["maps"] = maps.detach()

    handle = stage["module"].register_forward_hook(capture)
    model(sample)
    handle.remove()
    sample_generator = torch.Generator(device="cpu").manual_seed(seed)
    for image in captured["maps"]:
        energy = hydrodynamic_entropy(image, remove_mean=True, window=False)["energy"][0]
        _mask, count = _mask_for(energy, fraction, mode, sample_generator)
        fractions.append(count / energy.numel())
    mode_fraction_sum = sum(fractions) / len(fractions)
    return {"accuracy": correct / max(total, 1), "mean_mode_fraction": mode_fraction_sum, "n": total}
