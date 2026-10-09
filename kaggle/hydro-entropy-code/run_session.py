"""Kaggle session entry point. Uses every visible GPU, which is two T4s on Kaggle."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import torch


def output_dir() -> Path:
    path = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = Path("/kaggle/working/data") if Path("/kaggle/working").is_dir() else Path("outputs/data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def auroc(scores, labels) -> float:
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = sum(rank for rank, index in enumerate(order, start=1) if labels[index])
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def logistic_scores(features, labels):
    x = torch.tensor(features, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std
    y = torch.tensor(labels, dtype=torch.float32)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(x @ weight + bias).tolist()


def _gpu_worker(gpu: int, task: str, payload: dict, destination: str):
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{gpu}")
    try:
        if task == "session":
            from session_train import train_session
            result = train_session(device, payload)
        elif task == "pretrained":
            result = run_pretrained(device, payload["arch"], payload["limit"])
        elif task == "train":
            from train import train_cifar
            result = train_cifar(
                device,
                Path(payload["out_dir"]),
                Path(payload["data_root"]),
                epochs=payload["epochs"],
                batch_size=payload["batch_size"],
                seed=payload["seed"],
                random_labels=payload["random_labels"],
                regularize=payload["regularize"],
                probe_limit=payload["probe_limit"],
            )
        else:
            raise ValueError(task)
        torch.save({"ok": True, "result": result}, destination)
    except Exception:
        torch.save({"ok": False, "error": traceback.format_exc()}, destination)
        raise


def run_pretrained(device, arch: str, limit: int):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    from probe import evaluate_loader, summarize_depth

    specs = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 32),
        "resnet50_random": (models.resnet50, None, 32),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 32),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, 16),
    }
    builder, weights, batch_size = specs[arch]
    model = builder(weights=weights).to(device).eval()
    transform = (weights or models.ResNet50_Weights.DEFAULT).transforms()
    dataset = datasets.CIFAR100(data_dir(), train=False, download=True, transform=transform)
    loader = DataLoader(Subset(dataset, list(range(limit))), batch_size=batch_size, shuffle=False, num_workers=0)
    _stages, rows = evaluate_loader(model, loader, device)
    # CIFAR labels are not ImageNet classes, so only the unconditional curve is meaningful.
    summary = summarize_depth(rows)
    for item in summary:
        item["s_tilde_correct"] = None
        item["s_tilde_wrong"] = None
        item["gap_wrong_minus_correct"] = None
    print(arch, [(item["name"], round(item["s_tilde_all"], 3), item["spatial"]) for item in summary], flush=True)
    return {"arch": arch, "summary": summary}


def run_trained_eval(device, checkpoint: Path, limit: int):
    from torch.utils.data import DataLoader, Subset

    from probe import collapse_stage, evaluate_loader, mode_edit_accuracy, summarize_depth
    from train import cifar100_loaders, resnet18_cifar100

    _set, _loader, probe_loader = cifar100_loaders(data_dir(), batch_size=64)
    probe_loader = DataLoader(Subset(probe_loader.dataset, list(range(limit))), batch_size=64, shuffle=False, num_workers=0)
    model = resnet18_cifar100().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    stages, rows = evaluate_loader(model, probe_loader, device)
    summary = summarize_depth(rows)
    chosen = collapse_stage(summary)
    edits = []
    stage_by_name = {stage["name"]: stage for stage in stages}
    for stage in stages:
        for fraction in (0.5, 0.7, 0.9, 0.99):
            for mode in ("top", "lowpass", "random"):
                measured = mode_edit_accuracy(model, probe_loader, stage, fraction, mode, device)
                measured.update({"stage": stage["name"], "fraction": fraction, "mode": mode, "is_collapse": stage["name"] == chosen})
                edits.append(measured)
                print("edit", measured, flush=True)
    return {
        "summary": summary,
        "collapse_stage": chosen,
        "rows_collapse": [
            {
                "correct": row["correct"],
                "msp": row["msp"],
                "input_s_tilde": row["input_s_tilde"],
                **row["stages"][chosen],
            }
            for row in rows
        ],
        "edits": edits,
        "stage_names": [stage["name"] for stage in stages],
        "unused": list(stage_by_name),
    }


def confound_report(rows):
    labels = [1 if row["correct"] else 0 for row in rows]
    base_features = [[row["input_s_tilde"], row["norm"], row["centroid"], row["msp"]] for row in rows]
    full_features = [features + [row["s_tilde"]] for features, row in zip(base_features, rows)]
    base_auroc = auroc(logistic_scores(base_features, labels), labels)
    full_auroc = auroc(logistic_scores(full_features, labels), labels)
    entropy_only = auroc([-row["s_tilde"] for row in rows], labels)
    softmax_only = auroc([row["msp"] for row in rows], labels)
    return {
        "auroc_without_entropy": base_auroc,
        "auroc_with_entropy": full_auroc,
        "gain": full_auroc - base_auroc,
        "auroc_entropy_alone": entropy_only,
        "auroc_softmax_alone": softmax_only,
        "n": len(rows),
        "accuracy": sum(labels) / max(len(labels), 1),
    }


def save_plot(path: Path, curves: dict, xlabel: str, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in curves.items():
        ax.plot(range(len(values)), values, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _parallel(jobs: list[tuple[int, str, dict]], out: Path):
    """One process per GPU. `jobs` is a single wave that fits on the visible devices."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    processes = []
    destinations = []
    for gpu, task, payload in jobs:
        tag = payload.get("tag", task)
        destination = out / f"job_{tag}.pt"
        destinations.append(destination)
        process = ctx.Process(target=_gpu_worker, args=(gpu, task, payload, str(destination)))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    results = []
    for destination, process in zip(destinations, processes):
        if not destination.is_file():
            results.append({"ok": False, "exitcode": process.exitcode, "destination": str(destination)})
            continue
        loaded = torch.load(destination, map_location="cpu", weights_only=False)
        loaded["exitcode"] = process.exitcode
        results.append(loaded)
    return results


def _run_waves(jobs: list[tuple[str, dict]], out: Path):
    """Pair jobs across the visible GPUs. One GPU runs the queue one job at a time."""
    gpus = list(range(torch.cuda.device_count() or 1))
    collected = []
    width = len(gpus)
    for start in range(0, len(jobs), width):
        wave = []
        for offset, (task, payload) in enumerate(jobs[start:start + width]):
            wave.append((gpus[offset], task, payload))
        print("wave", [(gpu, task, payload.get("tag")) for gpu, task, payload in wave], flush=True)
        collected.extend(_parallel(wave, out))
    return collected


def _convert(value):
    if isinstance(value, dict):
        return {key: _convert(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write(out: Path, report: dict):
    (out / "results.json").write_text(json.dumps(_convert(report), indent=2), encoding="utf-8")
    print("wrote", out / "results.json", flush=True)


def _train_payload(out, epochs, batch_size, probe_limit, seed, random_labels, regularize, tag):
    return {
        "out_dir": str(out),
        "data_root": str(data_dir()),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "random_labels": random_labels,
        "regularize": regularize,
        "probe_limit": probe_limit,
        "tag": tag,
    }


# Stop launching a new wave once the predicted finish crosses this. A Kaggle
# GPU session is killed at 12 hours; the current wave is always allowed to end.
SESSION_LIMIT_S = 10.5 * 3600


def _session_job(out, tag, condition, legend, seed, epochs, subset, **overrides):
    payload = {
        "tag": tag,
        "condition": condition,
        "legend": legend,
        "seed": seed,
        "epochs": epochs,
        "subset_size": subset,
        "batch_size": 256,
        "random_labels": False,
        "label_noise": 0.0,
        "augment": True,
        "weight_decay": 5e-4,
        "init": "kaiming",
        "lr": 0.1,
        "train_probe": 4000,
        "out_dir": str(out),
        "data_root": str(data_dir()),
    }
    payload.update(overrides)
    return ("session", payload)


def _session_queue(out):
    """Headline contrast first, then the regularization controls, then extras.

    real_reg: real labels, crop/flip, weight decay.
    random_plain: one fixed random labeling, no crop/flip, no weight decay,
    so the net can memorize. real_plain and random_reg separate the label
    change from the regularization change.
    """
    jobs = []
    for seed in (0, 1, 2):
        jobs.append(_session_job(out, f"real_reg_s{seed}", "real_reg", "real, aug+decay", seed, 40, 10_000))
        jobs.append(_session_job(
            out, f"random_plain_s{seed}", "random_plain", "random, no aug", seed, 40, 10_000,
            random_labels=True, augment=False, weight_decay=0.0,
        ))
    for seed in (0, 1):
        jobs.append(_session_job(
            out, f"real_plain_s{seed}", "real_plain", "real, no aug", seed, 40, 10_000,
            augment=False, weight_decay=0.0,
        ))
        jobs.append(_session_job(
            out, f"random_reg_s{seed}", "random_reg", "random, aug+decay", seed, 40, 10_000,
            random_labels=True,
        ))
    for seed, (scheme, lr) in enumerate((("xavier", 0.1), ("orthogonal", 0.1), ("normal", 0.02)), start=0):
        jobs.append(_session_job(
            out, f"init_{scheme}", f"init_{scheme}", f"init {scheme}", seed, 25, 10_000,
            init=scheme, lr=lr,
        ))
    for noise in (0.2, 0.4):
        jobs.append(_session_job(
            out, f"noise{int(noise * 100)}_s0", f"noise{int(noise * 100)}", f"label noise {int(noise * 100)}%", 0, 30, 10_000,
            label_noise=noise,
        ))
    for seed in (0, 1):
        jobs.append(_session_job(out, f"full_real_reg_s{seed}", "full_real_reg", "full data, real", seed, 24, 50_000))
        jobs.append(_session_job(
            out, f"full_random_plain_s{seed}", "full_random_plain", "full data, random", seed, 24, 50_000,
            random_labels=True, augment=False, weight_decay=0.0,
        ))
    return jobs


def _seconds_per_image(jobs) -> float | None:
    images = 0
    seconds = 0.0
    for item in jobs:
        result = item.get("result") or {}
        if item.get("ok") and result.get("seconds") and result.get("images_this_run"):
            images += int(result["images_this_run"])
            seconds += float(result["seconds"])
    if images == 0:
        return None
    return seconds / images


def _predict_wave_seconds(wave_payloads, seconds_per_image) -> float:
    if seconds_per_image is None:
        return 0.0
    estimates = []
    for payload in wave_payloads:
        subset = int(payload["subset_size"])
        epochs = int(payload["epochs"])
        probe = 10_000 + min(int(payload.get("train_probe", 4000)), subset)
        images = subset * epochs + (epochs + 1) * probe
        estimates.append(images * seconds_per_image)
    return max(estimates) if estimates else 0.0


def _refresh_plots(out: Path, jobs: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for item in jobs:
        result = item.get("result") or {}
        if not item.get("ok") or not result.get("history"):
            continue
        grouped.setdefault(result["condition"], {"legend": result["legend"], "histories": [], "late": result["late_stage"]})
        grouped[result["condition"]]["histories"].append(result["history"])

    def draw(path, ylabel, value):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for condition, bundle in grouped.items():
            epochs = sorted({row["epoch"] for history in bundle["histories"] for row in history})
            means = []
            kept_epochs = []
            for epoch in epochs:
                values = [value(row, bundle["late"]) for history in bundle["histories"] for row in history if row["epoch"] == epoch]
                values = [item for item in values if item is not None]
                if not values:
                    continue
                kept_epochs.append(epoch)
                means.append(sum(values) / len(values))
            if means:
                ax.plot(kept_epochs, means, marker="o", markersize=3, label=bundle["legend"])
        ax.set_xlabel("epoch (-1 is initialization)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    if not grouped:
        return
    draw(out / "test_accuracy.png", "test accuracy", lambda row, _late: row["test_accuracy"])
    draw(out / "train_accuracy.png", "train accuracy", lambda row, _late: row["train_accuracy"])
    draw(out / "entropy_late.png", "normalized hydrodynamic entropy", lambda row, late: row["s_tilde"].get(late))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for item in jobs:
        result = item.get("result") or {}
        if not item.get("ok"):
            continue
        final = [row for row in result["history"] if row["epoch"] == result["epochs"] - 1]
        initial = [row for row in result["history"] if row["epoch"] == -1]
        if not final or not initial:
            continue
        names = list(final[0]["s_tilde"])
        ax.plot(range(len(names)), [final[0]["s_tilde"][name] for name in names], marker="o", label=result["tag"])
    ax.set_xlabel("stage")
    ax.set_ylabel("normalized hydrodynamic entropy")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "entropy_depth_final.png", dpi=140)
    plt.close(fig)


def main():
    import time

    out = output_dir()
    started = time.time()
    torch.backends.cudnn.benchmark = True
    print("cuda devices", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(index, props.name, round(props.total_memory / 1e9, 1), "GB", flush=True)
    from entropy import run_unit_tests
    run_unit_tests()

    from torchvision import datasets
    datasets.CIFAR10(data_dir(), train=True, download=True)
    datasets.CIFAR10(data_dir(), train=False, download=True)

    queue = _session_queue(out)
    report = {
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "plan": "cifar10 map cnn, one kaggle session",
        "session_limit_hours": SESSION_LIMIT_S / 3600,
        "queued": [payload["tag"] for _task, payload in queue],
        "jobs": [],
        "skipped": [],
    }
    _write(out, report)

    width = max(1, torch.cuda.device_count())
    for start in range(0, len(queue), width):
        wave = queue[start:start + width]
        payloads = [payload for _task, payload in wave]
        rate = _seconds_per_image(report["jobs"])
        predicted = _predict_wave_seconds(payloads, rate)
        elapsed = time.time() - started
        if rate is not None and elapsed + predicted > SESSION_LIMIT_S:
            for _task, payload in wave:
                report["skipped"].append({"tag": payload["tag"], "reason": "session budget", "predicted_s": round(predicted, 1)})
            # Everything left in the queue would start even later.
            for _task, payload in queue[start + width:]:
                report["skipped"].append({"tag": payload["tag"], "reason": "session budget"})
            print("stopping, predicted wave would cross the session limit", flush=True)
            _write(out, report)
            break
        print("wave", [payload["tag"] for payload in payloads], f"predicted_s={predicted:.0f}", flush=True)
        report["jobs"].extend(_run_waves(wave, out))
        report["elapsed_s"] = round(time.time() - started, 1)
        report["seconds_per_image"] = _seconds_per_image(report["jobs"])
        _refresh_plots(out, report["jobs"])
        _write(out, report)

    report["elapsed_s"] = round(time.time() - started, 1)
    _refresh_plots(out, report["jobs"])
    _write(out, report)
    print("session done", report["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
