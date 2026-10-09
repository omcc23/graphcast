"""One-session CIFAR-10 run.

The network is a small conv net whose maps stay at 32, 16, and 8, so every
probed stage has enough Fourier modes for hydrodynamic entropy. Training and
the entropy probe are separate: the FFT is not in the loss.
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from entropy import hydrodynamic_entropy
from probe import discover_stages


# Shared by every job so seed comparisons use the same images and, for a given
# label condition, the same targets. The run seed only changes initialization
# and the SGD shuffle.
INDEX_SEED = 0
RANDOM_LABEL_SEED = 12345


def map_cnn(num_classes: int = 10) -> nn.Module:
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block32 = block(3, 64)
            self.pool32 = nn.MaxPool2d(2)
            self.block16 = block(64, 128)
            self.pool16 = nn.MaxPool2d(2)
            self.block8 = block(128, 256)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.pool32(self.block32(x))
            x = self.pool16(self.block16(x))
            x = self.block8(x)
            return self.head(x)

    return _Net()


def apply_init(model: nn.Module, scheme: str) -> None:
    if scheme == "kaiming":
        return
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if scheme == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif scheme == "normal":
                nn.init.normal_(module.weight, 0, 0.01)
            else:
                raise ValueError(scheme)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class IndexedSet(Dataset):
    def __init__(self, base, indices, targets):
        self.base = base
        self.indices = list(indices)
        self.targets = list(targets)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _label = self.base[self.indices[index]]
        return image, int(self.targets[index])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize():
    return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def _targets_for(base_targets, indices, random_labels: bool, label_noise: float, num_classes: int, seed: int):
    clean = [int(base_targets[i]) for i in indices]
    generator = torch.Generator().manual_seed(seed)
    if random_labels:
        return torch.randint(0, num_classes, (len(indices),), generator=generator).tolist()
    if label_noise <= 0:
        return clean
    flip = torch.rand(len(indices), generator=generator) < label_noise
    drawn = torch.randint(0, num_classes, (len(indices),), generator=generator)
    assigned = []
    for index, label in enumerate(clean):
        if not bool(flip[index]):
            assigned.append(label)
            continue
        replacement = int(drawn[index])
        if replacement == label:
            replacement = (replacement + 1) % num_classes
        assigned.append(replacement)
    return assigned


def _loaders(data_root, subset_size, batch_size, random_labels, label_noise, augment, train_probe, num_workers):
    eval_tf = transforms.Compose([transforms.ToTensor(), _normalize()])
    train_tf = eval_tf
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _normalize(),
        ])
    train_base = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    eval_base = datasets.CIFAR10(data_root, train=True, download=True, transform=eval_tf)
    test_base = datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)
    generator = torch.Generator().manual_seed(INDEX_SEED)
    order = torch.randperm(len(train_base), generator=generator).tolist()
    subset_size = min(int(subset_size), len(order))
    indices = order[:subset_size]
    label_seed = RANDOM_LABEL_SEED if random_labels else 30_000 + int(round(label_noise * 1000))
    targets = _targets_for(train_base.targets, indices, random_labels, label_noise, 10, label_seed)
    train_set = IndexedSet(train_base, indices, targets)
    probe_count = min(int(train_probe), len(indices))
    train_probe_set = IndexedSet(eval_base, indices[:probe_count], targets[:probe_count])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_probe_loader = DataLoader(train_probe_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, train_probe_loader, test_loader, subset_size


@torch.no_grad()
def _measure(model, loader, stages, device):
    training = model.training
    model.eval()
    names = [stage["name"] for stage in stages]
    s_sum = {name: 0.0 for name in names}
    c_sum = {name: 0.0 for name in names}
    correct = 0
    count = 0
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            captured = {}
            hooks = []

            def make_hook(stage):
                def _hook(_module, _inputs, output):
                    stats = hydrodynamic_entropy(output)
                    captured[stage["name"]] = (
                        stats["s_tilde"].detach(),
                        stats["centroid"].detach(),
                    )
                return _hook

            for stage in stages:
                hooks.append(stage["module"].register_forward_hook(make_hook(stage)))
            logits = model(images)
            for handle in hooks:
                handle.remove()
            correct += int((logits.argmax(1) == targets).sum())
            count += int(targets.shape[0])
            for name in names:
                values, centroid = captured[name]
                s_sum[name] += float(values.sum())
                c_sum[name] += float(centroid.sum())
    finally:
        model.train(training)
    scale = max(count, 1)
    return {
        "accuracy": correct / scale,
        "n": count,
        "s_tilde": {name: s_sum[name] / scale for name in names},
        "centroid": {name: c_sum[name] / scale for name in names},
    }


def _parameter_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        total = total + parameter.detach().float().pow(2).sum()
    return float(torch.sqrt(total))


def train_session(device, payload: dict) -> dict:
    """Train one condition and probe hydrodynamic entropy after every epoch."""
    seed = int(payload["seed"])
    epochs = int(payload["epochs"])
    out_dir = payload["out_dir"]
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # The GPU worker is already a spawned process. Extra loader workers on a
    # 4-core Kaggle VM fight the two training processes.
    num_workers = 0
    train_loader, train_probe_loader, test_loader, subset_size = _loaders(
        payload["data_root"],
        int(payload["subset_size"]),
        int(payload["batch_size"]),
        bool(payload["random_labels"]),
        float(payload["label_noise"]),
        bool(payload["augment"]),
        int(payload.get("train_probe", 4000)),
        num_workers,
    )
    model = map_cnn().to(device)
    apply_init(model, payload["init"])
    example = next(iter(test_loader))[0][:1].to(device)
    model.eval()
    stages = discover_stages(model, example)
    if len(stages) < 3:
        raise RuntimeError(f"expected 3 spatial stages, found {[(s['name'], s['spatial']) for s in stages]}")
    late_name = stages[-1]["name"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(payload["lr"]),
        momentum=0.9,
        weight_decay=float(payload["weight_decay"]),
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    history_path = out_dir / f"{payload['tag']}_history.json"

    def record(epoch, loss, seconds, train_stats, test_stats):
        row = {
            "epoch": epoch,
            "seconds": round(seconds, 3),
            "loss": loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "parameter_norm": round(_parameter_norm(model), 6),
            "train_accuracy": train_stats["accuracy"],
            "test_accuracy": test_stats["accuracy"],
            "s_tilde": test_stats["s_tilde"],
            "centroid": test_stats["centroid"],
            "train_s_tilde": train_stats["s_tilde"],
        }
        history.append(row)
        import json
        history_path.write_text(json.dumps(history), encoding="utf-8")
        s_text = " ".join(f"{name}={test_stats['s_tilde'][name]:.3f}" for name in test_stats["s_tilde"])
        print(
            f"{payload['tag']} epoch={epoch} loss={loss if loss is not None else float('nan'):.3f} "
            f"train_acc={train_stats['accuracy']:.3f} test_acc={test_stats['accuracy']:.3f} "
            f"s[{s_text}] {seconds:.1f}s",
            flush=True,
        )

    started = time.time()
    record(-1, None, 0.0, _measure(model, train_probe_loader, stages, device), _measure(model, test_loader, stages, device))
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        epoch_started = time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
        train_stats = _measure(model, train_probe_loader, stages, device)
        test_stats = _measure(model, test_loader, stages, device)
        record(epoch, loss_sum / max(seen, 1), time.time() - epoch_started, train_stats, test_stats)
        schedule.step()

    checkpoint = out_dir / f"{payload['tag']}.pt"
    torch.save({"model": model.state_dict(), "tag": payload["tag"]}, checkpoint)
    return {
        "tag": payload["tag"],
        "condition": payload["condition"],
        "legend": payload["legend"],
        "seed": seed,
        "epochs": epochs,
        "subset_size": subset_size,
        "batch_size": int(payload["batch_size"]),
        "random_labels": bool(payload["random_labels"]),
        "label_noise": float(payload["label_noise"]),
        "augment": bool(payload["augment"]),
        "weight_decay": float(payload["weight_decay"]),
        "init": payload["init"],
        "lr": float(payload["lr"]),
        "stages": [{"name": stage["name"], "spatial": list(stage["spatial"])} for stage in stages],
        "late_stage": late_name,
        "images_this_run": subset_size * epochs + (epochs + 1) * (10_000 + min(int(payload.get("train_probe", 4000)), subset_size)),
        "seconds": round(time.time() - started, 3),
        "history": history,
        "checkpoint": str(checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
