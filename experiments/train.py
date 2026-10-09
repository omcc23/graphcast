"""CIFAR-100 ResNet-18 training with a per-epoch hydrodynamic-entropy probe.

Images are resized to 224 so the four stages have the same grids as an
ImageNet ResNet (56, 28, 14, 7). A 32-pixel input would leave the last stage
at 1x1, where the entropy has a single mode.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from entropy import hydrodynamic_entropy
from probe import discover_stages


def cifar100_loaders(root, batch_size, num_workers=0):
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
    probe_set = datasets.CIFAR100(root, train=False, download=True, transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    probe_loader = DataLoader(probe_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_set, train_loader, probe_loader


def resnet18_cifar100():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def _stage_entropy(model, stages, images):
    captured = {}
    hooks = []

    def make_hook(stage):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if stage["kind"] == "map":
                maps = tensor
            else:
                tokens = tensor[:, 1:, :]
                batch, count, dim = tokens.shape
                side = int(count ** 0.5)
                maps = tokens.transpose(1, 2).reshape(batch, dim, side, side)
            stats = hydrodynamic_entropy(maps)
            captured[stage["name"]] = stats["s_tilde"]
        return _hook

    for stage in stages:
        hooks.append(stage["module"].register_forward_hook(make_hook(stage)))
    logits = model(images)
    for handle in hooks:
        handle.remove()
    return logits, captured


@torch.no_grad()
def probe_epoch(model, loader, stages, device):
    model.eval()
    totals = {stage["name"]: 0.0 for stage in stages}
    correct = 0
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, captured = _stage_entropy(model, stages, images)
        correct += int((logits.argmax(1) == targets).sum())
        count += int(targets.shape[0])
        for name, values in captured.items():
            totals[name] += float(values.sum())
    return {
        "accuracy": correct / max(count, 1),
        "s_tilde": {name: totals[name] / max(count, 1) for name in totals},
    }


def train_cifar(device, out_dir, data_root, epochs=20, batch_size=64, seed=0, random_labels=False, regularize=None, probe_limit=2000):
    """Train one ResNet-18.

    regularize: None, "delay" (keep early-stage entropy high), or "collapse"
    (drive last-stage entropy down).
    """
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    _train_set, train_loader, probe_loader = cifar100_loaders(data_root, batch_size)
    if probe_limit:
        probe_loader = DataLoader(
            Subset(probe_loader.dataset, list(range(probe_limit))),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    model = resnet18_cifar100().to(device)
    if random_labels:
        generator = torch.Generator().manual_seed(seed + 17)
        shuffled = torch.randint(0, 100, (len(train_loader.dataset),), generator=generator)
        train_loader.dataset.targets = shuffled.tolist()

    example = next(iter(probe_loader))[0][:1].to(device)
    model.eval()
    stages = discover_stages(model, example)
    early_name = stages[0]["name"]
    late_name = stages[-1]["name"]
    print("stages", [(stage["name"], stage["spatial"]) for stage in stages], flush=True)
    path = out_dir / f"resnet18_{'random' if random_labels else 'true'}_{regularize or 'plain'}_s{seed}.pt"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, 3 * epochs // 4], gamma=0.1)
    history = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if regularize is None:
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets)
            else:
                logits, captured = _stage_entropy(model, stages, images)
                loss = nn.functional.cross_entropy(logits, targets)
                if regularize == "delay":
                    loss = loss + 0.1 * (1.0 - captured[early_name].mean())
                elif regularize == "collapse":
                    loss = loss + 0.1 * captured[late_name].mean()
                else:
                    raise ValueError(regularize)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
        schedule.step()
        measured = probe_epoch(model, probe_loader, stages, device)
        row = {"epoch": epoch, "loss": loss_sum / max(seen, 1), **measured}
        history.append(row)
        (out_dir / f"{path.stem}_history.json").write_text(json.dumps(history), encoding="utf-8")
        print(
            f"seed={seed} random_labels={random_labels} regularize={regularize} "
            f"epoch={epoch} loss={row['loss']:.3f} acc={measured['accuracy']:.3f} "
            f"s_early={measured['s_tilde'][early_name]:.3f} s_late={measured['s_tilde'][late_name]:.3f}",
            flush=True,
        )
    torch.save({"model": model.state_dict(), "history": history, "stages": [stage["name"] for stage in stages]}, path)
    return {
        "checkpoint": str(path),
        "history": history,
        "stage_names": [stage["name"] for stage in stages],
        "random_labels": random_labels,
        "regularize": regularize,
        "seed": seed,
    }
