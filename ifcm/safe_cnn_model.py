"""User-invoked ResNet-18 training/prediction for a frozen LIDC patient boundary."""

from pathlib import Path

import numpy as np

from ifcm.safe_cnn import CONCEPTS, VIEWS, sha


def torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision
    except ImportError as exc:
        raise RuntimeError("Use .venv-upstream (PyTorch/torchvision) for CNN tasks") from exc
    return torch, nn, F, torchvision


def images(source: dict, ids: list[str]):
    torch, _, F, _ = torch_modules()
    wanted = set(ids)
    crops = {}
    for filename in source["crop_files"]:
        with np.load(filename, allow_pickle=False) as z:
            for key in z.files:
                if key in wanted:
                    if key in crops:
                        raise ValueError(f"Duplicate crop {key}")
                    crops[key] = z[key]
    if set(crops) != wanted:
        raise ValueError("Missing requested crop IDs")
    output = []
    for id_ in ids:
        crop = crops[id_]
        if crop.ndim != 3 or crop.shape[2] != 5:
            raise ValueError(f"Invalid 5-slice crop for {id_}")
        v = np.clip((crop[:, :, 1:4].astype(np.float32) + 1000) / 1400, 0, 1)
        t = torch.from_numpy(v).permute(2, 0, 1)[None]
        output.append(F.interpolate(t, size=(96, 96), mode="bilinear", align_corners=False)[0])
    return torch.stack(output)


def view(x, name):
    torch, _, F, _ = torch_modules()
    if name == "tta0":
        return x
    if name == "tta1":
        return torch.rot90(x, 1, (2, 3))
    if name == "tta2":
        return torch.flip(x, (3,))
    if name == "tta3":
        return torch.flip(x, (2,))
    if name.startswith("blur"):
        sigma = float(name[4:])
        size = int(4 * sigma) | 1
        axis = torch.arange(size) - size // 2
        weight = torch.exp(-axis.float() ** 2 / (2 * sigma * sigma))
        weight /= weight.sum()
        channels = x.shape[1]
        x = F.conv2d(F.pad(x, (size // 2,) * 4, mode="reflect"),
                     weight.view(1, 1, 1, size).repeat(channels, 1, 1, 1), groups=channels)
        return F.conv2d(x, weight.view(1, 1, size, 1).repeat(channels, 1, 1, 1), groups=channels)
    if name.startswith("noise"):
        generator = torch.Generator().manual_seed(0)
        return (x + torch.randn(x.shape, generator=generator) * float(name[5:]) / 1400).clamp(0, 1)
    raise ValueError(f"Unknown view {name}")


def augment(x):
    torch, _, _, _ = torch_modules()
    if np.random.rand() < .5:
        x = torch.flip(x, (3,))
    if np.random.rand() < .5:
        x = torch.flip(x, (2,))
    return torch.rot90(x, np.random.randint(4), (2, 3))


def network(n_out: int, weights: Path | None = None):
    torch, nn, _, torchvision = torch_modules()
    model = torchvision.models.resnet18(weights=None)  # never downloads
    if weights is not None:
        raw = torch.load(weights, map_location="cpu", weights_only=True)
        expected = set(model.state_dict())
        missing = expected - set(raw)
        # Official older ResNet-18 checkpoints omit BatchNorm tracking counters.
        # PyTorch initializes those counters when loading an older state dict.
        if (set(raw) - expected or
                any(not key.endswith(".num_batches_tracked") for key in missing)):
            raise ValueError("ImageNet weights do not match torchvision ResNet-18")
        model.load_state_dict(raw, strict=True)
    model.fc = nn.Linear(512, n_out)
    return model


def train_model(source, ratings, train_ids, kind, seed, fold, weights, where):
    torch, _, F, _ = torch_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires CUDA; no CPU fallback is implicit")
    actual_seed = (100 * fold + seed if fold is not None else 1000 + seed) + (50 if kind == "malignancy" else 0)
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    x = images(source, train_ids)
    target = (ratings.loc[train_ids, list(CONCEPTS)].to_numpy(dtype=np.float32)
              if kind == "finding" else
              (ratings.loc[train_ids, "mal_mean"].to_numpy(dtype=np.float32) > 3).astype(np.float32)[:, None])
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError("Invalid training target")
    model = network(target.shape[1], weights).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 25 * int(np.ceil(len(train_ids) / 64))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    labels = torch.from_numpy(target)
    for epoch in range(25):
        model.train()
        permutation = np.random.permutation(len(train_ids))
        for start in range(0, len(permutation), 64):
            batch = permutation[start:start + 64]
            xb = augment((x[batch] - mean) / std).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(model(xb).float(), labels[batch].to("cuda"))
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
        print(f"epoch {epoch + 1}/25, loss {loss.item():.5f}", flush=True)
    checkpoint = where / "checkpoint.pt"
    with checkpoint.open("xb") as handle:
        torch.save(model.cpu().state_dict(), handle)
    return {"checkpoint_sha256": sha(checkpoint), "imagenet_weights_sha256": sha(weights),
            "actual_seed": actual_seed, "epochs": 25, "architecture": "resnet18_3slice_96",
            "target": "reader_fraction" if kind == "finding" else "binary_malignancy",
            "validation_ids": [], "validation_patients": [],
            "checkpoint_selection": "fixed_epoch_25_no_validation",
            "source_code_sha256": sha(Path(__file__))}


def predict_model(source, ratings, pred_ids, kind, checkpoint, output):
    torch, _, _, _ = torch_modules()
    n_out = len(CONCEPTS) if kind == "finding" else 1
    model = network(n_out)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    x = images(source, pred_ids)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    arrays = {}
    with torch.no_grad():
        for name in VIEWS:
            transformed = (view(x, name) - mean) / std
            arrays[name] = torch.cat([
                torch.sigmoid(model(transformed[start:start + 64].to(device)).float()).cpu()
                for start in range(0, len(pred_ids), 64)
            ]).numpy()
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array(pred_ids),
                            patients=ratings.loc[pred_ids].patient.to_numpy(dtype=str), **arrays)
