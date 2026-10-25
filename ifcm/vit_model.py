"""ViT finding predictor for the frozen LIDC patient split.

The three central CT slices are treated as channels, exactly as in the CNN
pathway. Only training-set reader fractions are targets. No test label enters
training, checkpoint selection, or prediction.
"""

from pathlib import Path

import numpy as np

from ifcm.safe_cnn import CONCEPTS, sha
from ifcm.safe_cnn_model import images as cnn_images, augment


MODELS = {
    "tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
EPOCHS = 10
BATCH = 32


def libraries():
    import torch
    import torch.nn.functional as F
    import timm
    return torch, F, timm


def images(source, ids):
    torch, F, _ = libraries()
    # CNN source already performs the audited HU window and central-slice extraction.
    return F.interpolate(cnn_images(source, ids), size=(224, 224),
                         mode="bilinear", align_corners=False)


def model_for(variant, pretrained):
    _, _, timm = libraries()
    if variant not in MODELS:
        raise ValueError(f"Unknown ViT variant: {variant}")
    return timm.create_model(MODELS[variant], pretrained=pretrained,
                             num_classes=len(CONCEPTS))


def train_model(source, ratings, train_ids, variant, seed, fold, where):
    torch, F, timm = libraries()
    if not torch.cuda.is_available():
        raise RuntimeError("ViT training requires CUDA")
    actual_seed = 100 * fold + seed if fold is not None else 1000 + seed
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    np.random.seed(actual_seed)
    x = images(source, train_ids)
    labels = torch.from_numpy(ratings.loc[train_ids, list(CONCEPTS)].to_numpy(dtype=np.float32))
    if not torch.isfinite(labels).all() or not ((labels >= 0) & (labels <= 1)).all():
        raise ValueError("Invalid reader-fraction training targets")
    network = model_for(variant, pretrained=True).cuda()
    opt = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(train_ids) / BATCH))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, total_steps=steps)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(EPOCHS):
        network.train()
        permutation = np.random.permutation(len(train_ids))
        for start in range(0, len(permutation), BATCH):
            batch = permutation[start:start + BATCH]
            xb = augment((x[batch] - mean) / std).cuda(non_blocking=True)
            yb = labels[batch].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.binary_cross_entropy_with_logits(network(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        print(f"{variant} fold={fold} seed={seed} epoch={epoch + 1}/{EPOCHS} loss={loss.item():.5f}", flush=True)
    checkpoint = where / "checkpoint.pt"
    with checkpoint.open("xb") as handle:
        torch.save(network.cpu().state_dict(), handle)
    return {"checkpoint_sha256": sha(checkpoint), "actual_seed": actual_seed,
            "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": 1e-4,
            "architecture": MODELS[variant], "timm_version": timm.__version__,
            "target": "reader_fraction", "checkpoint_selection": "fixed_epoch_10_no_validation",
            "validation_ids": [], "validation_patients": [], "source_code_sha256": sha(Path(__file__))}


def predict_model(source, ratings, pred_ids, variant, checkpoint, output):
    torch, _, _ = libraries()
    network = model_for(variant, pretrained=False)
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = network.to(device).eval()
    x = images(source, pred_ids)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    with torch.inference_mode():
        values = torch.cat([
            torch.sigmoid(network(((x[start:start + BATCH] - mean) / std).to(device)).float()).cpu()
            for start in range(0, len(pred_ids), BATCH)
        ]).numpy()
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array(pred_ids),
                            patients=ratings.loc[pred_ids].patient.to_numpy(dtype=str),
                            tta0=values)
