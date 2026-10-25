"""Frozen, patient-paired CNN vs ViT LIDC diagnosis comparison.

This script reads already evaluated predictions only. It never fits models or
selects hyperparameters from held-out labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ifcm.b1_inputs import lidc_context
from ifcm.safe_cnn import BASE


RUNS = {
    "CNN": ("isbi-b1-lidc-v1", "isbi-lidc-strict-v3"),
    "ViT-Tiny": ("isbi-vit-tiny-ab-v1", "isbi-vit-tiny-lidc-v1"),
    "ViT-Small": ("isbi-vit-small-ab-v1", "isbi-vit-small-lidc-v1"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output", default="isbi-vit-encoder-comparison-v1")
    parser.add_argument("--cnn-run", default=RUNS["CNN"][0])
    parser.add_argument("--cnn-source", default=RUNS["CNN"][1])
    parser.add_argument("--tiny-run", default=RUNS["ViT-Tiny"][0])
    parser.add_argument("--tiny-source", default=RUNS["ViT-Tiny"][1])
    parser.add_argument("--small-run", default=RUNS["ViT-Small"][0])
    parser.add_argument("--small-source", default=RUNS["ViT-Small"][1])
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    root = Path(__file__).resolve().parents[1]
    runs = {"CNN": (args.cnn_run, args.cnn_source),
            "ViT-Tiny": (args.tiny_run, args.tiny_source),
            "ViT-Small": (args.small_run, args.small_source)}
    scores = {}
    ids_ref = patients_ref = None
    for encoder, (run, source) in runs.items():
        _, split, ratings = lidc_context(root, source)
        clear = split["clear_test_ids"]
        for model in ("a", "b1", "b2"):
            path = BASE / run / model / "test_predictions.npz"
            with np.load(path, allow_pickle=False) as z:
                ids = z["ids"].astype(str)
                patients = z["patients"].astype(str)
                prob = z["p_mal"].astype(float)
            if list(ids) != split["test_ids"] or len(prob) != len(ids):
                raise ValueError(f"Run {run} has different test ID alignment")
            if ids_ref is None:
                ids_ref, patients_ref = ids, patients
            elif not np.array_equal(ids_ref, ids) or not np.array_equal(patients_ref, patients):
                raise ValueError("Encoder predictions are not paired by nodule and patient")
            scores[(encoder, model)] = prob
    _, split, ratings = lidc_context(root, runs["CNN"][1])
    mask = np.isin(ids_ref, split["clear_test_ids"])
    if mask.sum() != 462 or len(np.unique(patients_ref[mask])) != 195:
        raise ValueError("Test denominator or patient count changed")
    y = (ratings.loc[ids_ref, "mal_mean"].to_numpy() > 3).astype(int)
    groups = patients_ref[mask]
    ordered = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in ordered}
    rng = np.random.default_rng(20261025)
    draws = [np.concatenate([by_group[g] for g in rng.choice(ordered, len(ordered), replace=True)])
             for _ in range(args.bootstrap)]
    ys = y[mask]
    results = []
    for model in ("a", "b1", "b2"):
        cnn = scores[("CNN", model)][mask]
        for encoder in ("ViT-Tiny", "ViT-Small"):
            vit = scores[(encoder, model)][mask]
            diffs = [roc_auc_score(ys[idx], vit[idx]) - roc_auc_score(ys[idx], cnn[idx])
                     for idx in draws if len(np.unique(ys[idx])) == 2]
            low, high = np.quantile(diffs, [.025, .975])
            results.append({"model": model, "comparison": f"{encoder} minus CNN",
                            "n_nodules": int(mask.sum()), "n_patients": len(ordered),
                            "cnn_auroc": roc_auc_score(ys, cnn),
                            "vit_auroc": roc_auc_score(ys, vit),
                            "difference": roc_auc_score(ys, vit) - roc_auc_score(ys, cnn),
                            "ci_low": low, "ci_high": high,
                            "valid_bootstrap": len(diffs), "unit": "patient"})
    output = BASE / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "paired_encoder_auroc.csv", index=False)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {output / 'paired_encoder_auroc.csv'}")


if __name__ == "__main__":
    main()
