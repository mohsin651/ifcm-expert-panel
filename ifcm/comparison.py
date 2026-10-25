"""Strict aligned Model A adapter and A/B1/B2 provenance checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ifcm.b1 import ROOT, _json_new, _json_read, _npz_new, apply_calibration, fit_calibration, run_dir
from ifcm.b2 import _labels
from ifcm.safe_cnn import sha


def _paths(run_id):
    parent = run_dir(run_id).parent
    return parent / "b1", parent / "b2", parent / "a"


def _load_inputs(args):
    b1, b2, a = _paths(args.run_id)
    manifest = _json_read(b1 / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder:
        raise ValueError("B1 dataset or encoder mismatch")
    for name, expected in manifest["artifact_hashes"].items():
        if sha(b1 / name) != expected:
            raise ValueError(f"B1 source changed: {name}")
    proof = _json_read(b1 / "prediction_manifest.json")
    if proof["prediction_sha256"] != sha(b1 / "test_predictions.npz"):
        raise ValueError("B1 test predictions changed")
    with np.load(b1 / "edge_mean.npz", allow_pickle=False) as z:
        edge = z["edge_mean"]
        concepts, diagnoses = list(z["concepts"]), list(z["diagnoses"])
    return b1, b2, a, manifest, edge, concepts, diagnoses


def _predict(source: Path, edge: np.ndarray, calibration: dict, dataset: str):
    with np.load(source, allow_pickle=False) as z:
        result = {name: z[name] for name in ("ids", "patients", "split", "fold", "concepts", "diagnoses", "finding_mean")}
    score = result["finding_mean"] @ edge
    prob, decision, p_mal = apply_calibration(score, calibration, dataset)
    result.update({"raw_score": score, "calibrated": prob, "decision": decision})
    if p_mal is not None:
        result["p_mal"] = p_mal
    return result


def dry_run(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    print(json.dumps({"method": "A_weighted_sum_v1", "dataset": args.dataset,
                      "encoder": args.encoder, "protocol": base["protocol"],
                      "b1_source": str(b1), "a_output": str(a), "b2_output": str(b2),
                      "fit_rows": base["rows"]["fit"], "test_rows": base["rows"]["test"],
                      "concepts": concepts, "diagnoses": diagnoses,
                      "test_labels_used": False}, indent=2))


def fit(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    if a.exists():
        raise FileExistsError(f"Model A run already exists: {a}")
    with np.load(b1 / "fit_predictions.npz", allow_pickle=False) as z:
        ids, x = z["ids"].astype(str), z["finding_mean"]
    mask, labels = _labels(args.dataset, args.encoder, ids, base["cnn_run_id"])
    calibration = fit_calibration((x @ edge)[mask], labels)
    calibration["fit_ids"] = list(ids[mask])
    calibration["split"] = "strict_outer_fit_inner_oof" if args.dataset == "lidc" else "train_plus_valid"
    predictions = _predict(b1 / "fit_predictions.npz", edge, calibration, args.dataset)
    a.mkdir(parents=True, exist_ok=False)
    _json_new(a / "calibration.json", calibration)
    _npz_new(a / "fit_predictions.npz", **predictions)
    _json_new(a / "manifest.json", {
        "method": "A_weighted_sum_v1", "dataset": args.dataset, "encoder": args.encoder,
        "protocol": base["protocol"], "b1_manifest_sha256": sha(b1 / "manifest.json"),
        "b1_fit_sha256": sha(b1 / "fit_predictions.npz"),
        "b1_test_sha256": sha(b1 / "test_predictions.npz"),
        "code_sha256": sha(ROOT / "ifcm/comparison.py"),
        "artifact_hashes": {name: sha(a / name) for name in ("calibration.json", "fit_predictions.npz")},
        "concepts": concepts, "diagnoses": diagnoses, "calibration_ids": calibration["fit_ids"],
        "test_labels_used": False})
    print(f"Model A fit saved: {a}; calibration rows={calibration['n_fit']}")


def _frozen(b1, a, manifest):
    for key, path in (("b1_manifest_sha256", b1 / "manifest.json"),
                      ("b1_fit_sha256", b1 / "fit_predictions.npz"),
                      ("b1_test_sha256", b1 / "test_predictions.npz")):
        if manifest[key] != sha(path):
            raise ValueError(f"Model A input changed: {path}")
    if manifest["code_sha256"] != sha(ROOT / "ifcm/comparison.py"):
        raise ValueError("Model A code changed")
    for name, digest in manifest["artifact_hashes"].items():
        if sha(a / name) != digest:
            raise ValueError(f"Model A artifact changed: {name}")


def predict(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    if (a / "test_predictions.npz").exists() or (a / "prediction_manifest.json").exists():
        raise FileExistsError("Model A test predictions already exist")
    manifest = _json_read(a / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder:
        raise ValueError("Model A manifest mismatch")
    _frozen(b1, a, manifest)
    result = _predict(b1 / "test_predictions.npz", edge,
                      _json_read(a / "calibration.json"), args.dataset)
    out = a / "test_predictions.npz"
    _npz_new(out, **result)
    _json_new(a / "prediction_manifest.json", {"sha256": sha(out),
              "ids": list(map(str, result["ids"])), "test_labels_used": False})
    print(f"Model A held-out predictions saved: {out}; rows={len(result['ids'])}")


def verify(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    manifest = _json_read(a / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder:
        raise ValueError("Model A manifest mismatch")
    _frozen(b1, a, manifest)
    proof = _json_read(a / "prediction_manifest.json")
    if proof["sha256"] != sha(a / "test_predictions.npz") or proof["test_labels_used"] is not False:
        raise ValueError("Model A held-out proof failed")
    calibration = _json_read(a / "calibration.json")
    for stage in ("fit", "test"):
        expected = _predict(b1 / f"{stage}_predictions.npz", edge, calibration, args.dataset)
        with np.load(a / f"{stage}_predictions.npz", allow_pickle=False) as z:
            if set(z.files) != set(expected):
                raise ValueError("Model A archive fields differ")
            for name, value in expected.items():
                if not np.array_equal(z[name], value):
                    raise ValueError(f"Model A {stage} mismatch: {name}")
    if proof["ids"] != list(map(str, expected["ids"])):
        raise ValueError("Model A test IDs differ")
    print(json.dumps({"status": "PASS", "method": "A_weighted_sum_v1",
                      "dataset": args.dataset, "test_labels_used": False}, indent=2))


def parity(args):
    b1, b2, a, base, edge, concepts, diagnoses = _load_inputs(args)
    for folder in (a, b2):
        if not (folder / "prediction_manifest.json").exists():
            raise FileNotFoundError(f"Missing completed predictions: {folder}")
    for stage in ("fit", "test"):
        files = [a / f"{stage}_predictions.npz", b1 / f"{stage}_predictions.npz", b2 / f"{stage}_predictions.npz"]
        with np.load(files[0], allow_pickle=False) as za, np.load(files[1], allow_pickle=False) as zb, np.load(files[2], allow_pickle=False) as zc:
            for name in ("ids", "patients", "split", "fold", "concepts", "diagnoses", "finding_mean"):
                if not np.array_equal(za[name], zb[name]) or not np.array_equal(zb[name], zc[name]):
                    raise ValueError(f"A/B1/B2 mismatch: {stage}/{name}")
            for name in ("finding_mu", "finding_nu", "finding_pi"):
                if not np.array_equal(zb[name], zc[name]):
                    raise ValueError(f"B1/B2 finding mismatch: {stage}/{name}")
    print(json.dumps({"status": "PASS", "dataset": args.dataset,
                      "models": ["A", "B1", "B2"], "fit_rows": base["rows"]["fit"],
                      "test_rows": base["rows"]["test"], "alignment": "identical"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--encoder", choices=("cnn", "monet", "biomedclip"))
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "fit", "predict", "verify", "parity"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    args.encoder = args.encoder or ("cnn" if args.dataset == "lidc" else "monet")
    try:
        if args.dry_run:
            dry_run(args)
        elif args.fit:
            fit(args)
        elif args.predict:
            predict(args)
        elif args.verify:
            verify(args)
        else:
            parity(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
