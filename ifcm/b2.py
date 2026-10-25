"""Model B2: B1 findings plus signed LLM-member edge uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ifcm.b1 import ROOT, RUNS, _json_new, _json_read, _npz_new, apply_calibration, fit_calibration, run_dir
from ifcm.b1_inputs import derm_context, derm_diagnoses, lidc_context
from ifcm.b2_inputs import member_bounds
from ifcm.intuitionistic_reasoning import IFS, reason_b1, reason_b2
from ifcm.safe_cnn import sha


def _paths(run_id: str) -> tuple[Path, Path]:
    parent = run_dir(run_id).parent
    return parent / "b1", parent / "b2"


def _inputs(run_id: str, dataset: str, encoder: str):
    b1, target = _paths(run_id)
    base = _json_read(b1 / "manifest.json")
    proof = _json_read(b1 / "prediction_manifest.json")
    if base["dataset"] != dataset or base["encoder"] != encoder:
        raise ValueError("B1 dataset/encoder mismatch")
    for name, expected in base["artifact_hashes"].items():
        if sha(b1 / name) != expected:
            raise ValueError(f"B1 changed: {name}")
    if proof["prediction_sha256"] != sha(b1 / "test_predictions.npz"):
        raise ValueError("B1 test predictions changed")
    panel, bounds, edges, unsure = member_bounds(ROOT, dataset)
    with np.load(b1 / "edge_mean.npz", allow_pickle=False) as z:
        if not np.array_equal(z["edge_mean"], panel.mean):
            raise ValueError("B1/B2 edge means differ")
    return b1, target, base, panel, bounds, edges, unsure


def _prediction(b1_path: Path, bounds, calibration: dict) -> dict[str, np.ndarray]:
    with np.load(b1_path, allow_pickle=False) as z:
        names = ("ids", "patients", "split", "fold", "concepts", "diagnoses",
                 "finding_mean", "finding_mu", "finding_nu", "finding_pi")
        out = {name: z[name] for name in names}
    finding = IFS(out["finding_mu"], out["finding_nu"], out["finding_pi"])
    result = reason_b2(finding, bounds.lower, bounds.upper)
    data_type = "lidc" if list(out["diagnoses"]) == ["MAL", "BEN"] else "derm"
    probability, decision, p_mal = apply_calibration(result.score, calibration, data_type)
    out.update({"diagnosis_mu": result.diagnosis.mu,
                "diagnosis_nu": result.diagnosis.nu, "diagnosis_pi": result.diagnosis.pi,
                "positive_lower": result.positive_lower, "positive_upper": result.positive_upper,
                "negative_lower": result.negative_lower, "negative_upper": result.negative_upper,
                "raw_score": result.score, "calibrated": probability, "decision": decision})
    if p_mal is not None:
        out["p_mal"] = p_mal
    return out


def _labels(dataset: str, encoder: str, ids: np.ndarray, cnn_run_id: str):
    if dataset == "lidc":
        _, split, ratings = lidc_context(ROOT, cnn_run_id)
        fit = set(split["clear_fit_ids"])
        mask = np.array([str(id_) in fit for id_ in ids])
        if mask.sum() != 740 or list(ids[mask]) != split["clear_fit_ids"]:
            raise ValueError("LIDC calibration IDs differ from strict fit")
        y = (ratings.loc[ids[mask], "mal_mean"].to_numpy() > 3).astype(int)
        return mask, np.stack([y, 1 - y], axis=1)
    meta, splits, _ = derm_context(ROOT, encoder)
    expected = [f"row:{i:04d}" for i in np.concatenate([splits["train"], splits["valid"]])]
    if list(ids) != expected:
        raise ValueError("Derm calibration indices differ")
    return np.ones(len(ids), bool), derm_diagnoses(meta)[np.concatenate([splits["train"], splits["valid"]])]


def dry_run(args):
    b1, target, base, panel, bounds, edges, unsure = _inputs(args.run_id, args.dataset, args.encoder)
    print(json.dumps({"method": "B2_signed_ifs_v1", "dataset": args.dataset,
                      "encoder": args.encoder, "b1_source": str(b1),
                      "output_directory": str(target), "members": len(panel.members),
                      "edges": list(panel.mean.shape), "mean_raw_half_width": float(bounds.raw_half_width.mean()),
                      "fit_rows": base["rows"]["fit"], "test_rows": base["rows"]["test"],
                      "test_labels_used": False}, indent=2))


def fit(args):
    b1, target, base, panel, bounds, edges, unsure = _inputs(args.run_id, args.dataset, args.encoder)
    if target.exists():
        raise FileExistsError(f"B2 run already exists: {target}")
    with np.load(b1 / "fit_predictions.npz", allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        finding = IFS(z["finding_mu"], z["finding_nu"], z["finding_pi"])
    mask, labels = _labels(args.dataset, args.encoder, ids, base["cnn_run_id"])
    raw = reason_b2(finding, bounds.lower, bounds.upper).score
    calibration = fit_calibration(raw[mask], labels)
    calibration["fit_ids"] = list(ids[mask])
    calibration["split"] = "strict_outer_fit_inner_oof" if args.dataset == "lidc" else "train_plus_valid"
    predictions = _prediction(b1 / "fit_predictions.npz", bounds, calibration)
    target.mkdir(parents=True, exist_ok=False)
    _npz_new(target / "edge_bounds.npz", mean=bounds.mean, lower=bounds.lower,
             upper=bounds.upper, raw_half_width=bounds.raw_half_width,
             left_width=bounds.left_width, right_width=bounds.right_width,
             member_edges=edges, member_unsure=unsure,
             members=np.asarray(panel.members), concepts=np.asarray(panel.concepts),
             diagnoses=np.asarray(panel.diagnoses))
    _json_new(target / "calibration.json", calibration)
    _npz_new(target / "fit_predictions.npz", **predictions)
    _json_new(target / "manifest.json", {
        "method": "B2_signed_ifs_v1", "dataset": args.dataset, "encoder": args.encoder,
        "protocol": base["protocol"], "b1_manifest_sha256": sha(b1 / "manifest.json"),
        "b1_fit_sha256": sha(b1 / "fit_predictions.npz"),
        "b1_test_sha256": sha(b1 / "test_predictions.npz"),
        "code_hashes": {str(p): sha(p) for p in (ROOT / "ifcm/b2.py", ROOT / "ifcm/b2_inputs.py")},
        "artifact_hashes": {name: sha(target / name) for name in
                            ("edge_bounds.npz", "calibration.json", "fit_predictions.npz")},
        "members": list(panel.members), "concepts": list(panel.concepts),
        "diagnoses": list(panel.diagnoses), "calibration_ids": calibration["fit_ids"],
        "test_labels_used": False,
        "warning": "Signed edge intervals are heuristic, not calibrated confidence bounds."})
    print(f"B2 fit saved: {target}; calibration rows={calibration['n_fit']}")


def _frozen(b1, target, manifest):
    if manifest["b1_manifest_sha256"] != sha(b1 / "manifest.json") or manifest["b1_fit_sha256"] != sha(b1 / "fit_predictions.npz") or manifest["b1_test_sha256"] != sha(b1 / "test_predictions.npz"):
        raise ValueError("B1 source changed")
    for path, digest in manifest["code_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("B2 code changed")
    for name, digest in manifest["artifact_hashes"].items():
        if sha(target / name) != digest:
            raise ValueError("B2 artifact changed")


def predict(args):
    b1, target, base, panel, bounds, edges, unsure = _inputs(args.run_id, args.dataset, args.encoder)
    out = target / "test_predictions.npz"
    if out.exists() or (target / "prediction_manifest.json").exists():
        raise FileExistsError("B2 test predictions already exist")
    manifest = _json_read(target / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder:
        raise ValueError("B2 manifest dataset mismatch")
    _frozen(b1, target, manifest)
    with np.load(target / "edge_bounds.npz", allow_pickle=False) as z:
        if not np.array_equal(z["lower"], bounds.lower) or not np.array_equal(z["upper"], bounds.upper):
            raise ValueError("B2 edge bounds changed")
    calibration = _json_read(target / "calibration.json")
    predictions = _prediction(b1 / "test_predictions.npz", bounds, calibration)
    _npz_new(out, **predictions)
    _json_new(target / "prediction_manifest.json", {"sha256": sha(out),
              "ids": list(map(str, predictions["ids"])), "test_labels_used": False})
    print(f"B2 held-out predictions saved: {out}; rows={len(predictions['ids'])}")


def verify(args):
    b1, target, base, panel, bounds, edges, unsure = _inputs(args.run_id, args.dataset, args.encoder)
    manifest = _json_read(target / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder:
        raise ValueError("B2 manifest dataset mismatch")
    _frozen(b1, target, manifest)
    with np.load(target / "edge_bounds.npz", allow_pickle=False) as z:
        for name, value in (("mean", bounds.mean), ("lower", bounds.lower),
                            ("upper", bounds.upper), ("member_edges", edges),
                            ("member_unsure", unsure)):
            if not np.array_equal(z[name], value):
                raise ValueError(f"B2 edge mismatch: {name}")
    calibration = _json_read(target / "calibration.json")
    proof = _json_read(target / "prediction_manifest.json")
    if proof["sha256"] != sha(target / "test_predictions.npz") or proof["test_labels_used"] is not False:
        raise ValueError("B2 prediction proof failed")
    for stage in ("fit", "test"):
        expected = _prediction(b1 / f"{stage}_predictions.npz", bounds, calibration)
        with np.load(target / f"{stage}_predictions.npz", allow_pickle=False) as z:
            if set(z.files) != set(expected):
                raise ValueError("B2 archive fields differ")
            for name, value in expected.items():
                if not np.array_equal(z[name], value):
                    raise ValueError(f"B2 {stage} mismatch: {name}")
    if proof["ids"] != list(map(str, expected["ids"])):
        raise ValueError("B2 test ID coverage failed")
    # This exact identity is a hard operator invariant, not an outcome check.
    with np.load(b1 / "fit_predictions.npz", allow_pickle=False) as z:
        finding = IFS(z["finding_mu"], z["finding_nu"], z["finding_pi"])
    if not np.array_equal(reason_b2(finding, bounds.mean, bounds.mean).score,
                          reason_b1(finding, bounds.mean).score):
        raise ValueError("B2 zero-width limit does not equal B1")
    print(json.dumps({"status": "PASS", "dataset": args.dataset,
                      "fit_rows": base["rows"]["fit"], "test_rows": base["rows"]["test"],
                      "zero_width_equals_b1": True, "test_labels_used": False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--encoder", choices=("cnn", "monet", "biomedclip"))
    parser.add_argument("--run-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "fit", "predict", "verify"):
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
        else:
            verify(args)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
