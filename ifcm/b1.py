"""Issue 3: isolated, fit-only-calibrated Model B1 CLI.

Examples: python -m ifcm.b1 --dataset lidc --run-id NAME --dry-run
          python -m ifcm.b1 --dataset lidc --run-id NAME --fit
          python -m ifcm.b1 --dataset lidc --run-id NAME --predict
          python -m ifcm.b1 --dataset lidc --run-id NAME --verify

No work occurs on import. No GPU inference or LLM regeneration is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.special import expit

from ifcm.b1_inputs import (
    DERM_CONCEPTS, DERM_DIAGNOSES, LIDC_CONCEPTS, LIDC_DIAGNOSES,
    Rows, derm_context, derm_diagnoses, derm_findings, derm_probe_draws,
    fit_derm_probes, lidc_context, lidc_rows, load_panel,
)
from ifcm.intuitionistic_reasoning import IFS, finding_ifs_from_draws, reason_b1
from ifcm.safe_cnn import sha


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "fresh_runs/ifs_comparison"
CODE_PATHS = (ROOT / "ifcm/b1.py", ROOT / "ifcm/b1_inputs.py",
              ROOT / "ifcm/intuitionistic_reasoning.py")


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("run-id must be a simple 1–80 character name")
    return RUNS / run_id / "b1"


def _json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _npz_new(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def source_paths(dataset: str, encoder: str, cnn_folder: Path | None,
                 panel_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if dataset == "lidc":
        assert cnn_folder is not None
        files = [ROOT / "results_lidc/lidc_ratings.csv",
                 cnn_folder / "source_manifest.json", cnn_folder / "outer_split.json",
                 cnn_folder / "boundary_verification.json",
                 cnn_folder / "predictions/inner_oof.npz",
                 cnn_folder / "predictions/outer_test.npz"]
    else:
        files = [ROOT / "meta/meta.csv", *(ROOT / f"meta/{part}_indexes.csv"
                                            for part in ("train", "valid", "test")),
                 ROOT / f"results/emb_{encoder}.npz"]
    return tuple([*files, *panel_paths])


def source_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.resolve()): sha(path) for path in paths}


def fit_calibration(score: np.ndarray, labels: np.ndarray) -> dict:
    """One independent two-parameter logistic fit per diagnosis, no test rows."""
    from sklearn.linear_model import LogisticRegression

    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (score.ndim != 2 or score.shape != labels.shape or score.size == 0 or
            not np.isfinite(score).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Invalid calibration scores/labels")
    slope, intercept, positives = [], [], []
    for d in range(score.shape[1]):
        if len(np.unique(labels[:, d])) != 2:
            raise ValueError("Each diagnosis needs both calibration classes")
        model = LogisticRegression(max_iter=1000)
        model.fit(score[:, d:d + 1], labels[:, d])
        slope.append(float(model.coef_[0, 0]))
        intercept.append(float(model.intercept_[0]))
        positives.append(int(labels[:, d].sum()))
    return {"slope": slope, "intercept": intercept,
            "positives": positives, "n_fit": len(score),
            "method": "independent_sklearn_logistic_default_C1_max_iter1000"}


def apply_calibration(score: np.ndarray, calibration: dict, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    score = np.asarray(score, dtype=np.float64)
    slope = np.asarray(calibration["slope"], dtype=np.float64)
    intercept = np.asarray(calibration["intercept"], dtype=np.float64)
    if (score.ndim != 2 or score.shape[1] != len(slope) or slope.shape != intercept.shape or
            not np.isfinite(score).all() or not np.isfinite(slope).all() or
            not np.isfinite(intercept).all()):
        raise ValueError("Invalid calibration shape or coefficients")
    probability = expit(score * slope[None, :] + intercept[None, :])
    if dataset == "lidc":
        if probability.shape[1] != 2:
            raise ValueError("LIDC needs MAL and BEN scores")
        p_mal = probability[:, 0] / probability.sum(axis=1)
        return probability, (p_mal > 0.5).astype(np.int64), p_mal
    if dataset == "derm":
        if probability.shape[1] != 4:
            raise ValueError("Derm needs four diagnosis scores")
        return probability, probability.argmax(axis=1).astype(np.int64), None
    raise ValueError("Unknown dataset")


def prediction_arrays(rows: Rows, edge_mean: np.ndarray, calibration: dict,
                      dataset: str, stage: str) -> dict[str, np.ndarray]:
    if stage not in {"fit", "test"}:
        raise ValueError("B1 stage must be fit or test")
    if len(rows.ids) != len(set(map(str, rows.ids))) or len(rows.ids) != len(rows.patients):
        raise ValueError("Prediction IDs are duplicated or misaligned")
    mean = rows.draws.mean(axis=2)
    finding = finding_ifs_from_draws(rows.draws, mean)
    reasoned = reason_b1(finding, edge_mean)
    calibrated, decision, p_mal = apply_calibration(reasoned.score, calibration, dataset)
    output = {
        "ids": np.asarray(rows.ids, dtype=str),
        "patients": np.asarray(rows.patients, dtype=str),
        "split": np.full(len(rows.ids), stage),
        "fold": np.asarray(rows.fold, dtype=np.int64),
        "concepts": np.asarray(rows.concepts, dtype=str),
        "diagnoses": np.asarray(LIDC_DIAGNOSES if dataset == "lidc" else DERM_DIAGNOSES),
        "finding_mean": mean,
        "finding_mu": finding.mu, "finding_nu": finding.nu, "finding_pi": finding.pi,
        "diagnosis_mu": reasoned.diagnosis.mu,
        "diagnosis_nu": reasoned.diagnosis.nu,
        "diagnosis_pi": reasoned.diagnosis.pi,
        "positive_lower": reasoned.positive_lower,
        "positive_upper": reasoned.positive_upper,
        "negative_lower": reasoned.negative_lower,
        "negative_upper": reasoned.negative_upper,
        "raw_score": reasoned.score,
        "calibrated": calibrated,
        "decision": decision,
    }
    if p_mal is not None:
        output["p_mal"] = p_mal
    return output


def _context(args: argparse.Namespace) -> tuple[object, Path | None, dict, object]:
    panel = load_panel(ROOT, args.dataset)
    if args.dataset == "lidc":
        if args.encoder != "cnn":
            raise ValueError("Issue 3 strict LIDC adapter supports only CNN")
        folder, split, ratings = lidc_context(ROOT, args.cnn_run_id)
        return panel, folder, split, ratings
    if args.encoder not in {"monet", "biomedclip"}:
        raise ValueError("Derm encoder must be monet or biomedclip")
    meta, splits, image = derm_context(ROOT, args.encoder)
    return panel, None, splits, (meta, image)


def _preview(args: argparse.Namespace, panel, cnn_folder, split, data) -> dict:
    sources = source_paths(args.dataset, args.encoder, cnn_folder, panel.paths)
    if args.dataset == "lidc":
        rows_fit = len(split["fit_ids"])
        rows_test = len(split["test_ids"])
        calibration_rows = len(split["clear_fit_ids"])
        patient_counts = {"fit": len(split["fit_patients"]),
                          "test": len(split["test_patients"])}
        protocol = "strict_outer_original_seed0"
    else:
        rows_fit = len(split["train"]) + len(split["valid"])
        rows_test = len(split["test"])
        calibration_rows = rows_fit
        patient_counts = None
        protocol = "derm_official_index_split"
    target = run_dir(args.run_id)
    return {
        "method": "B1_signed_ifs_v1", "dataset": args.dataset, "encoder": args.encoder,
        "protocol": protocol, "cnn_run_id": args.cnn_run_id if args.dataset == "lidc" else None,
        "view": "tta0" if args.dataset == "lidc" else "img_clean_0",
        "concepts": list(panel.concepts), "diagnoses": list(panel.diagnoses),
        "panel_styles": list(panel.styles), "panel_members": list(panel.members),
        "rows": {"fit": rows_fit, "test": rows_test, "calibration": calibration_rows},
        "patients": patient_counts,
        "source_hashes": source_hashes(sources),
        "code_hashes": source_hashes(CODE_PATHS),
        "output_directory": str(target),
        "expected_outputs": [str(target / name) for name in
                             ("manifest.json", "edge_mean.npz", "calibration.json",
                              "fit_predictions.npz", "test_predictions.npz")],
        "limitations": "No test labels are used for fitting; no held-out metrics are computed. "
                       "Derm embeddings align by saved row position, not a source image ID.",
    }


def dry_run(args: argparse.Namespace) -> None:
    panel, folder, split, data = _context(args)
    print(json.dumps(_preview(args, panel, folder, split, data), indent=2))


def fit(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id)
    if target.exists():
        raise FileExistsError(f"B1 run already exists: {target}")
    panel, folder, split, data = _context(args)
    preview = _preview(args, panel, folder, split, data)
    if args.dataset == "lidc":
        ratings = data
        rows = lidc_rows(folder, split, ratings, "fit")
        fit_set = set(split["clear_fit_ids"])
        mask = np.array([id_ in fit_set for id_ in rows.ids])
        if mask.sum() != 740:
            raise ValueError("Expected 740 clear-label LIDC calibration nodules")
        y_mal = (ratings.loc[rows.ids[mask], "mal_mean"].to_numpy() > 3).astype(int)
        labels = np.stack([y_mal, 1 - y_mal], axis=1)
        probes = None
    else:
        meta, image = data
        fit_indices = np.concatenate([split["train"], split["valid"]])
        concepts = derm_findings(meta)
        probes = fit_derm_probes(image, concepts, fit_indices)
        draws = derm_probe_draws(image[fit_indices], probes)
        rows = Rows(np.array([f"row:{i:04d}" for i in fit_indices]),
                    np.full(len(fit_indices), "", dtype="<U1"), draws,
                    np.full(len(fit_indices), -1), DERM_CONCEPTS)
        mask = np.ones(len(rows.ids), bool)
        labels = derm_diagnoses(meta)[fit_indices]
    finding = finding_ifs_from_draws(rows.draws, rows.draws.mean(axis=2))
    raw_score = reason_b1(finding, panel.mean).score
    calibration = fit_calibration(raw_score[mask], labels)
    calibration["fit_ids"] = list(map(str, rows.ids[mask]))
    calibration["fit_patients"] = sorted(set(map(str, rows.patients[mask]))) if args.dataset == "lidc" else []
    calibration["split"] = "outer_fit_inner_oof" if args.dataset == "lidc" else "train_plus_valid"
    predictions = prediction_arrays(rows, panel.mean, calibration, args.dataset, "fit")
    target.mkdir(parents=True, exist_ok=False)
    _npz_new(target / "edge_mean.npz", edge_mean=panel.mean,
             concepts=np.array(panel.concepts), diagnoses=np.array(panel.diagnoses),
             members=np.array(panel.members))
    if probes is not None:
        _npz_new(target / "probes.npz", **probes)
    _json_new(target / "calibration.json", calibration)
    _npz_new(target / "fit_predictions.npz", **predictions)
    artifacts = ["edge_mean.npz", "calibration.json", "fit_predictions.npz"]
    if probes is not None:
        artifacts.append("probes.npz")
    preview["artifact_hashes"] = {name: sha(target / name) for name in artifacts}
    preview["calibration_ids"] = calibration["fit_ids"]
    preview["calibration_patients"] = calibration["fit_patients"]
    preview["stage"] = "fit_complete"
    _json_new(target / "manifest.json", preview)
    print(f"B1 fit saved: {target}; calibration rows={calibration['n_fit']}")


def _check_frozen(manifest: dict) -> None:
    for path, expected in manifest["source_hashes"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"B1 source changed: {path}")
    for path, expected in manifest["code_hashes"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"B1 code changed: {path}")


def predict(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id)
    output = target / "test_predictions.npz"
    if output.exists() or (target / "prediction_manifest.json").exists():
        raise FileExistsError("B1 held-out predictions already exist; refusing overwrite")
    manifest = _json_read(target / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder or manifest["cnn_run_id"] != (
            args.cnn_run_id if args.dataset == "lidc" else None):
        raise ValueError("B1 CLI arguments disagree with frozen run")
    _check_frozen(manifest)
    panel, folder, split, data = _context(args)
    if list(panel.members) != manifest["panel_members"] or list(panel.concepts) != manifest["concepts"]:
        raise ValueError("B1 panel changed")
    with np.load(target / "edge_mean.npz", allow_pickle=False) as z:
        if not np.array_equal(z["edge_mean"], panel.mean):
            raise ValueError("B1 edge mean changed")
    calibration = _json_read(target / "calibration.json")
    if args.dataset == "lidc":
        rows = lidc_rows(folder, split, data, "test")
    else:
        meta, image = data
        test_indices = split["test"]
        with np.load(target / "probes.npz", allow_pickle=False) as z:
            probes = {name: z[name] for name in ("coef", "intercept", "active")}
        draws = derm_probe_draws(image[test_indices], probes)
        rows = Rows(np.array([f"row:{i:04d}" for i in test_indices]),
                    np.full(len(test_indices), "", dtype="<U1"), draws,
                    np.full(len(test_indices), -1), DERM_CONCEPTS)
    predictions = prediction_arrays(rows, panel.mean, calibration, args.dataset, "test")
    _npz_new(output, **predictions)
    _json_new(target / "prediction_manifest.json",
              {"prediction_sha256": sha(output),
               "fit_manifest_sha256": sha(target / "manifest.json"),
               "ids": list(map(str, rows.ids)),
               "patients": sorted(set(map(str, rows.patients))) if args.dataset == "lidc" else [],
               "test_labels_used": False})
    print(f"B1 held-out predictions saved: {output}; rows={len(rows.ids)}; no scores computed")


def _check_archive(path: Path, ids: list[str], patients: list[str], concepts: list[str],
                   diagnoses: list[str], calibration: dict, edge: np.ndarray,
                   dataset: str, stage: str, fold: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as z:
        if (list(map(str, z["ids"])) != ids or list(map(str, z["patients"])) != patients or
                list(z["split"]) != [stage] * len(ids) or
                not np.array_equal(z["fold"], fold) or
                list(map(str, z["concepts"])) != concepts or
                list(map(str, z["diagnoses"])) != diagnoses):
            raise ValueError(f"B1 row/column alignment failed: {path}")
        n, c, d = len(ids), len(concepts), len(diagnoses)
        for name, shape in (("finding_mean", (n, c)), ("finding_mu", (n, c)),
                            ("finding_nu", (n, c)), ("finding_pi", (n, c)),
                            ("diagnosis_mu", (n, d)), ("diagnosis_nu", (n, d)),
                            ("diagnosis_pi", (n, d)), ("raw_score", (n, d)),
                            ("calibrated", (n, d))):
            value = z[name]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid B1 array {name}: {path}")
        finding = IFS(z["finding_mu"], z["finding_nu"], z["finding_pi"])
        if (np.any(z["finding_mean"] < finding.lower - 1e-10) or
                np.any(z["finding_mean"] > finding.upper + 1e-10)):
            raise ValueError("B1 finding mean lies outside its interval")
        result = reason_b1(finding, edge)
        for name, value in (("diagnosis_mu", result.diagnosis.mu),
                            ("diagnosis_nu", result.diagnosis.nu),
                            ("diagnosis_pi", result.diagnosis.pi),
                            ("raw_score", result.score),
                            ("positive_lower", result.positive_lower),
                            ("positive_upper", result.positive_upper),
                            ("negative_lower", result.negative_lower),
                            ("negative_upper", result.negative_upper)):
            if not np.allclose(z[name], value, rtol=0, atol=1e-12):
                raise ValueError(f"B1 propagation mismatch: {name}")
        probability, decision, p_mal = apply_calibration(result.score, calibration, dataset)
        if (not np.allclose(z["calibrated"], probability, rtol=0, atol=1e-12) or
                not np.array_equal(z["decision"], decision) or
                (p_mal is not None and not np.allclose(z["p_mal"], p_mal, rtol=0, atol=1e-12))):
            raise ValueError("B1 calibration/decision mismatch")
        if (np.any((z["calibrated"] < 0) | (z["calibrated"] > 1)) or
                np.any((z["finding_mean"] < 0) | (z["finding_mean"] > 1))):
            raise ValueError("B1 probability out of range")
        if any(name in z.files for name in ("y", "y_true", "test_label", "diagnosis_label")):
            raise ValueError("B1 prediction archive must not contain diagnosis labels")


def verify(args: argparse.Namespace) -> None:
    target = run_dir(args.run_id)
    manifest = _json_read(target / "manifest.json")
    if manifest["dataset"] != args.dataset or manifest["encoder"] != args.encoder or manifest["cnn_run_id"] != (
            args.cnn_run_id if args.dataset == "lidc" else None):
        raise ValueError("B1 CLI arguments disagree with frozen run")
    _check_frozen(manifest)
    for name, expected in manifest["artifact_hashes"].items():
        if sha(target / name) != expected:
            raise ValueError(f"B1 artifact changed: {name}")
    proof = _json_read(target / "prediction_manifest.json")
    if proof["fit_manifest_sha256"] != sha(target / "manifest.json") or proof["prediction_sha256"] != sha(target / "test_predictions.npz"):
        raise ValueError("B1 held-out prediction proof changed")
    panel, folder, split, data = _context(args)
    if list(panel.members) != manifest["panel_members"] or list(panel.concepts) != manifest["concepts"]:
        raise ValueError("B1 panel changed")
    with np.load(target / "edge_mean.npz", allow_pickle=False) as z:
        edge = z["edge_mean"]
        if not np.array_equal(edge, panel.mean):
            raise ValueError("B1 edge source mismatch")
    calibration = _json_read(target / "calibration.json")
    if args.dataset == "lidc":
        ratings = data
        fit_ids, test_ids = split["fit_ids"], split["test_ids"]
        fit_patients = list(map(str, ratings.loc[fit_ids].patient))
        test_patients = list(map(str, ratings.loc[test_ids].patient))
        if (set(fit_patients) & set(test_patients) or
                calibration["fit_ids"] != split["clear_fit_ids"] or
                set(calibration["fit_patients"]) != set(ratings.loc[split["clear_fit_ids"]].patient)):
            raise ValueError("B1 calibration crossed patient boundary")
        fit_fold = np.array([split["inner_fold_by_patient"][p] for p in fit_patients])
        test_fold = np.full(len(test_ids), -1)
    else:
        fit_indices = np.concatenate([split["train"], split["valid"]])
        fit_ids = [f"row:{i:04d}" for i in fit_indices]
        test_ids = [f"row:{i:04d}" for i in split["test"]]
        fit_patients = [""] * len(fit_ids)
        test_patients = [""] * len(test_ids)
        if calibration["fit_ids"] != fit_ids:
            raise ValueError("B1 Derm calibration IDs changed")
        fit_fold = np.full(len(fit_ids), -1)
        test_fold = np.full(len(test_ids), -1)
    expected_proof_patients = sorted(set(test_patients)) if args.dataset == "lidc" else []
    if (proof["ids"] != test_ids or proof["patients"] != expected_proof_patients or
            proof["test_labels_used"] is not False or
            calibration["n_fit"] != len(calibration["fit_ids"])):
        raise ValueError("B1 prediction/calibration coverage mismatch")
    _check_archive(target / "fit_predictions.npz", fit_ids, fit_patients,
                   manifest["concepts"], manifest["diagnoses"], calibration, edge,
                   args.dataset, "fit", fit_fold)
    _check_archive(target / "test_predictions.npz", test_ids, test_patients,
                   manifest["concepts"], manifest["diagnoses"], calibration, edge,
                   args.dataset, "test", test_fold)
    print(json.dumps({"status": "PASS", "method": "B1_signed_ifs_v1",
                      "dataset": args.dataset, "fit_rows": len(fit_ids),
                      "test_rows": len(test_ids), "test_labels_used": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("lidc", "derm"))
    parser.add_argument("--encoder", choices=("cnn", "monet", "biomedclip"), default=None)
    parser.add_argument("--cnn-run-id", default="isbi-lidc-strict-v3")
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
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
