"""Leakage-safe LIDC CNN orchestration. No work occurs on import.

All commands write only to a new run under fresh_runs/ifs_comparison.
Training and inference are explicit user-invoked subcommands.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "fresh_runs/ifs_comparison"
RATINGS = ROOT / "results_lidc/lidc_ratings.csv"
LEGACY = ROOT / "results_lidc/lidc_cnn.npz"  # IDs/concept order only; never predictions
CONCEPTS = (
    "spiculated", "lobulated", "ill_defined", "sharp_margin", "solid", "part_solid",
    "ground_glass", "round", "irregular_shape", "benign_calc", "other_calc", "fat",
    "large", "small",
)
VIEWS = ("tta0", "tta1", "tta2", "tta3", "blur1", "blur2", "blur4", "noise100", "noise200")
N_FOLDS, N_SEEDS = 5, 5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id) or run_id in {".", ".."}:
        raise ValueError("run-id must be a simple 1–80 character name")
    return BASE / run_id / "cnn"


def ratings_table() -> pd.DataFrame:
    r = pd.read_csv(RATINGS)
    b = pd.DataFrame({
        "nodule": r.nodule.astype(str), "patient": r.patient.astype(str),
        "malignancy": r.malignancy,
        "spiculated": r.spiculation >= 3, "lobulated": r.lobulation >= 3,
        "ill_defined": r.margin <= 2, "sharp_margin": r.margin >= 4,
        "solid": r.texture == 5, "part_solid": r.texture.between(2, 4),
        "ground_glass": r.texture == 1, "round": r.sphericity >= 4,
        "irregular_shape": r.sphericity <= 2,
        "benign_calc": r.calcification.isin([1, 2, 3, 5]),
        "other_calc": r.calcification == 4, "fat": r.internalStructure == 3,
        "large": r.diameter > 15, "small": r.diameter < 6,
    })
    g = b.groupby("nodule", sort=True)
    if (g.patient.nunique() != 1).any():
        raise ValueError("Nodule mapped to multiple patients")
    n = g[list(CONCEPTS)].mean().astype(float)
    n["patient"] = g.patient.first()
    n["mal_mean"] = g.malignancy.mean()
    return n


def crop_index(paths: list[Path]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key in mapping:
                    raise ValueError(f"Duplicate crop ID {key}")
                mapping[key] = path
    return mapping


def patient_split(patients: np.ndarray, seed: int = 0) -> tuple[list[str], list[str]]:
    ordered = np.array(patients, dtype=str).copy()
    np.random.default_rng(seed).shuffle(ordered)
    test = set(ordered[: int(len(ordered) * .4)])
    return sorted(set(ordered) - test), sorted(test)


def inner_folds(patients: list[str], seed: int = 17) -> dict[str, int]:
    ordered = np.array(sorted(patients), dtype=str)
    np.random.default_rng(seed).shuffle(ordered)
    return {str(patient): i % N_FOLDS for i, patient in enumerate(ordered)}


def assert_boundary(fit: set[str], test: set[str], inner: dict[str, int],
                    train: set[str] | None = None, predicted: set[str] | None = None) -> None:
    if not fit or not test or fit & test or set(inner) != fit:
        raise ValueError("Invalid outer patient split or inner coverage")
    if set(inner.values()) != set(range(N_FOLDS)):
        raise ValueError("Missing inner fold")
    if train is not None and (not train or not train <= fit or train & test):
        raise ValueError("Training contains an outer-test patient")
    if predicted is not None and train is not None and predicted & train:
        raise ValueError("Prediction patients overlap training patients")


def validate_split(source: dict, split: dict, n: pd.DataFrame) -> None:
    ids = source["ids"]
    if source["protocol"] != "strict_outer_original_seed0" or ids != sorted(set(ids)):
        raise ValueError("Wrong cohort protocol or unordered/duplicate crop IDs")
    eligible = n.loc[[i for i in n.index if i in set(ids) and n.at[i, "mal_mean"] != 3]]
    eligible_fit, test = patient_split(eligible.patient.unique(), 0)
    fit = sorted(set(eligible_fit) | (set(n.loc[ids].patient) - set(test)))
    if split["fit_patients"] != fit or split["test_patients"] != test:
        raise ValueError("Outer patient mask differs from original seed-0 definition")
    if split["inner_fold_by_patient"] != inner_folds(fit):
        raise ValueError("Inner patient folds changed")
    assert_boundary(set(fit), set(test), split["inner_fold_by_patient"])
    fit_ids = [i for i in ids if n.at[i, "patient"] in fit]
    test_ids = [i for i in ids if n.at[i, "patient"] in test]
    if split["fit_ids"] != fit_ids or split["test_ids"] != test_ids:
        raise ValueError("Nodule IDs do not match patient masks")
    clear_fit = [i for i in fit_ids if n.at[i, "mal_mean"] != 3]
    clear_test = [i for i in test_ids if n.at[i, "mal_mean"] != 3]
    if (split["clear_fit_ids"] != clear_fit or split["clear_test_ids"] != clear_test or
        (len(clear_fit), len(clear_test), len(test)) != (740, 462, 195)):
        raise ValueError("Eligible split counts or IDs changed")
    if split["calibration_patients"] != sorted(set(n.loc[clear_fit].patient)):
        raise ValueError("Calibration patient set changed")
    if not set(split["selection_patients"]) <= set(split["calibration_patients"]):
        raise ValueError("Selection contains an ineligible or test patient")


def inputs(folder: Path) -> tuple[dict, dict, pd.DataFrame]:
    split = read(folder / "outer_split.json")
    source = read(folder / "source_manifest.json")
    n = ratings_table()
    paths = [Path(p) for p in source["crop_files"]]
    for p, expected in source["hashes"].items():
        if sha(Path(p)) != expected:
            raise ValueError(f"Source file changed: {p}")
    for p, expected in source["code_hashes"].items():
        if sha(Path(p)) != expected:
            raise ValueError(f"Run source code changed: {p}")
    if set(source["ids"]) != set(crop_index(paths)):
        raise ValueError("Crop IDs changed since preparation")
    if not set(source["ids"]) <= set(n.index):
        raise ValueError("Crop nodule missing reader ratings")
    validate_split(source, split, n)
    return source, split, n


def prepare(args: argparse.Namespace) -> None:
    folder = run_dir(args.run_id)
    if folder.exists():
        raise FileExistsError(f"Run already exists: {folder}")
    crop_pattern = args.crops_glob if Path(args.crops_glob).is_absolute() else str(ROOT / args.crops_glob)
    paths = [Path(p) for p in sorted(glob.glob(crop_pattern))]
    if not paths:
        raise FileNotFoundError(f"No crop bundles matching {args.crops_glob}")
    with np.load(LEGACY, allow_pickle=False) as z:
        expected_ids = set(map(str, z["ids"]))
        if tuple(z["concepts"]) != CONCEPTS:
            raise ValueError("Legacy concept order does not match")
    available = crop_index(paths)
    missing = expected_ids - set(available)
    extra = set(available) - expected_ids
    if missing or extra:
        raise ValueError(f"Original cohort crop coverage: {len(available)}/{len(expected_ids)}; "
                         f"missing={len(missing)}, extra={len(extra)}. Supply all original crop bundles.")
    n = ratings_table()
    if not expected_ids <= set(n.index):
        raise ValueError("Crop nodule missing reader ratings")
    eligible = n.loc[[i for i in n.index if i in expected_ids and n.at[i, "mal_mean"] != 3]]
    fit, test = patient_split(eligible.patient.unique(), 0)
    # Patients with only indeterminate nodules remain on the fit side.
    fit = sorted(set(fit) | (set(n.loc[sorted(expected_ids)].patient) - set(test)))
    inner = inner_folds(fit)
    assert_boundary(set(fit), set(test), inner)
    ids = sorted(expected_ids)
    outer_fit_ids = [i for i in ids if n.at[i, "patient"] in fit]
    outer_test_ids = [i for i in ids if n.at[i, "patient"] in test]
    clear_fit = [i for i in outer_fit_ids if n.at[i, "mal_mean"] != 3]
    clear_test = [i for i in outer_test_ids if n.at[i, "mal_mean"] != 3]
    if (len(clear_fit), len(clear_test), len(test)) != (740, 462, 195):
        raise ValueError("Original-seed0 split no longer matches the audited 740/462/195")
    selection_candidates = sorted(set(n.loc[clear_fit].patient))
    rng = np.random.default_rng(29)
    selected = sorted(map(str, rng.choice(selection_candidates, size=max(1, round(.2 * len(selection_candidates))), replace=False)))
    source = {"protocol": "strict_outer_original_seed0", "cohort": "original_1573",
              "crop_files": [str(p.resolve()) for p in paths], "ids": ids,
              "concepts": list(CONCEPTS), "views": list(VIEWS),
              "hashes": {str(p.resolve()): sha(p) for p in [*paths, RATINGS, LEGACY]},
              "code_hashes": {str(Path(__file__).resolve()): sha(Path(__file__)),
                              str(ROOT / "ifcm/safe_cnn_model.py"): sha(ROOT / "ifcm/safe_cnn_model.py")},
              "historical_cnn_predictions_used": False}
    split = {"seed": 0, "inner_seed": 17, "selection_seed": 29,
             "fit_patients": fit, "test_patients": test, "inner_fold_by_patient": inner,
             "selection_patients": selected, "calibration_patients": sorted(set(n.loc[clear_fit].patient)),
             "fit_ids": outer_fit_ids, "test_ids": outer_test_ids,
             "clear_fit_ids": clear_fit, "clear_test_ids": clear_test,
             "counts": {"all_crops": len(ids), "fit_crops": len(outer_fit_ids),
                        "test_crops": len(outer_test_ids), "clear_fit": 740,
                        "clear_test": 462, "eligible_fit_patients": 293,
                        "eligible_test_patients": 195}}
    print(json.dumps({"output": str(folder), "counts": split["counts"],
                      "crop_files": source["crop_files"], "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return
    folder.mkdir(parents=True, exist_ok=False)
    dump_new(folder / "source_manifest.json", source)
    dump_new(folder / "outer_split.json", split)


def task_ids(split: dict, n: pd.DataFrame, stage: str, fold: int | None) -> tuple[list[str], list[str]]:
    fit, test = set(split["fit_patients"]), set(split["test_patients"])
    inner = split["inner_fold_by_patient"]
    assert_boundary(fit, test, inner)
    if stage == "inner":
        if fold not in range(N_FOLDS):
            raise ValueError("Inner fold must be 0..4")
        pred_patients = {p for p, k in inner.items() if k == fold}
        train_patients = fit - pred_patients
    else:
        train_patients, pred_patients = fit, test
    assert_boundary(fit, test, inner, train_patients, pred_patients)
    train = [i for i in split["fit_ids"] if n.at[i, "patient"] in train_patients]
    pred_pool = split["fit_ids"] if stage == "inner" else split["test_ids"]
    pred = [i for i in pred_pool if n.at[i, "patient"] in pred_patients]
    return train, pred


def task_path(folder: Path, stage: str, kind: str, fold: int | None, seed: int) -> Path:
    tag = f"fold{fold}_seed{seed}" if stage == "inner" else f"seed{seed}"
    return folder / "tasks" / stage / kind / tag


def validate_task_manifest(manifest: dict, train: list[str], pred: list[str],
                           n: pd.DataFrame, split_hash: str, source_hash: str) -> None:
    if (manifest["train_ids"] != train or manifest["prediction_ids"] != pred or
        manifest["train_patients"] != sorted(set(n.loc[train].patient)) or
        manifest["prediction_patients"] != sorted(set(n.loc[pred].patient)) or
        manifest["outer_split_sha256"] != split_hash or
        manifest["source_manifest_sha256"] != source_hash):
        raise ValueError("Task manifest disagrees with frozen split/source")


def task(args: argparse.Namespace) -> None:
    folder = run_dir(args.run_id)
    source, split, n = inputs(folder)
    stage = "inner" if args.command.endswith("inner") else "final"
    fold = args.fold if stage == "inner" else None
    if args.seed not in range(N_SEEDS):
        raise ValueError("Seed must be 0..4")
    train, pred = task_ids(split, n, stage, fold)
    if args.kind == "malignancy":
        train = [i for i in train if n.at[i, "mal_mean"] != 3]
    train_patients = set(n.loc[train].patient)
    pred_patients = set(n.loc[pred].patient)
    assert_boundary(set(split["fit_patients"]), set(split["test_patients"]),
                    split["inner_fold_by_patient"], train_patients, pred_patients)
    where = task_path(folder, stage, args.kind, fold, args.seed)
    command = args.command
    print(json.dumps({"task": command, "stage": stage, "kind": args.kind,
                      "fold": fold, "seed": args.seed, "train_nodules": len(train),
                      "predict_nodules": len(pred), "train_patients": len(train_patients),
                      "predict_patients": len(pred_patients), "path": str(where),
                      "dry_run": args.dry_run}, indent=2))
    if args.dry_run:
        return
    if command.startswith("train"):
        if where.exists():
            if (where / "manifest.json").is_file() and (where / "checkpoint.pt").is_file():
                previous = read(where / "manifest.json")
                if (previous["train_ids"] == train and previous["prediction_ids"] == pred and
                    previous["outer_split_sha256"] == sha(folder / "outer_split.json") and
                    previous["source_manifest_sha256"] == sha(folder / "source_manifest.json") and
                    previous["checkpoint_sha256"] == sha(where / "checkpoint.pt")):
                    print("Completed matching task exists; skipping without overwrite")
                    return
            raise FileExistsError(f"Incomplete or conflicting task exists: {where}; inspect it, do not overwrite")
        if not args.weights or not Path(args.weights).is_file():
            raise FileNotFoundError("Provide local ResNet-18 ImageNet .pth via --weights; downloading is disabled")
        from ifcm.safe_cnn_model import train_model
        where.mkdir(parents=True, exist_ok=False)
        try:
            details = train_model(source, n, train, args.kind, args.seed, fold, Path(args.weights), where)
            manifest = {"stage": stage, "kind": args.kind, "fold": fold, "seed": args.seed,
                        "train_ids": train, "train_patients": sorted(train_patients),
                        "prediction_ids": pred, "prediction_patients": sorted(pred_patients),
                        "outer_split_sha256": sha(folder / "outer_split.json"),
                        "source_manifest_sha256": sha(folder / "source_manifest.json"),
                        **details}
            dump_new(where / "manifest.json", manifest)
        except Exception:
            print(f"Incomplete task retained at {where}; inspect before retrying")
            raise
    else:
        manifest = read(where / "manifest.json")
        validate_task_manifest(manifest, train, pred, n,
                               sha(folder / "outer_split.json"), sha(folder / "source_manifest.json"))
        if manifest["checkpoint_sha256"] != sha(where / "checkpoint.pt"):
            raise ValueError("Checkpoint does not match task manifest")
        output = where / "predictions.npz"
        if output.exists():
            with np.load(output, allow_pickle=False) as z:
                proof = where / "prediction_manifest.json"
                if (proof.is_file() and read(proof)["prediction_sha256"] == sha(output) and
                    list(z["ids"]) == pred and all(v in z.files for v in VIEWS)):
                    print("Completed matching predictions exist; skipping without overwrite")
                    return
            raise FileExistsError(f"Conflicting prediction output: {output}")
        from ifcm.safe_cnn_model import predict_model
        predict_model(source, n, pred, args.kind, where / "checkpoint.pt", output)
        dump_new(where / "prediction_manifest.json",
                 {"prediction_sha256": sha(output), "checkpoint_sha256": manifest["checkpoint_sha256"],
                  "prediction_ids": pred, "prediction_patients": sorted(pred_patients)})


def assemble(args: argparse.Namespace) -> None:
    folder = run_dir(args.run_id)
    source, split, n = inputs(folder)
    stage = args.stage
    ids = split["fit_ids"] if stage == "inner" else split["test_ids"]
    output = folder / "predictions" / ("inner_oof.npz" if stage == "inner" else "outer_test.npz")
    if output.exists():
        raise FileExistsError(output)
    expected = {(kind, fold, seed) for kind in ("finding", "malignancy")
                for fold in (range(N_FOLDS) if stage == "inner" else [None])
                for seed in range(N_SEEDS)}
    print(f"Assembling {stage}: {len(expected)} prediction shards; {len(ids)} nodule rows -> {output}")
    if args.dry_run:
        return
    lookup = {i: j for j, i in enumerate(ids)}
    p = {v: np.full((len(ids), len(CONCEPTS), N_SEEDS), np.nan, np.float32) for v in VIEWS}
    m = {v: np.full((len(ids), N_SEEDS), np.nan, np.float32) for v in VIEWS}
    for kind, fold, seed in sorted(expected, key=lambda x: (x[0], -1 if x[1] is None else x[1], x[2])):
        where = task_path(folder, stage, kind, fold, seed)
        manifest = read(where / "manifest.json")
        proof = read(where / "prediction_manifest.json")
        expected_train, expected_pred = task_ids(split, n, stage, fold)
        if kind == "malignancy":
            expected_train = [i for i in expected_train if n.at[i, "mal_mean"] != 3]
        validate_task_manifest(manifest, expected_train, expected_pred, n,
                               sha(folder / "outer_split.json"), sha(folder / "source_manifest.json"))
        if (proof["prediction_sha256"] != sha(where / "predictions.npz") or
            proof["checkpoint_sha256"] != manifest["checkpoint_sha256"]):
            raise ValueError("Prediction provenance mismatch")
        with np.load(where / "predictions.npz", allow_pickle=False) as z:
            row_ids = list(map(str, z["ids"]))
            if (row_ids != manifest["prediction_ids"] or len(set(row_ids)) != len(row_ids) or
                list(z["patients"]) != list(n.loc[row_ids].patient)):
                raise ValueError("Prediction IDs do not match manifest")
            for v in VIEWS:
                array = z[v]
                expected_shape = (len(row_ids), len(CONCEPTS) if kind == "finding" else 1)
                if array.shape != expected_shape or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                    raise ValueError(f"Invalid prediction array {where}: {v}")
                target = p[v] if kind == "finding" else m[v]
                for row, id_ in enumerate(row_ids):
                    j = lookup[id_]
                    if np.isfinite(target[j, ..., seed]).any():
                        raise ValueError(f"Duplicate prediction: {id_}, seed {seed}")
                    target[j, ..., seed] = array[row] if kind == "finding" else array[row, 0]
    if any(not np.isfinite(a).all() for a in [*p.values(), *m.values()]):
        raise ValueError("Incomplete prediction coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array(ids), patients=n.loc[ids].patient.to_numpy(dtype=str),
                            concepts=np.array(CONCEPTS),
                            fold=np.array([split["inner_fold_by_patient"].get(n.at[i, "patient"], -1) for i in ids]),
                            seeds=np.arange(N_SEEDS),
                            **{f"P_{v}": a for v, a in p.items()}, **{f"M_{v}": a for v, a in m.items()})


def verify(args: argparse.Namespace) -> None:
    folder = run_dir(args.run_id)
    source, split, n = inputs(folder)
    fit, test = set(split["fit_patients"]), set(split["test_patients"])
    inner = split["inner_fold_by_patient"]
    assert_boundary(fit, test, inner)
    assert set(split["selection_patients"]) <= fit
    assert set(split["calibration_patients"]) <= fit
    for stage in ("inner", "final"):
        ids = split["fit_ids"] if stage == "inner" else split["test_ids"]
        for kind in ("finding", "malignancy"):
            for fold in (range(N_FOLDS) if stage == "inner" else [None]):
                for seed in range(N_SEEDS):
                    expected_train, expected_pred = task_ids(split, n, stage, fold)
                    if kind == "malignancy":
                        expected_train = [i for i in expected_train if n.at[i, "mal_mean"] != 3]
                    where = task_path(folder, stage, kind, fold, seed)
                    m = read(where / "manifest.json")
                    validate_task_manifest(m, expected_train, expected_pred, n,
                                           sha(folder / "outer_split.json"),
                                           sha(folder / "source_manifest.json"))
                    if (not (where / "checkpoint.pt").is_file() or
                        sha(where / "checkpoint.pt") != m["checkpoint_sha256"]):
                        raise ValueError(f"Boundary/provenance failure: {where}")
                    with np.load(where / "predictions.npz", allow_pickle=False) as z:
                        proof = read(where / "prediction_manifest.json")
                        if (list(z["ids"]) != expected_pred or
                            list(z["patients"]) != list(n.loc[expected_pred].patient) or
                            proof["prediction_sha256"] != sha(where / "predictions.npz") or
                            proof["checkpoint_sha256"] != m["checkpoint_sha256"] or
                            proof["prediction_ids"] != expected_pred):
                            raise ValueError(f"Prediction ID mismatch: {where}")
        archive = folder / "predictions" / ("inner_oof.npz" if stage == "inner" else "outer_test.npz")
        with np.load(archive, allow_pickle=False) as z:
            if list(z["ids"]) != ids or list(z["patients"]) != list(n.loc[ids].patient) or tuple(z["concepts"]) != CONCEPTS:
                raise ValueError(f"Assembled alignment failure: {archive}")
            if list(z["seeds"]) != list(range(N_SEEDS)):
                raise ValueError("Seed coverage failure")
            for kind, shape in (("P", (len(ids), len(CONCEPTS), N_SEEDS)),
                                ("M", (len(ids), N_SEEDS))):
                for view in VIEWS:
                    a = z[f"{kind}_{view}"]
                    if a.shape != shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
                        raise ValueError(f"Invalid assembled {kind}_{view}")
    result = {"status": "PASS", "protocol": "strict_outer_original_seed0",
              "fit_clear_nodules": len(split["clear_fit_ids"]),
              "test_clear_nodules": len(split["clear_test_ids"]),
              "test_patients": len(test), "source_manifest_sha256": sha(folder / "source_manifest.json"),
              "outer_split_sha256": sha(folder / "outer_split.json")}
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        dump_new(folder / "boundary_verification.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train-inner", "predict-inner", "train-final", "predict-outer", "assemble", "verify-boundary"):
        p = sub.add_parser(command)
        p.add_argument("--run-id", required=True)
        p.add_argument("--dry-run", action="store_true")
        if command == "prepare":
            p.add_argument("--crops-glob", default="results_lidc/lidc_crops_*.npz")
        elif command.startswith(("train", "predict")):
            p.add_argument("--kind", choices=("finding", "malignancy"), required=True)
            if command.endswith("inner"):
                p.add_argument("--fold", type=int, required=True)
            p.add_argument("--seed", type=int, required=True)
            if command.startswith("train"):
                p.add_argument("--weights", type=Path)
        elif command == "assemble":
            p.add_argument("--stage", choices=("inner", "final"), required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "assemble":
            assemble(args)
        elif args.command == "verify-boundary":
            verify(args)
        else:
            task(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
