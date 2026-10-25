"""Separate, patient-safe ViT-Tiny/Small LIDC finding runs for A/B1/B2.

Usage: python -m ifcm.safe_vit prepare --run-id NAME --variant tiny
       python -m ifcm.safe_vit all --run-id NAME
Existing CNN and LLM outputs are read-only. Every output lives under NAME.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ifcm import safe_cnn as c
from ifcm.vit_model import MODELS

REFERENCE = "isbi-lidc-strict-v3"


def prepare(run_id: str, variant: str):
    folder = c.run_dir(run_id)
    if folder.exists():
        raise FileExistsError(folder)
    reference = c.run_dir(REFERENCE)
    source, split, _ = c.inputs(reference)
    c.validate_split(source, split, c.ratings_table())
    source = dict(source)
    source["architecture"] = MODELS[variant]
    source["views"] = ["tta0"]
    source["historical_cnn_predictions_used"] = False
    source["reference_split"] = str(reference / "outer_split.json")
    source["reference_split_sha256"] = c.sha(reference / "outer_split.json")
    source["code_hashes"] = {
        **source["code_hashes"],
        str(Path(__file__).resolve()): c.sha(Path(__file__)),
        str(Path(__file__).with_name("vit_model.py").resolve()):
            c.sha(Path(__file__).with_name("vit_model.py")),
    }
    folder.mkdir(parents=True, exist_ok=False)
    c.dump_new(folder / "source_manifest.json", source)
    c.dump_new(folder / "outer_split.json", split)
    c.inputs(folder)
    print(json.dumps({"run": str(folder), "variant": variant,
                      "counts": split["counts"]}, indent=2))


def variant_of(source):
    matches = [key for key, name in MODELS.items() if name == source.get("architecture")]
    if len(matches) != 1:
        raise ValueError("Run is not a supported ViT variant")
    return matches[0]


def where_for(folder, stage, fold, seed):
    return c.task_path(folder, stage, "finding", fold, seed)


def task(folder, stage, fold, seed):
    source, split, ratings = c.inputs(folder)
    variant = variant_of(source)
    train, pred = c.task_ids(split, ratings, stage, fold)
    where = where_for(folder, stage, fold, seed)
    split_hash = c.sha(folder / "outer_split.json")
    source_hash = c.sha(folder / "source_manifest.json")
    if where.exists():
        manifest = c.read(where / "manifest.json")
        c.validate_task_manifest(manifest, train, pred, ratings, split_hash, source_hash)
        if c.sha(where / "checkpoint.pt") != manifest["checkpoint_sha256"]:
            raise ValueError(f"Changed checkpoint: {where}")
    else:
        from ifcm.vit_model import train_model
        where.mkdir(parents=True, exist_ok=False)
        details = train_model(source, ratings, train, variant, seed, fold, where)
        manifest = {"stage": stage, "kind": "finding", "fold": fold, "seed": seed,
                    "train_ids": train, "train_patients": sorted(set(ratings.loc[train].patient)),
                    "prediction_ids": pred,
                    "prediction_patients": sorted(set(ratings.loc[pred].patient)),
                    "outer_split_sha256": split_hash, "source_manifest_sha256": source_hash,
                    **details}
        c.dump_new(where / "manifest.json", manifest)
    output = where / "predictions.npz"
    if output.exists():
        proof = c.read(where / "prediction_manifest.json")
        if proof["prediction_sha256"] != c.sha(output):
            raise ValueError(f"Changed predictions: {output}")
        return
    from ifcm.vit_model import predict_model
    predict_model(source, ratings, pred, variant, where / "checkpoint.pt", output)
    c.dump_new(where / "prediction_manifest.json", {
        "prediction_sha256": c.sha(output), "checkpoint_sha256": manifest["checkpoint_sha256"],
        "prediction_ids": pred, "prediction_patients": sorted(set(ratings.loc[pred].patient)),
    })


def assemble(folder, stage):
    source, split, ratings = c.inputs(folder)
    variant_of(source)
    ids = split["fit_ids"] if stage == "inner" else split["test_ids"]
    output = folder / "predictions" / ("inner_oof.npz" if stage == "inner" else "outer_test.npz")
    if output.exists():
        return
    draws = np.full((len(ids), len(c.CONCEPTS), c.N_SEEDS), np.nan, dtype=np.float32)
    lookup = {id_: i for i, id_ in enumerate(ids)}
    for fold in (range(c.N_FOLDS) if stage == "inner" else [None]):
        for seed in range(c.N_SEEDS):
            where = where_for(folder, stage, fold, seed)
            manifest = c.read(where / "manifest.json")
            train, pred = c.task_ids(split, ratings, stage, fold)
            c.validate_task_manifest(manifest, train, pred, ratings,
                                     c.sha(folder / "outer_split.json"),
                                     c.sha(folder / "source_manifest.json"))
            proof = c.read(where / "prediction_manifest.json")
            if (proof["prediction_sha256"] != c.sha(where / "predictions.npz") or
                    proof["checkpoint_sha256"] != manifest["checkpoint_sha256"]):
                raise ValueError(f"Changed shard: {where}")
            with np.load(where / "predictions.npz", allow_pickle=False) as z:
                if (list(z["ids"]) != pred or
                        list(z["patients"]) != list(ratings.loc[pred].patient) or
                        z["tta0"].shape != (len(pred), len(c.CONCEPTS))):
                    raise ValueError(f"Invalid prediction IDs/shape: {where}")
                if not np.isfinite(z["tta0"]).all() or np.any((z["tta0"] < 0) | (z["tta0"] > 1)):
                    raise ValueError(f"Invalid prediction range: {where}")
                for row, id_ in enumerate(pred):
                    index = lookup[id_]
                    if np.isfinite(draws[index, :, seed]).any():
                        raise ValueError(f"Duplicate prediction: {id_}/{seed}")
                    draws[index, :, seed] = z["tta0"][row]
    if not np.isfinite(draws).all():
        raise ValueError("Incomplete prediction coverage")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=np.array(ids),
                            patients=ratings.loc[ids].patient.to_numpy(dtype=str),
                            concepts=np.array(c.CONCEPTS),
                            fold=np.array([split["inner_fold_by_patient"].get(ratings.at[i, "patient"], -1)
                                           for i in ids]), seeds=np.arange(c.N_SEEDS), P_tta0=draws)
    print(f"Assembled {stage}: {draws.shape} -> {output}", flush=True)


def verify(folder):
    source, split, ratings = c.inputs(folder)
    variant_of(source)
    if c.sha(Path(source["reference_split"])) != source["reference_split_sha256"]:
        raise ValueError("Reference split changed")
    if c.sha(folder / "outer_split.json") != source["reference_split_sha256"]:
        raise ValueError("ViT split differs from verified CNN split")
    for stage in ("inner", "final"):
        ids = split["fit_ids"] if stage == "inner" else split["test_ids"]
        for fold in (range(c.N_FOLDS) if stage == "inner" else [None]):
            for seed in range(c.N_SEEDS):
                where = where_for(folder, stage, fold, seed)
                manifest = c.read(where / "manifest.json")
                train, pred = c.task_ids(split, ratings, stage, fold)
                c.validate_task_manifest(manifest, train, pred, ratings,
                                         c.sha(folder / "outer_split.json"),
                                         c.sha(folder / "source_manifest.json"))
                if c.sha(where / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                    raise ValueError(f"Changed checkpoint: {where}")
                proof = c.read(where / "prediction_manifest.json")
                if (proof["prediction_sha256"] != c.sha(where / "predictions.npz") or
                        proof["prediction_ids"] != pred):
                    raise ValueError(f"Changed shard: {where}")
        archive = folder / "predictions" / ("inner_oof.npz" if stage == "inner" else "outer_test.npz")
        with np.load(archive, allow_pickle=False) as z:
            if (list(z["ids"]) != ids or list(z["patients"]) != list(ratings.loc[ids].patient) or
                    tuple(z["concepts"]) != c.CONCEPTS or list(z["seeds"]) != list(range(c.N_SEEDS)) or
                    z["P_tta0"].shape != (len(ids), len(c.CONCEPTS), c.N_SEEDS) or
                    not np.isfinite(z["P_tta0"]).all()):
                raise ValueError(f"Invalid assembled predictions: {archive}")
    result = {"status": "PASS", "protocol": source["protocol"],
              "fit_clear_nodules": len(split["clear_fit_ids"]),
              "test_clear_nodules": len(split["clear_test_ids"]),
              "test_patients": len(split["test_patients"]),
              "source_manifest_sha256": c.sha(folder / "source_manifest.json"),
              "outer_split_sha256": c.sha(folder / "outer_split.json")}
    c.dump_new(folder / "boundary_verification.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "all", "verify"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=tuple(MODELS))
    args = parser.parse_args()
    if args.command == "prepare":
        if args.variant is None:
            parser.error("prepare requires --variant")
        prepare(args.run_id, args.variant)
    elif args.command == "all":
        folder = c.run_dir(args.run_id)
        for fold in range(c.N_FOLDS):
            for seed in range(c.N_SEEDS):
                task(folder, "inner", fold, seed)
        assemble(folder, "inner")
        for seed in range(c.N_SEEDS):
            task(folder, "final", None, seed)
        assemble(folder, "final")
        verify(folder)
    else:
        verify(c.run_dir(args.run_id))


if __name__ == "__main__":
    main()
