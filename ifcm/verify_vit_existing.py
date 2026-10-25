"""Repeatable, read-only integrity check for a completed ViT LIDC run."""

import argparse
import json

import numpy as np

from ifcm import safe_cnn as c
from ifcm.safe_vit import variant_of, where_for


def check(run_id):
    folder = c.run_dir(run_id)
    source, split, ratings = c.inputs(folder)
    variant = variant_of(source)
    boundary = c.read(folder / "boundary_verification.json")
    expected = {"status": "PASS", "protocol": source["protocol"],
                "fit_clear_nodules": len(split["clear_fit_ids"]),
                "test_clear_nodules": len(split["clear_test_ids"]),
                "test_patients": len(split["test_patients"]),
                "source_manifest_sha256": c.sha(folder / "source_manifest.json"),
                "outer_split_sha256": c.sha(folder / "outer_split.json")}
    if boundary != expected:
        raise ValueError("Saved boundary proof does not match current inputs")
    if c.sha(folder / "outer_split.json") != source["reference_split_sha256"]:
        raise ValueError("ViT patient split differs from frozen reference")
    for stage in ("inner", "final"):
        ids = split["fit_ids"] if stage == "inner" else split["test_ids"]
        archive = folder / "predictions" / ("inner_oof.npz" if stage == "inner" else "outer_test.npz")
        with np.load(archive, allow_pickle=False) as z:
            if (list(z["ids"]) != ids or list(z["patients"]) != list(ratings.loc[ids].patient) or
                    tuple(z["concepts"]) != c.CONCEPTS or list(z["seeds"]) != list(range(c.N_SEEDS)) or
                    z["P_tta0"].shape != (len(ids), len(c.CONCEPTS), c.N_SEEDS)):
                raise ValueError(f"Assembled alignment mismatch: {archive}")
            joined = z["P_tta0"].copy()
        lookup = {id_: j for j, id_ in enumerate(ids)}
        for fold in (range(c.N_FOLDS) if stage == "inner" else [None]):
            for seed in range(c.N_SEEDS):
                where = where_for(folder, stage, fold, seed)
                manifest = c.read(where / "manifest.json")
                train, pred = c.task_ids(split, ratings, stage, fold)
                c.validate_task_manifest(manifest, train, pred, ratings,
                                         expected["outer_split_sha256"],
                                         expected["source_manifest_sha256"])
                if c.sha(where / "checkpoint.pt") != manifest["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint changed: {where}")
                proof = c.read(where / "prediction_manifest.json")
                if (c.sha(where / "predictions.npz") != proof["prediction_sha256"] or
                        proof["checkpoint_sha256"] != manifest["checkpoint_sha256"] or
                        proof["prediction_ids"] != pred):
                    raise ValueError(f"Prediction shard changed: {where}")
                with np.load(where / "predictions.npz", allow_pickle=False) as z:
                    if (list(z["ids"]) != pred or list(z["patients"]) != list(ratings.loc[pred].patient) or
                            z["tta0"].shape != (len(pred), len(c.CONCEPTS)) or
                            not np.array_equal(joined[[lookup[i] for i in pred], :, seed], z["tta0"])):
                        raise ValueError(f"Assembled values differ from shard: {where}")
        if not np.isfinite(joined).all() or np.any((joined < 0) | (joined > 1)):
            raise ValueError(f"Invalid assembled values: {archive}")
    print(json.dumps({"status": "PASS", "run_id": run_id, "variant": variant,
                      "inner_rows": len(split["fit_ids"]), "test_rows": len(split["test_ids"]),
                      "test_patients": len(split["test_patients"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    check(args.run_id)


if __name__ == "__main__":
    main()
