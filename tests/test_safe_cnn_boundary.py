"""Small synthetic checks; never load torch, crops, or research results."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ifcm.safe_cnn import (assert_boundary, crop_index, inner_folds, patient_split,
                           run_dir, sha, task_ids, validate_task_manifest)


class SplitBoundaryTests(unittest.TestCase):
    def test_deterministic_patient_split_and_inner_coverage(self):
        patients = np.array([f"p{i:02d}" for i in range(25)])
        a, b = patient_split(patients, 0)
        self.assertEqual((len(a), len(b)), (15, 10))
        self.assertEqual((a, b), patient_split(patients, 0))
        folds = inner_folds(a, 17)
        assert_boundary(set(a), set(b), folds)
        self.assertEqual(set(folds), set(a))
        self.assertEqual(set(folds.values()), set(range(5)))

    def test_outer_test_training_contamination_rejected(self):
        fit = {f"p{i}" for i in range(5)}
        test = {"heldout"}
        folds = {f"p{i}": i for i in range(5)}
        with self.assertRaisesRegex(ValueError, "outer-test"):
            assert_boundary(fit, test, folds, train={"p0", "heldout"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_boundary(fit, test, folds, train={"p0"}, predicted={"p0"})

    def test_task_ids_preserve_patient_boundary(self):
        fit = [f"p{i}" for i in range(5)]
        test = ["heldout"]
        split = {"fit_patients": fit, "test_patients": test,
                 "inner_fold_by_patient": {p: i for i, p in enumerate(fit)},
                 "fit_ids": [f"n{i}" for i in range(5)], "test_ids": ["n5"]}
        n = pd.DataFrame({"patient": fit + test}, index=split["fit_ids"] + split["test_ids"])
        inner_train, inner_pred = task_ids(split, n, "inner", 2)
        self.assertEqual(inner_pred, ["n2"])
        self.assertNotIn("n2", inner_train)
        final_train, outer_pred = task_ids(split, n, "final", None)
        self.assertEqual((final_train, outer_pred), (split["fit_ids"], ["n5"]))

    def test_unique_crop_ids_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a.npz", Path(temporary) / "b.npz"
            np.savez_compressed(a, n1=np.zeros((4, 4, 5)))
            np.savez_compressed(b, n1=np.zeros((4, 4, 5)))
            self.assertEqual(crop_index([a]), {"n1": a})
            self.assertEqual(sha(a), sha(a))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                crop_index([a, b])

    def test_manifest_rejects_wrong_training_patient(self):
        n = pd.DataFrame({"patient": ["fit", "heldout"]}, index=["n0", "n1"])
        manifest = {"train_ids": ["n0"], "prediction_ids": ["n1"],
                    "train_patients": ["fit"], "prediction_patients": ["heldout"],
                    "outer_split_sha256": "split", "source_manifest_sha256": "source"}
        validate_task_manifest(manifest, ["n0"], ["n1"], n, "split", "source")
        manifest["train_patients"] = ["heldout"]
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_task_manifest(manifest, ["n0"], ["n1"], n, "split", "source")

    def test_run_id_rejects_paths(self):
        with self.assertRaises(ValueError):
            run_dir("../escape")
        self.assertEqual(run_dir("safe-01").name, "cnn")


if __name__ == "__main__":
    unittest.main()
