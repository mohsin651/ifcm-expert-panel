"""Small synthetic checks for comparison, inference and paper utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ifcm.b1 import apply_calibration
from ifcm.comparison import _predict
from ifcm.evaluate_comparison import _attribution_pairs, _bootstrap_pairs


class ComparisonTests(unittest.TestCase):
    def test_model_a_uses_exact_matrix_product_and_fixed_sigmoids(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "b1.npz"
            x = np.array([[0.2, 0.8], [0.7, 0.1]])
            edge = np.array([[0.6, -0.5], [-0.4, 0.8]])
            np.savez_compressed(source, ids=["n1", "n2"], patients=["p1", "p2"],
                                split=["test", "test"], fold=[-1, -1],
                                concepts=["c1", "c2"], diagnoses=["MAL", "BEN"],
                                finding_mean=x)
            cal = {"slope": [0.8, 1.2], "intercept": [-0.1, 0.2]}
            result = _predict(source, edge, cal, "lidc")
            np.testing.assert_array_equal(result["raw_score"], x @ edge)
            expected = apply_calibration(x @ edge, cal, "lidc")
            np.testing.assert_array_equal(result["calibrated"], expected[0])
            np.testing.assert_array_equal(result["decision"], expected[1])

    def test_grouped_bootstrap_uses_patient_groups(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        scores = {"a": np.array([0.2, 0.8, 0.4, 0.7, 0.1, 0.9]),
                  "b1": np.array([0.3, 0.7, 0.5, 0.6, 0.2, 0.8]),
                  "b2": np.array([0.4, 0.6, 0.6, 0.5, 0.3, 0.7])}
        groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
        pairs = _bootstrap_pairs(y, scores, groups, np.ones(6, bool), 50, 2, "malignancy")
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["bootstrap_unit"], "patient")
        self.assertEqual(pairs[0]["n_groups"], 3)
        self.assertEqual(pairs[0]["valid_bootstrap"], 50)

    def test_attribution_bootstrap_rejects_one_class_resamples(self):
        labels = np.array([1, 0, 1, 0])
        groups = np.array(["p1", "p2", "p3", "p4"])
        scores = {"first": np.array([0.8, 0.2, 0.7, 0.3]),
                  "second": np.array([0.6, 0.4, 0.55, 0.45])}
        rows = _attribution_pairs(labels, scores, groups, np.ones(4, bool), 50, 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["valid_bootstrap"] + rows[0]["invalid_bootstrap"], 50)

    def test_latex_table_does_not_require_jinja(self):
        try:
            from ifcm.paper_outputs import _latex_table
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                self.skipTest("Matplotlib is optional in the GPU-only environment")
            raise
        latex = _latex_table(pd.DataFrame({"model_name": ["A&B"], "auroc": [0.85]}))
        self.assertIn(r"model\_name", latex)
        self.assertIn(r"A\&B", latex)
        self.assertIn("0.8500", latex)


if __name__ == "__main__":
    unittest.main()
