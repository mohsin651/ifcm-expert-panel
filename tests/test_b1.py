"""Small synthetic checks for the Issue 3 B1 adapter; no research jobs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ifcm.b1 import _check_archive, apply_calibration, fit_calibration, prediction_arrays, run_dir
from ifcm.b1_inputs import DERM_CONCEPTS, Rows, _style_weights, derm_probe_draws


class PanelTests(unittest.TestCase):
    def test_direct_and_choice_weights_preserve_sign_and_alignment(self):
        models = ("model-a", "model-b")
        concepts = ("c1", "c2")
        diagnoses = ("MAL", "BEN")
        direct, choice = [], []
        for m in models:
            for c in concepts:
                for template in range(3):
                    for order in range(3):
                        choice.append({"model": m, "concept": c, "template": template,
                                       "order": order, "p_MAL": 0.7, "p_BEN": 0.2,
                                       "p_unsure": 0.1})
                        for d in diagnoses:
                            direct.append({"model": m, "concept": c, "diag": d,
                                           "template": template, "order": order,
                                           "p_up": 0.8 if d == "MAL" else 0.1,
                                           "p_down": 0.1 if d == "MAL" else 0.8})
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            pd.DataFrame(direct).to_csv(base / "direct.csv", index=False)
            pd.DataFrame(choice).to_csv(base / "choice.csv", index=False)
            d = _style_weights((base / "direct.csv",), "comp", models, concepts, diagnoses)
            q = _style_weights((base / "choice.csv",), "choice", models, concepts, diagnoses)
            self.assertAlmostEqual(d.loc[("model-a|comp", "c1", "MAL")], 0.7)
            self.assertAlmostEqual(d.loc[("model-a|comp", "c1", "BEN")], -0.7)
            self.assertAlmostEqual(q.loc[("model-b|choice", "c2", "MAL")],
                                   2 * (0.7 / (0.9 + 1e-6)) - 1)
            self.assertLess(q.loc[("model-b|choice", "c2", "BEN")], 0)
            broken = pd.DataFrame(choice).iloc[:-1]
            broken.to_csv(base / "broken.csv", index=False)
            with self.assertRaisesRegex(ValueError, "9 prompts"):
                _style_weights((base / "broken.csv",), "choice", models, concepts, diagnoses)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        ids = np.array(["n1", "n2", "n3", "n4"])
        patients = np.array(["p1", "p1", "p2", "p3"])
        draws = np.array([
            [[0.2, 0.3, 0.4], [0.8, 0.7, 0.9]],
            [[0.7, 0.8, 0.6], [0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
            [[0.9, 0.8, 0.7], [0.2, 0.1, 0.3]],
        ])
        self.rows = Rows(ids, patients, draws, np.array([0, 0, 1, 2]), ("c1", "c2"))
        self.edge = np.array([[0.6, -0.5], [-0.4, 0.7]])
        self.calibration = {"slope": [1.2, 0.8], "intercept": [-0.1, 0.2]}

    def test_fit_calibration_has_one_slope_and_intercept_per_diagnosis(self):
        scores = np.array([[-0.6, 0.6], [-0.2, 0.2], [0.2, -0.2], [0.6, -0.6]])
        labels = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        cal = fit_calibration(scores, labels)
        self.assertEqual(cal["n_fit"], 4)
        self.assertEqual(len(cal["slope"]), 2)
        self.assertEqual(cal["positives"], [2, 2])
        with self.assertRaisesRegex(ValueError, "both calibration classes"):
            fit_calibration(scores, np.zeros_like(labels))

    def test_prediction_archive_recomputes_and_rejects_row_mismatch(self):
        arrays = prediction_arrays(self.rows, self.edge, self.calibration, "lidc", "test")
        self.assertEqual(arrays["finding_mu"].shape, (4, 2))
        self.assertEqual(arrays["diagnosis_pi"].shape, (4, 2))
        self.assertTrue(np.all(arrays["finding_pi"] > 0))
        self.assertTrue(np.array_equal(arrays["split"], ["test"] * 4))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pred.npz"
            np.savez_compressed(path, **arrays)
            kwargs = (list(self.rows.ids), list(self.rows.patients), list(self.rows.concepts),
                      ["MAL", "BEN"], self.calibration, self.edge, "lidc", "test", self.rows.fold)
            _check_archive(path, *kwargs)
            with self.assertRaisesRegex(ValueError, "alignment"):
                _check_archive(path, ["wrong", *list(self.rows.ids[1:])], *kwargs[1:])

    def test_lidc_normalization_is_distinct_from_raw_malignancy_sigmoid(self):
        p, decision, p_mal = apply_calibration(np.array([[0.3, -0.2]]), self.calibration, "lidc")
        self.assertAlmostEqual(p_mal[0], p[0, 0] / p[0].sum())
        self.assertEqual(decision[0], int(p_mal[0] > 0.5))

    def test_run_id_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):
            run_dir("../outside")

    def test_inactive_derm_bootstrap_probe_returns_zero(self):
        coef = np.zeros((len(DERM_CONCEPTS), 30, 2))
        intercept = np.zeros((len(DERM_CONCEPTS), 30))
        active = np.ones((len(DERM_CONCEPTS), 30), bool)
        active[0, 0] = False
        draws = derm_probe_draws(np.array([[0.1, 0.2]]),
                                 {"coef": coef, "intercept": intercept, "active": active})
        self.assertEqual(draws.shape, (1, len(DERM_CONCEPTS), 30))
        self.assertEqual(draws[0, 0, 0], 0)
        self.assertEqual(draws[0, 0, 1], 0.5)


if __name__ == "__main__":
    unittest.main()
