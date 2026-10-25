"""Small CPU-only check that saved prediction IDs never require pickle."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


class PredictionIoTests(unittest.TestCase):
    def test_patient_ids_are_plain_strings(self):
        import torch

        from ifcm.safe_cnn import CONCEPTS, VIEWS
        from ifcm.safe_cnn_model import predict_model

        def tiny_model(n_out):
            return torch.nn.Sequential(torch.nn.Flatten(),
                                       torch.nn.Linear(3 * 96 * 96, n_out))

        ids = ["n1", "n2"]
        ratings = pd.DataFrame({"patient": pd.Series(["p1", "p2"], index=ids, dtype=object)})
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            output = Path(temporary) / "predictions.npz"
            torch.save(tiny_model(len(CONCEPTS)).state_dict(), checkpoint)
            with (patch("ifcm.safe_cnn_model.network", side_effect=tiny_model),
                  patch("ifcm.safe_cnn_model.images", return_value=torch.zeros(2, 3, 96, 96)),
                  patch("torch.cuda.is_available", return_value=False)):
                predict_model({}, ratings, ids, "finding", checkpoint, output)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(list(archive["ids"]), ids)
                self.assertEqual(list(archive["patients"]), ["p1", "p2"])
                for name in VIEWS:
                    self.assertEqual(archive[name].shape, (2, len(CONCEPTS)))


if __name__ == "__main__":
    unittest.main()
