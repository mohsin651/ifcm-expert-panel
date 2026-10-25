"""CPU-only tests for the CNN ImageNet checkpoint compatibility check."""

import unittest
from pathlib import Path
from unittest.mock import patch


class ResNetWeightsTests(unittest.TestCase):
    def test_older_official_checkpoint_without_bn_counters_loads(self):
        import torchvision

        from ifcm.safe_cnn_model import network

        state = torchvision.models.resnet18(weights=None).state_dict()
        old_state = {key: value for key, value in state.items()
                     if not key.endswith(".num_batches_tracked")}
        with patch("torch.load", return_value=old_state):
            model = network(14, Path("unused.pth"))
        self.assertEqual(model.fc.out_features, 14)

    def test_missing_required_weight_is_rejected(self):
        import torchvision

        from ifcm.safe_cnn_model import network

        state = torchvision.models.resnet18(weights=None).state_dict()
        state.pop("conv1.weight")
        with patch("torch.load", return_value=state):
            with self.assertRaisesRegex(ValueError, "do not match"):
                network(14, Path("unused.pth"))


if __name__ == "__main__":
    unittest.main()
