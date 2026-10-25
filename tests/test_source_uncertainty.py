import unittest

import numpy as np

from ifcm.intuitionistic_reasoning import from_interval
from ifcm.source_uncertainty import decompose


class SourceUncertaintyTests(unittest.TestCase):
    def test_additive_decomposition_and_monotone_increments(self):
        mean = np.array([[0.5, 0.7], [0.2, 0.9]])
        finding = from_interval(mean - 0.1, np.minimum(mean + 0.1, 1))
        edge = np.array([[0.6, -0.3], [-0.4, 0.8]])
        result = decompose(mean, finding, edge, edge - 0.1, edge + 0.1)
        self.assertTrue(np.allclose(result["neither"] + result["image_added"] +
                                    result["edge_added"], result["both"]))
        self.assertTrue((result["image_added"] >= -1e-10).all())
        self.assertTrue((result["edge_added"] >= -1e-10).all())

    def test_zero_edge_width_adds_no_edge_hesitation(self):
        mean = np.array([[0.4, 0.6]])
        finding = from_interval(np.array([[0.2, 0.5]]), np.array([[0.7, 0.8]]))
        edge = np.array([[0.8], [-0.4]])
        result = decompose(mean, finding, edge, edge, edge)
        self.assertTrue(np.allclose(result["edge_added"], 0))
        self.assertTrue(np.allclose(result["both"], result["image_only"]))


if __name__ == "__main__":
    unittest.main()
