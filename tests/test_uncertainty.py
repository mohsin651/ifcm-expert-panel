import unittest

import numpy as np

from ifcm.uncertainty import exact_flip_probabilities


class ExactFlipTests(unittest.TestCase):
    def test_two_finding_and_rule(self):
        # P(AND is true) = p1*p2, including deterministic endpoints.
        x = np.array([[0.2, 0.7], [1.0, 0.4], [0.0, 1.0]])
        rule = lambda values: (values[:, 0] + values[:, 1] > 1.5).astype(int)
        result = exact_flip_probabilities(x, rule, baseline=np.zeros(len(x), dtype=int))
        np.testing.assert_allclose(result, [0.14, 0.4, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
