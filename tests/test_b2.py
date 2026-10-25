"""Synthetic B2 invariants; no medical data or inference."""

import unittest

import numpy as np

from ifcm.intuitionistic_reasoning import finding_ifs_from_draws, panel_edge_bounds, reason_b1, reason_b2


class B2Tests(unittest.TestCase):
    def setUp(self):
        self.findings = finding_ifs_from_draws(np.array([[[0.4, 0.5, 0.6]]]))

    def test_agreeing_members_have_narrower_bounds_than_conflicting_members(self):
        agree = panel_edge_bounds(np.array([[[0.5]], [[0.6]]]), np.zeros((2, 1, 1)))
        conflict = panel_edge_bounds(np.array([[[-0.6]], [[0.6]]]), np.zeros((2, 1, 1)))
        self.assertLess(agree.raw_half_width[0, 0], conflict.raw_half_width[0, 0])
        self.assertGreater(reason_b2(self.findings, agree.lower, agree.upper).diagnosis.mu[0, 0], 0)
        self.assertGreaterEqual(reason_b2(self.findings, conflict.lower, conflict.upper).diagnosis.pi[0, 0],
                                reason_b2(self.findings, agree.lower, agree.upper).diagnosis.pi[0, 0])

    def test_zero_edge_width_exactly_reduces_to_b1(self):
        weight = np.array([[0.3]])
        b1 = reason_b1(self.findings, weight)
        b2 = reason_b2(self.findings, weight, weight)
        np.testing.assert_array_equal(b1.score, b2.score)
        np.testing.assert_array_equal(b1.diagnosis.pi, b2.diagnosis.pi)


if __name__ == "__main__":
    unittest.main()
