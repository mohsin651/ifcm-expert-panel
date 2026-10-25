"""Synthetic, CPU-only invariants for the proposed signed IFS operator."""

import unittest

import numpy as np

from ifcm.intuitionistic_reasoning import (
    IFS, finding_ifs_from_draws, from_interval, panel_edge_bounds,
    point_ifs, reason_b1, reason_b2,
)


class IntuitionisticReasoningTests(unittest.TestCase):
    def test_interval_mapping_and_point_limit(self):
        value = from_interval(np.array([[0.2, 0.0]]), np.array([[0.7, 1.0]]))
        np.testing.assert_allclose(value.mu, [[0.2, 0.0]])
        np.testing.assert_allclose(value.nu, [[0.3, 0.0]])
        np.testing.assert_allclose(value.pi, [[0.5, 1.0]])
        point = point_ifs(np.array([[0.4]]))
        np.testing.assert_allclose(point.mu, [[0.4]])
        np.testing.assert_allclose(point.nu, [[0.6]])
        np.testing.assert_array_equal(point.pi, [[0.0]])

    def test_invalid_ifs_and_interval_rejected(self):
        with self.assertRaises(ValueError):
            IFS(np.array([[0.8]]), np.array([[0.5]]), np.array([[-0.3]]))
        with self.assertRaises(ValueError):
            from_interval(np.array([[0.8]]), np.array([[0.2]]))
        with self.assertRaises(ValueError):
            from_interval(np.array([[np.nan]]), np.array([[1.0]]))

    def test_draw_envelope_is_seed_mean_aligned(self):
        draws = np.array([[[0.0, 0.0, 0.0, 0.0, 1.0]]])
        result = finding_ifs_from_draws(draws, mean=np.array([[0.2]]))
        self.assertLessEqual(result.lower[0, 0], 0.2)
        self.assertGreaterEqual(result.upper[0, 0], 0.2)
        self.assertGreater(result.pi[0, 0], 0)
        with self.assertRaisesRegex(ValueError, "seed mean"):
            finding_ifs_from_draws(draws, mean=np.array([[0.3]]))
        with self.assertRaisesRegex(ValueError, "quantiles"):
            finding_ifs_from_draws(draws, quantiles=(0.05, 0.95))

    def test_identical_draws_are_point_valued(self):
        result = finding_ifs_from_draws(np.full((2, 3, 5), 0.4))
        np.testing.assert_allclose(result.pi, 0, atol=1e-15)

    def test_float32_seed_mean_is_accepted(self):
        draws = np.array([[[0.1, 0.3, 0.5, 0.7, 0.9]]], dtype=np.float32)
        result = finding_ifs_from_draws(draws, draws.mean(axis=2))
        self.assertEqual(result.mu.shape, (1, 1))

    def test_panel_width_tracks_raw_and_clipped_values(self):
        edges = np.array([[[0.9]], [[0.9]]])
        unsure = np.array([[[0.4]], [[0.4]]])
        result = panel_edge_bounds(edges, unsure)
        np.testing.assert_allclose(result.raw_half_width, [[0.4]])
        np.testing.assert_allclose(result.lower, [[0.5]])
        np.testing.assert_allclose(result.upper, [[1.0]])
        np.testing.assert_allclose(result.left_width, [[0.4]])
        np.testing.assert_allclose(result.right_width, [[0.1]])

    def test_b2_zero_edge_uncertainty_equals_b1(self):
        x = from_interval(np.array([[0.2, 0.4], [0.1, 0.7]]),
                          np.array([[0.8, 0.9], [0.5, 0.9]]))
        w = np.array([[0.6, -0.4], [-0.3, 0.2]])
        a = reason_b1(x, w)
        b = reason_b2(x, w, w)
        for name in ("mu", "nu", "pi"):
            np.testing.assert_array_equal(getattr(a.diagnosis, name),
                                          getattr(b.diagnosis, name))
        np.testing.assert_array_equal(a.score, b.score)

    def test_positive_and_negative_toy_case(self):
        result = reason_b1(point_ifs(np.array([[0.8, 0.5]])),
                           np.array([[0.6], [-0.4]]))
        np.testing.assert_allclose(result.diagnosis.mu, [[0.384]])
        np.testing.assert_allclose(result.diagnosis.nu, [[0.104]])
        np.testing.assert_allclose(result.diagnosis.pi, [[0.512]])
        np.testing.assert_allclose(result.score, [[0.28]])

    def test_knowledge_uncertainty_expands_hesitancy(self):
        x = from_interval(np.array([[0.6, 0.6]]), np.array([[0.9, 0.9]]))
        point = reason_b1(x, np.array([[0.6], [-0.4]]))
        uncertain = reason_b2(x, np.array([[0.5], [-0.5]]),
                              np.array([[0.7], [-0.3]]))
        self.assertLessEqual(uncertain.diagnosis.mu[0, 0], point.diagnosis.mu[0, 0])
        self.assertLessEqual(uncertain.diagnosis.nu[0, 0], point.diagnosis.nu[0, 0])
        self.assertGreaterEqual(uncertain.diagnosis.pi[0, 0], point.diagnosis.pi[0, 0])
        np.testing.assert_allclose(uncertain.diagnosis.mu, [[0.165]])
        np.testing.assert_allclose(uncertain.diagnosis.nu, [[0.0666]])

    def test_neutral_edge_absent_finding_and_max_hesitancy(self):
        neutral = reason_b1(point_ifs(np.array([[1.0]])), np.array([[0.0]]))
        absent = reason_b1(point_ifs(np.array([[0.0]])), np.array([[1.0]]))
        unknown = reason_b1(from_interval(np.array([[0.0]]), np.array([[1.0]])),
                            np.array([[1.0]]))
        for value in (neutral, absent, unknown):
            np.testing.assert_allclose(value.diagnosis.mu, [[0]])
            np.testing.assert_allclose(value.diagnosis.nu, [[0]])
            np.testing.assert_allclose(value.diagnosis.pi, [[1]])

    def test_opposite_evidence_becomes_conflict(self):
        result = reason_b1(point_ifs(np.array([[1.0, 1.0]])),
                           np.array([[1.0], [-1.0]]))
        np.testing.assert_allclose(result.diagnosis.mu, [[0]])
        np.testing.assert_allclose(result.diagnosis.nu, [[0]])
        np.testing.assert_allclose(result.diagnosis.pi, [[1]])

    def test_sign_reversal_swaps_support_and_opposition(self):
        x = point_ifs(np.array([[0.6, 0.9]]))
        weights = np.array([[0.5], [-0.3]])
        positive = reason_b1(x, weights)
        reversed_ = reason_b1(x, -weights)
        np.testing.assert_allclose(positive.diagnosis.mu, reversed_.diagnosis.nu)
        np.testing.assert_allclose(positive.diagnosis.nu, reversed_.diagnosis.mu)
        np.testing.assert_allclose(positive.score, -reversed_.score)

    def test_point_edge_monotonicity(self):
        x = point_ifs(np.array([[0.8]]))
        a = reason_b1(x, np.array([[0.2]])).score[0, 0]
        b = reason_b1(x, np.array([[0.7]])).score[0, 0]
        c = reason_b1(x, np.array([[-0.2]])).score[0, 0]
        d = reason_b1(x, np.array([[-0.7]])).score[0, 0]
        self.assertGreater(b, a)
        self.assertLess(d, c)

    def test_finite_saturated_aggregation(self):
        result = reason_b1(point_ifs(np.ones((3, 100))), np.ones((100, 2)))
        np.testing.assert_allclose(result.diagnosis.mu, 1)
        np.testing.assert_allclose(result.diagnosis.nu, 0)
        np.testing.assert_allclose(result.diagnosis.pi, 0)
        self.assertTrue(np.isfinite(result.score).all())

    def test_nested_uncertainty_never_lowers_hesitancy(self):
        rng = np.random.default_rng(17)
        x = rng.uniform(0.15, 0.85, (20, 7))
        w = rng.uniform(-0.7, 0.7, (7, 3))
        narrow = reason_b2(
            from_interval(x - 0.02, x + 0.02), w - 0.02, w + 0.02)
        wide = reason_b2(
            from_interval(x - 0.12, x + 0.12), w - 0.12, w + 0.12)
        self.assertTrue(np.all(wide.diagnosis.mu <= narrow.diagnosis.mu + 1e-12))
        self.assertTrue(np.all(wide.diagnosis.nu <= narrow.diagnosis.nu + 1e-12))
        self.assertTrue(np.all(wide.diagnosis.pi >= narrow.diagnosis.pi - 1e-12))

    def test_shape_nan_missing_and_interval_order_guards(self):
        x = point_ifs(np.array([[0.5, 0.3]]))
        with self.assertRaisesRegex(ValueError, "align"):
            reason_b1(x, np.array([[0.5]]))
        with self.assertRaises(ValueError):
            reason_b1(x, np.array([[np.nan], [0.1]]))
        with self.assertRaisesRegex(ValueError, "lower endpoint"):
            reason_b2(x, np.array([[0.6], [0.2]]), np.array([[0.4], [0.3]]))
        with self.assertRaises(ValueError):
            panel_edge_bounds(np.ones((2, 2, 1)), np.ones((2, 1, 1)))


if __name__ == "__main__":
    unittest.main()
