"""Regression checks for the bridge-sufficient observability score in Eq. (6)."""
import math
import unittest

import torch

from obsalign.config import Config
from obsalign.observability import estimate
from obsalign.sketch import Sketch


def source(visible, reliability=1.0, count=10):
    """One semantic unit with matching observed modality prototypes."""
    visible = torch.tensor(visible, dtype=torch.bool)
    present = visible[:, None].double()
    mean = torch.zeros(len(visible), 1, 2, dtype=torch.float64)
    mean[:, 0, 0] = visible.double()
    return Sketch(count * present, mean, torch.zeros_like(present),
                  reliability * present, visible)


class ObservabilityTests(unittest.TestCase):
    def test_supported_sides_without_bridge_have_zero_score_and_gate(self):
        observation = estimate([source([True, False]), source([False, True])], Config())
        self.assertTrue(torch.all(observation.support > 0))
        self.assertEqual(observation.direct[0, 1, 0].item(), 0.0)
        self.assertEqual(observation.paths[0, 1, 0].item(), 0.0)
        self.assertEqual(observation.score[0, 1, 0].item(), 0.0)
        self.assertEqual(observation.gate[0, 1, 0].item(), 0.0)

    def test_direct_bridge_matches_closed_form_with_configured_coefficients(self):
        for coefficients in ({}, {"alpha": 0.3, "beta": 0.7, "eta": 0.2,
                                  "delta": 0.4, "b0": 0.6}):
            with self.subTest(coefficients=coefficients):
                cfg = Config(max_path_length=1, **coefficients)
                observation = estimate([source([True, True])], cfg)
                # D=1, P=0, Cons=1, Supp=log(11), Unc=1/2.
                logit = (cfg.alpha * math.log(2) + cfg.beta
                         + cfg.eta * math.log(11) - cfg.delta / 2 - cfg.b0)
                expected = 0.5 / (1 + math.exp(-logit))
                self.assertAlmostEqual(observation.score[0, 1, 0].item(), expected, places=6)
                expected_gate = expected * min(1.0, max(
                    0.0, (expected - cfg.tau_low) / (cfg.tau_high - cfg.tau_low)))
                self.assertAlmostEqual(observation.gate[0, 1, 0].item(), expected_gate, places=6)

    def test_path_only_bridge_remains_observable(self):
        cfg = Config(max_path_length=2)
        observation = estimate([source([True, True, False]),
                                source([False, True, True])], cfg)
        self.assertEqual(observation.direct[0, 2, 0].item(), 0.0)
        self.assertAlmostEqual(observation.paths[0, 2, 0].item(), 0.25, places=6)
        # The normalized two-hop path contributes gamma0/2 = 1/4.
        logit = math.log(1.25) + 0.5 * math.log(11) - 1
        expected = 0.2 / (1 + math.exp(-logit))
        self.assertGreater(observation.score[0, 2, 0].item(), 0.0)
        self.assertAlmostEqual(observation.score[0, 2, 0].item(), expected, places=6)

    def test_disabling_paths_removes_path_only_observability(self):
        cfg = Config(max_path_length=2, gamma_path=0.0)
        observation = estimate([source([True, True, False]),
                                source([False, True, True])], cfg)
        self.assertGreater(observation.paths[0, 2, 0].item(), 0.0)
        self.assertEqual(observation.score[0, 2, 0].item(), 0.0)
        self.assertEqual(observation.gate[0, 2, 0].item(), 0.0)

    def test_tiny_positive_bridge_is_preserved(self):
        cfg = Config(max_path_length=1, rho_min=0.0)
        reliability = 1e-9
        observation = estimate([source([True, True], reliability, count=1e9)], cfg)
        direct = reliability ** 2
        support = 1e9 * reliability
        logit = (math.log1p(direct) + 0.5 * math.log1p(support)
                 - 1 / (1 + direct))
        expected = direct / (1 + direct) / (1 + math.exp(-logit))
        actual = observation.score[0, 1, 0].item()
        self.assertGreater(actual, 0.0)
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0))

    def test_scores_are_symmetric_bounded_and_zero_on_diagonal(self):
        observation = estimate([source([True, True, False, False]),
                                source([False, True, True, False]),
                                source([False, False, False, True])], Config())
        for value in (observation.score, observation.gate):
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.all((value >= 0) & (value <= 1)))
            torch.testing.assert_close(value, value.transpose(0, 1))
            self.assertTrue(torch.all(value.diagonal(dim1=0, dim2=1) == 0))
        self.assertTrue(torch.all(observation.gate <= observation.score))
        self.assertTrue(torch.all(observation.score[3] == 0))


if __name__ == "__main__":
    unittest.main()
