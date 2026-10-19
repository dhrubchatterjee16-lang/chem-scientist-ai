"""
Tests for Conformal Prediction & Applicability Domain
"""

import unittest
import numpy as np
from chem_scientist.conformal import ConformalMolecularEngine, ApplicabilityDomainDetector

class TestConformal(unittest.TestCase):
    def test_conformal_coverage_monotonicity(self):
        engine = ConformalMolecularEngine()
        interval = engine.predict_conformal_interval(point_prediction=-12.5)
        # 95% margin must be greater than or equal to 90% margin
        self.assertGreaterEqual(interval.margin_95, interval.margin_90)
        self.assertLessEqual(interval.lower_bound_95, interval.lower_bound_90)
        self.assertGreaterEqual(interval.upper_bound_95, interval.upper_bound_90)

    def test_applicability_domain_detection(self):
        detector = ApplicabilityDomainDetector()
        np.random.seed(42)
        train_latents = np.random.randn(50, 4)
        detector.fit(train_latents)

        # In-distribution sample
        in_sample = np.random.randn(4) * 0.5
        # Out-of-distribution sample (extreme values)
        ood_sample = np.array([100.0, -100.0, 50.0, -80.0])

        self.assertLess(detector.compute_distance(in_sample), detector.compute_distance(ood_sample))
        self.assertFalse(detector.is_inside(ood_sample))

if __name__ == "__main__":
    unittest.main()
