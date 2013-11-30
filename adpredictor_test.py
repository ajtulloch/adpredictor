import protobufs.adpredictor_pb2 as pb
from adpredictor import AdPredictor
import numpy as np
import unittest
import util


class AdPredictorTest(unittest.TestCase):
    @staticmethod
    def _create_predictor(beta=0.05, prior=0.3, epsilon=0.01, num_features=10):
        config = AdPredictor.Config(beta, prior, epsilon, num_features)
        return AdPredictor(config)

    @staticmethod
    def _create_feature_vector(num_features):
        return [util.bias_feature()] + \
            [pb.Feature()] * (num_features - 1)

    def setUp(self):
        self._predictor = self._create_predictor()

    def test_bias_correctly_initialized(self):
        num_features, num_priors = 10, 10
        for prior in np.random.rand(num_priors):
            predictor = self._create_predictor(prior=prior, num_features=10)
            self.assertAlmostEqual(
                predictor.predict(self._create_feature_vector(num_features)),
                prior)

    def test_importance_of_empty_feature(self):
        self.assertEqual(self._predictor._importance(pb.Feature()), 0.0)

    def test_importance_of_set_feature(self):
        f = pb.Feature(feature=10, value=5)
        self._predictor._set_weight(f, pb.Gaussian(mean=0.5, variance=0.5))
        self.assertGreater(self._predictor._importance(f), 0.0)

    def test_importance_is_monotonic_in_mean(self):
        f = pb.Feature(feature=10, value=5)
        importances = []
        for mean in np.linspace(0.0, 3.0, 10):
            self._predictor._set_weight(f, pb.Gaussian(mean=0.5, variance=0.5))
            importances.append(self._predictor._importance(f))
        self.assertEqual(sorted(importances), importances)

    def test_dynamics_shift_towards_prior(self):
        p = self._create_predictor(epsilon=0.05)
        initial = pb.Gaussian(mean=5.0, variance=0.5)
        adjusted = p._apply_dynamics(initial)
        self.assertGreater(initial.mean, adjusted.mean)
        self.assertLess(initial.variance, adjusted.variance)

    def test_dynamics_neutral_for_prior(self):
        p = self._create_predictor(epsilon=0.05)
        initial = util.prior_weight()
        adjusted = p._apply_dynamics(initial)
        self.assertAlmostEqual(initial.mean, adjusted.mean)
        self.assertAlmostEqual(initial.variance, adjusted.variance)

    def test_dynamic_is_null_for_epsilon_zero(self):
        p = self._create_predictor(epsilon=0.0)
        initial = pb.Gaussian(mean=5.0, variance=0.5)
        adjusted = p._apply_dynamics(initial)
        self.assertAlmostEqual(initial.mean, adjusted.mean)
        self.assertAlmostEqual(initial.variance, adjusted.variance)
