from scipy.stats import norm
import logging
import numpy as np
import protobufs.adpredictor_pb2 as pb
import util
from collections import namedtuple

logger = logging.getLogger(__name__)


class AdPredictor(object):
    Config = namedtuple(
        'Config',
        ['beta', 'prior_probability', 'epsilon', 'num_features'])

    def __init__(self, config):
        self._config = config
        self._weights = {}

        # Initial bias weight
        bias_weight = util.prior_bias_weight(
            config.prior_probability,
            config.beta,
            config.num_features)

        self._set_weight(util.bias_feature(), bias_weight)

    def predict(self, features):
        logger.info("Predicting: %s features", len(features))
        assert len(features) == self._config.num_features

        total_mean, total_variance = self._active_mean_variance(features)
        return norm.cdf(total_mean / total_variance)

    def train(self, features, label):
        logger.info("Training: %s, %s features", label, len(features))
        assert len(features) == self._config.num_features

        y = util.label_to_float(label)
        total_mean, total_variance = self._active_mean_variance(features)
        v, w = util.gaussian_corrections(y * total_mean / np.sqrt(total_variance))

        for feature in features:
            weight = self._get_weight(feature)
            mean_delta = y * weight.variance / np.sqrt(total_variance) * v
            variance_multiplier = 1.0 - weight.variance / total_variance * w
            updated = pb.Gaussian(
                mean=weight.mean + mean_delta,
                variance=weight.variance * variance_multiplier)

            self._set_weight(feature, self._apply_dynamics(updated))

    def _active_mean_variance(self, features):
        means = (self._get_weight(f).mean for f in features)
        variances = (self._get_weight(f).variance for f in features)
        return sum(means), sum(variances) + self._config.beta ** 2

    def _get_weight(self, feature):
        return self._weights.get(
            util.serialize_feature(feature),
            util.prior_weight())

    def _set_weight(self, feature, weight):
        logger.debug("Setting feature: %s frow weight: %s to weight: %s",
                     util.pp(feature),
                     util.pp(self._get_weight(feature)),
                     util.pp(weight))
        assert not np.isnan(weight.mean)
        assert weight.variance >= 0.0
        self._weights[util.serialize_feature(feature)] = weight

    @property
    def weights(self):
        return [(util.deserialize_feature(f), w)
                for (f, w) in self._weights.iteritems()]

    def _apply_dynamics(self, weight):
        prior = util.prior_weight()
        adjusted_variance = weight.variance * prior.variance / \
            ((1.0 - self._config.epsilon) * prior.variance +
             self._config.epsilon * weight.variance)
        adjusted_mean = adjusted_variance * (
            (1.0 - self._config.epsilon) * weight.mean / weight.variance +
            self._config.epsilon * prior.mean / prior.variance)

        adjusted = pb.Gaussian(mean=adjusted_mean, variance=adjusted_variance)
        logger.debug("Adjusting weight %s to %s",
                     util.pp(weight), util.pp(adjusted))
        return adjusted

    def _importance(self, feature):
        prior_prediction = self.predict(
            [util.bias_feature()] +
            [pb.Feature()] * (self._config.num_features - 1))
        with_weight_prediction = self.predict(
            [util.bias_feature()] +
            [pb.Feature()] * (self._config.num_features - 2) + [feature])

        return util.kl_divergence(with_weight_prediction, prior_prediction)
