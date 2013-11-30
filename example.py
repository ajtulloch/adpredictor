# Local Variables:
# coding: utf-8
# End:

from adpredictor import AdPredictor
from collections import namedtuple
import argh
import brewer2mpl
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import protobufs.adpredictor_pb2 as pb
import util
import os

logger = logging.getLogger(__name__)


class Sampler(object):
    """Class that manages sampling feature_vector/label pairs given a
    simulation config
    """
    def __init__(self, simulation):
        self._simulation = simulation
        self._biased_weights = self._construct_biased_weights(simulation)
        self._num_samples = 0

    def __iter__(self):
        """Implementing the Python iterator protocol
        """
        return self

    def next(self):
        """Implementing the Python iterator protocol
        """
        feature_vector = [pb.Feature(
            feature=f,
            value=np.random.randint(0, self._cardinality(f)))
            for f in range(self._simulation.predictor_config.num_features)]
        label = self._label(feature_vector)
        self._num_samples += 1
        return (feature_vector, label)

    def get_bias_for_feature(self, feature):
        return self._biased_weights.get(util.serialize_feature(feature))

    @staticmethod
    def _construct_biased_weights(simulation):
        biased_weights = {}
        for feature, value in itertools.product(
                range(1, simulation.predictor_config.num_features),
                range(simulation.feature_cardinality)):
            key = util.serialize_feature(
                pb.Feature(feature=feature, value=value))
            if np.random.rand() < simulation.biased_feature_proportion:
                direction = np.random.rand() < \
                    simulation.predictor_config.prior_probability
                biased_weights[key] = direction
                logger.info("Biased truth feature (%s, %s) to %s",
                            feature, value, direction)
        return biased_weights

    def _cardinality(self, f):
        return 1 if f == 0 else self._simulation.feature_cardinality

    def _biased_weights_label(self, features):
        for f in np.random.permutation(features):
            feature_weight = self.get_bias_for_feature(f)
            if feature_weight is not None:
                logger.debug("Hit in biased_weights (%s, %s) with bias %s",
                             f.feature, f.value, feature_weight)
                return feature_weight
        logger.debug("Missed in biased weights")
        return None

    def _default_label(self):
        return np.random.rand() < \
            self._simulation.predictor_config.prior_probability

    def _label(self, features):
        if self._num_samples > self._simulation.biased_feature_effect_length:
            return self._default_label()
        biased_label = self._biased_weights_label(features)
        if biased_label is not None:
            return biased_label
        return self._default_label()


class SimulationRunner(object):
    Simulation = namedtuple(
        'Simulation',
        ['predictor_config', 'feature_cardinality', 'num_examples',
         'biased_feature_proportion', 'directory', 'extension',
         'visualization_interval', 'biased_feature_effect_length'])

    COLORS = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors

    def __init__(self, simulation):
        self._simulation = simulation
        self._predictor = AdPredictor(simulation.predictor_config)
        self._sampler = Sampler(simulation)

    def _current_weights_by_feature(self):
        by_feature = lambda kv: kv[0].feature
        by_feature_value = lambda kv: (kv[0].feature, kv[0].value)
        weights = sorted(self._predictor.weights, key=by_feature_value)
        for feature, group in itertools.groupby(weights, key=by_feature):
            yield feature, [(f, w.mean, w.variance) for (f, w) in group]

    def _plot_weights(self):
        for color, (feature, weights) in itertools.izip(
                itertools.cycle(self.COLORS),
                self._current_weights_by_feature()):
            _, means, variances = zip(*weights)
            logging.debug("Feature %s, Weights: %s", feature, weights)

            label = "F{}".format(feature) if feature != 0 else "Bias"
            plt.scatter(means, variances,
                        label=label, color=color, alpha=0.8, s=40)

    def _annotate_biased_weights(self):
        for _, weights in self._current_weights_by_feature():
            for (feature, mean, variance) in weights:
                bias_weight = self._sampler.get_bias_for_feature(feature)
                if bias_weight is not None:
                    plt.annotate('+' if bias_weight else '-', (mean, variance),
                                 size=40)

    def _visualize(self, num_examples):
        plt.clf()

        self._plot_weights()
        self._annotate_biased_weights()

        plt.title(u"(μ, σ²) after {} examples".format(num_examples))
        plt.xlabel(u"μ")
        plt.ylabel(u"σ²")
        plt.legend(loc='best')
        plt.xlim(-4, 4)
        plt.ylim(-0.1, 1.1)

        filename = "{:03d}.{}".format(num_examples, self._simulation.extension)
        logger.info("Saving graph to %s", filename)
        plt.savefig(os.path.join(self._simulation.directory, filename),
                    dpi=300)

    def run(self):
        samples = itertools.islice(self._sampler,
                                   self._simulation.num_examples)
        for iteration, (features, label) in enumerate(samples):
            self._predictor.train(features, label)
            if iteration % self._simulation.visualization_interval == 0:
                self._visualize(iteration)


def main(
        verbose=False,
        beta=0.05,
        prior_probability=0.5,
        epsilon=0.05,
        num_features=8,
        feature_cardinality=5,
        num_examples=100,
        visualization_interval=100,
        biased_feature_proportion=0.2,
        biased_feature_effect_length=10 ** 100,
        directory="/tmp/adpredictor/",
        extension="png"):
    # Initialize globals
    np.random.seed(1)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    # Construct settings
    simulation = SimulationRunner.Simulation(
        predictor_config=AdPredictor.Config(
            beta=beta,
            prior_probability=prior_probability,
            epsilon=epsilon,
            num_features=num_features),
        feature_cardinality=feature_cardinality,
        num_examples=num_examples,
        directory=directory,
        biased_feature_proportion=biased_feature_proportion,
        biased_feature_effect_length=biased_feature_effect_length,
        visualization_interval=visualization_interval,
        extension=extension)

    # Train and output graphs
    simulation_runner = SimulationRunner(simulation)
    simulation_runner.run()


if __name__ == "__main__":
    argh.dispatch_command(main)
