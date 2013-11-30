from protobuf_to_dict import protobuf_to_dict
import protobufs.adpredictor_pb2 as pb
from scipy.stats import norm
import numpy as np
import json

MAX_ABS_SURPRISE = 5.0


def pp(message):
    """Pretty print a given protocol buffer message
    """
    return json.dumps(protobuf_to_dict(message))


def bias_feature():
    return pb.Feature(feature=0, value=0)


def prior_bias_weight(prior_probability, beta, num_features):
    """The prior weight on the bias such that on initialization of a model
    with the given parameters, P(y | x, initial_weights) = prior.
    """

    bias_mean = norm.ppf(prior_probability) * (beta ** 2 + num_features)
    return pb.Gaussian(mean=bias_mean, variance=1.0)


def prior_weight():

    """The global prior on non-bias weights
    """
    return pb.Gaussian(mean=0.0, variance=1.0)


def gaussian_corrections(t):
    """Returns the additive and multiplicative corrections for the mean
    and variance of a trunctated Gaussian random variable.

    In Trueskill/AdPredictor papers, denoted
    - V(t)
    - W(t) = V(t) * (V(t) + t)

    Returns (v(t), w(t))
    """
    # Clipping avoids numerical issues from ~0/~0.
    t = np.clip(t, -MAX_ABS_SURPRISE, MAX_ABS_SURPRISE)
    v = norm.pdf(t) / norm.cdf(t)
    w = v * (v + t)
    return (v, w)


def kl_divergence(p, q):
    """Computes the Kullback-Liebler divergence between two Bernoulli
    random variables with probability p and q.
    Algebraically, KL(p || q)
    """
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def label_to_float(label):
    assert type(label) == bool
    return 1.0 if label else -1.0


def serialize_feature(feature):
    return feature.SerializeToString()


def deserialize_feature(string):
    f = pb.Feature()
    f.ParseFromString(string)
    return f
