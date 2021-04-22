"""
This module contains different policies for making decisions.
"""

from copy import deepcopy

import numpy as np

import featuremaps
import truedistribution
import utils


# -------------------------------------------------------------------------
# region (Abstract) Base Policy
# -------------------------------------------------------------------------
class BasePolicy:
    """Base class for policies."""

    def __init__(self, init=None, cost=None, featuremap=None, config=None):
        """Initialize a policy.

        Args:
            init: Indicator how to initialize the policy.
            cost: The cost factor of the utility.
            featuremap: A featuremap to apply to the inputs (not needed if init
                is an instance of a BasePolicy.
            config: Configuration dictionary.
        """
        self.theta = None
        self.cost = cost
        self.fm = featuremap
        self.config = deepcopy(config)
        self.init = init

        if self.fm is None:
            self.fm = featuremaps.FeatureMapIdentity()

        if init is not None:
            if isinstance(self.init, np.ndarray):
                self.theta = self.init.copy()
            elif isinstance(self.init, BasePolicy):
                self.theta = self.init.theta.copy()
                self.cost = self.init.cost
                self.fm = self.init.fm
                self.config = deepcopy(self.init.config)
            else:
                self._init_theta()

    def _init_theta(self):
        """Initialize theta."""
        if self.init == "normal":
            self.theta = np.random.randn(self.fm.n_components)
        elif self.init == "uniform":
            self.theta = np.random.rand(self.fm.n_components)
        elif self.init in "zeros":
            self.theta = np.zeros(self.fm.n_components)
        else:
            raise RuntimeError(f"Unknown initialization {self.init}.")

    def set_theta(self, theta):
        """
        Set the weight vector theta.

        Args:
            theta: np.ndarray, the weight vector theta
        """
        self.theta = theta
        self.fm.n_components = len(theta)

    def sample(self, x):
        """
        Sample decisions for given inputs.

        Args:
            x: The inputs for which to sample (binary) decisions (np.ndarray).

        Returns:
            d: A binary (0/1) vector np.ndarray of length x.shape[0]
        """
        raise NotImplementedError("Subclass must override sample(x).")

    def copy(self):
        """Create and return a copy."""
        raise NotImplementedError("Subclass must override copy().")


# endregion


# -------------------------------------------------------------------------
# region Stochastic Logistic Policy (with variations)
# -------------------------------------------------------------------------
class LogisticPolicy(BasePolicy):
    """A policy based on generalized logistic regression."""

    def __init__(self, init, cost=None, featuremap=None, config=None):
        """Initialize a logistic policy."""
        super().__init__(init, cost, featuremap, config)
        self.keep_positive = self.config["keep_positive"]
        self.type = "semi_logistic" if self.keep_positive else "logistic"

    def sample(self, x):
        """
        Sample decisions for given inputs.

        Args:
            x: The inputs for which to sample (binary) decisions (np.ndarray).

        Returns:
            d: A binary (0/1) vector np.ndarray of length x.shape[0]
        """
        if self.theta is None:
            self.fm.fit(x)
            self._init_theta()
        yprob = utils.sigmoid(np.matmul(self.fm(x), self.theta))
        d = np.round(yprob)
        explore = np.ones(len(yprob)).astype(bool)
        if self.keep_positive:
            explore &= yprob < 0.5
        d[explore] = np.random.binomial(1, yprob[explore])
        return d.astype(float)

    def copy(self):
        return LogisticPolicy(self)


# endregion


# -------------------------------------------------------------------------
# region Deterministic Logistic Policy
# -------------------------------------------------------------------------
class DeterministicThreshold(BasePolicy):
    """A deterministic threshold policy."""

    def __init__(self, init, cost, featuremap=None):
        """Initialize a logistic policy."""
        super().__init__(init, cost, featuremap)
        self.type = "deterministic_threshold"

    def sample(self, x):
        """
        Compute decisions for given inputs.

        Args:
            x: The inputs for which to sample (binary) decisions (np.ndarray).

        Returns:
            d: A binary (0/1) vector np.ndarray of length x.shape[0]
        """
        if self.theta is None:
            self.fm.fit(x)
            self._init_theta()
        return (
            utils.sigmoid(np.matmul(self.fm(x), self.theta)) > self.cost
        ).astype(float)

    def set_rule(self, func):
        """Override the sample function."""
        self.sample = func
        self.theta = None

    def set_threshold(self, thresh):
        """Override the sample function by a threshold."""
        self.sample = lambda x: (x[:, 1] > thresh).astype(float)
        self.theta = None

    def copy(self):
        return DeterministicThreshold(self, self.cost)


# endregion


# -------------------------------------------------------------------------
# region Bernoulli (fully randomized) Policy
# -------------------------------------------------------------------------
class Bernoulli(BasePolicy):
    """A fully randomized fair coin flip policy."""

    def __init__(self):
        """Initialize a logistic policy."""
        super().__init__()
        self.type = "bernoulli"
        self.theta = None

    def sample(self, x):
        """
        Compute decisions for given inputs.

        Args:
            x: The inputs for which to sample (binary) decisions (np.ndarray).

        Returns:
            d: A binary (0/1) vector np.ndarray of length x.shape[0]
        """
        return np.random.randint(0, 2, x.shape[0])

    def copy(self):
        return self


# endregion


# -------------------------------------------------------------------------
# region Static helper functions and global variables
# -------------------------------------------------------------------------
def get_optimal_policy(opt, cost, featuremap=None):
    """
    Compute the optimal policy either from a threshold, a weight vector,
    or examples.

    This is a bit dirty. Basically we need to figure out whether we can compute
    the optimal deterministic policy for the given distribution analytically.
    Therefore we do a whole lot of checks for the distribution to figure this
    out. Could be done better.

    Args:
        opt: Can be a single float (threshold) an np.ndarray of 2 elements
            (given theta) or a tuple (x,y) with datapoints x, y.
        cost: The cost factor in the utility.
        featuremap: The featuremap to be used.

    Returns:
        The optimal `DeterministicThreshold` policy.
    """
    # if we can find the optimal one it is going to be deterministic
    pi = DeterministicThreshold("zeros", cost, featuremap)
    # threshold
    if isinstance(opt, float):
        pi.set_threshold(opt)
    elif isinstance(opt, truedistribution.BaseDistribution):
        if hasattr(opt, "threshold"):
            pi = get_optimal_policy(opt.threshold(cost), cost, featuremap=None)
        else:
            if opt.is_1d:
                pi.set_rule(
                    lambda x: (
                        opt.sample_labels(x, None, yproba=True)[1] > cost
                    ).astype(float)
                )
            else:
                pi = None
    else:
        pi = None
    return pi


# endregion
