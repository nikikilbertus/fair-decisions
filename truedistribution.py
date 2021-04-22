"""
This module contains a class representing a true underlying distribution.
"""

import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats.distributions import truncnorm
from sklearn.model_selection import train_test_split

import utils
from data import get_data


# -------------------------------------------------------------------------
# region (Abstract) Base Distribution
# -------------------------------------------------------------------------
class BaseDistribution:
    """A generative model of the underlying ground truth distribution."""

    def __init__(self, config):
        """
        Initialize the true distribution.

        Args:
            config: The configuration dictionary.
        """
        self.config = config
        self.feature_dim = None
        self.is_1d = False
        self._test_set = None

    def sample_features(self, n):
        """
        Sample feature variables.

        Args:
            n: Number of examples.

        Returns:
            2-tuple of np.ndarrays:
                [0]: x, an iid sample of features from the true distribution
                [1]: s, an iid sample of protected from the true distribution
        """
        raise NotImplementedError("Subclass must override `sample_features`.")

    def sample_labels(self, x, s, yproba=False):
        """
        Sample labels given feature variables.

        Args:
            x: Feature examples.
            s: Protected attributes.
            yproba: Whether to return the probabilities of the binary labels.

        Returns:
            An iid sample of labels from the true distribution given the
            features x and protected attribute s (np.ndarray).
        """
        raise NotImplementedError("Subclass must override `sample_labels`.")

    def sample_all(self, n, yproba=False):
        """
        Draw a full sample of features and labels.

        Args:
            n: The number of exmaples to draw.
            yproba: Whether to return the probabilities of the binary labels.

        Returns:
            x, s, y, (yproba): np.ndarrays of features, binary labels, and the
                corresponding probabilities for the labels. The latter only if
                `yproba=True`.
        """
        x, s = self.sample_features(n=n)
        if yproba:
            y, yproba = self.sample_labels(x, s, yproba=yproba)
            return x, y, s, yproba
        else:
            y = self.sample_labels(x, s, yproba=yproba)
            return x, y, s

    def get_test(self, n=1000000):
        """Get a test set for this distribution.

        A testset of the given size is sampled and stored. If the test set
        function is called multiple times, with the same `n`, the cached data
        is returned. If `n` changes, a new test set is sampled and returned.

        Args:
            n: The number of examples to use for the test set.

        Returns:
            3-tuple of np.ndarrays
                [0]: x, the features
                [1]: y, the labels
                [2]: s, the protected attributes
        """
        if self._test_set is None or self._test_set[0].shape[0] != n:
            self._test_set = self.sample_all(n)
        return self._test_set


# endregion


# -------------------------------------------------------------------------
# region Dummy Custom Synthetic 1D Distribution for Proof of Concept
# -------------------------------------------------------------------------
class DummyDistribution1D(BaseDistribution):
    """A simple generative model of the true distribution."""

    def __init__(self, config):
        """
        Initialize the true distribution.

        Args:
            config: The configuration dictionary.
        """
        super().__init__(config)
        self.type = "custom1d"
        self.theta = np.array(config["theta"])
        self.feature_dim = len(self.theta)
        if "split_support" not in config["custom_tweaks"]:
            self.threshold = self._threshold
        self.is_1d = True

    def sample_features(self, n, **kwargs):
        """
        Draw examples only for the features of the true distribution.

        Args:
            n: The number of examples to draw.

        Returns:
            x: np.ndarray with the features of dimension (n, k), where k is
                either 1 or 2 depending on whether a constant is added
        """
        if self.config["protected_fraction"] is not None:
            s = (
                np.random.rand(n, 1) < self.config["protected_fraction"]
            ).astype(int)
            x = 3.5 * np.random.randn(n, 1) + 3 * (0.5 - s)
        else:
            s = np.full(n, np.nan)
            x = 3.5 * np.random.randn(n, 1)

        if self.config["protected_as_feature"]:
            x = np.concatenate((x, s.reshape(-1, 1)), axis=1)
        if self.config["add_constant"]:
            x = np.hstack([np.ones([n, 1]), x])
        return x, s.ravel()

    def sample_labels(self, x, s, yproba=False):
        """
        Draw examples of labels for given features.

        Args:
            x: Given features (usually obtained by calling `sample_features`).
            s: Sensitive attribute.
            yproba: Whether to return the probabilities of the binary labels.

        Returns:
            y: np.ndarray of binary (0/1) labels (if `yproba=False`)
            y, yproba: np.ndarrays of binary (0/1) labels as well as the
                original probabilities of the labels (if `yproba=False`)
        """
        yprob = utils.sigmoid(x.dot(self.theta))

        if "bump_left" in self.config["custom_tweaks"]:
            yprob += np.exp(-(x[:, 1] + 6) ** 2 * 2) * 0.5
            yprob = np.maximum(np.minimum(yprob, 1), 0)
        if "bump_right" in self.config["custom_tweaks"]:
            yprob -= np.exp(-(x[:, 1] - 5) ** 2 * 0.8) * 0.35
            yprob = np.maximum(np.minimum(yprob, 1), 0)
        if "split_support" in self.config["custom_tweaks"]:
            yprob = 0.8 * utils.sigmoid(0.6 * (x[:, 1] + 3)) * utils.sigmoid(
                -5 * (x[:, 1] - 3)
            ) + utils.sigmoid(x[:, 1] - 5)

        y = np.random.binomial(1, yprob)
        if yproba:
            return y, yprob
        return y

    def _threshold(self, cost):
        """The threshold for this policy."""
        if len(self.theta) == 1:
            return 0.0
        if len(self.theta) == 2:
            return utils.get_threshold(self.theta, cost)
        else:
            raise RuntimeError("Scalar threshold exists only for 1D.")


# endregion


# -------------------------------------------------------------------------
# region Fico Score Based Distribution
# -------------------------------------------------------------------------
class InverseCDF(BaseDistribution):
    """
    A simple generative model of FICO scores estimated from real data.

    This is currently unused in the experiments as it did not yield any
    interesting insights beyond the 1D synthetic examples that we looked at.
    """

    def __init__(self, config, group_weights=None):
        """
        Initialize the true distribution.

        Args:
            config: The configuration dictionary.
            group_weights: How to weight the possible groups in the mixture.
                Dictionary with keys in 'black', 'white', 'asian', 'hispanic'.
        """
        super().__init__(config)
        self.type = "inv_cdf"
        self.feature_dim = 2 if self.config["add_constant"] else 1
        self.thresh = None
        self.inv_cdfs = None
        self.pdfs = None
        self.prob_bnds = None
        self.score_bnds = None
        self.is_1d = True
        if group_weights is None:
            self.weights = {"white": 0.0, "black": 1.0}
        else:
            self.weights = group_weights
        if sum(self.weights.values()) != 1:
            raise ValueError("Weights must sum to 1.")
        self._set_inverse_cdf()

    def _set_inverse_cdf(self):
        """Compute the inverse CDF from data."""
        data_dir = os.path.abspath(self.config["path"])
        data_dir = os.path.join(data_dir, self.config["type"])
        self.inv_cdfs = {}
        self.pdfs = {}
        lo_prob, hi_prob = 0, 1000
        lo_score, hi_score = 0, 1000
        for group in self.weights:
            data_path = os.path.join(data_dir, "marginals_" + group + ".npz")
            data = np.load(data_path)
            score, proba, cdf = data["fico"], data["proba"], data["cdf"]
            # score = (score - 300.0) / 550.0
            self.inv_cdfs[group] = interp1d(
                cdf, score, kind="linear", assume_sorted=False
            )
            self.pdfs[group] = interp1d(
                score, proba, kind="linear", assume_sorted=False
            )
            lo_prob = max(lo_prob, min(proba))
            hi_prob = min(hi_prob, max(proba))
            lo_score = max(lo_score, min(score))
            hi_score = min(hi_score, max(score))
        self.prob_bnds = (1.001 * lo_prob, 0.999 * hi_prob)
        self.score_bnds = (lo_score, hi_score)

    def set_threshold_get_cost(self, threshold):
        """Find the cost corresponding to a given score threshold."""
        self.thresh = float(threshold)
        scores = np.linspace(*self.score_bnds, 300)
        return utils.find_x_for_y(self.pdf(scores), scores, threshold)

    def inv_cdf(self, prob):
        """Get the inverse CDF value."""
        x = np.zeros_like(prob, dtype="float64")
        for group, w in self.weights.items():
            x += self.weights[group] * self.inv_cdfs[group](prob)
        return x

    def pdf(self, score):
        """Get the probability of repayment."""
        x = np.zeros_like(score, dtype="float64")
        for group, w in self.weights.items():
            x += self.weights[group] * self.pdfs[group](score)
        return x

    def sample_features(self, n, **kwargs):
        """
        Draw examples only for the features of the true distribution.

        Args:
            n: The number of examples to draw.

        Returns:
            x: np.ndarray with the features of dimension (n, 2)
        """
        # Assume there are only two groups
        if len(self.weights.keys()) != 2:
            raise RuntimeError("Can only sample for two groups.")
        group_names = sorted(list(self.weights))
        s = np.random.binomial(1, self.weights[group_names[0]], (n, 1)).astype(
            int
        )
        x = np.zeros_like(s, dtype="float64")
        for group in group_names:
            probs = (
                np.random.rand(n, 1) * np.diff(self.prob_bnds)[0]
                + self.prob_bnds[0]
            )
            x += s * self.inv_cdfs[group](probs)
            s = 1 - s

        if self.config["add_constant"]:
            x = np.hstack([np.ones([n, 1]), x])
        return x, s

    def sample_labels(self, x, s, yproba=False):
        """
        Draw examples of labels for given features.

        Args:
            x: Given features (usually obtained by calling `sample_features`).
            s: Sensitive attribute.
            yproba: Whether to return the probabilities of the binary labels.

        Returns:
            y: np.ndarray of binary (0/1) labels (if `yproba=False`)
            y, yproba: np.ndarrays of binary (0/1) labels as well as the
                original probabilities of the labels (if `yproba=False`)
        """
        if x.shape[1] == 2:
            yprob = self.pdf(x[:, 1])
        else:
            yprob = self.pdf(x)
        y = np.random.binomial(1, yprob)
        if yproba:
            return y, yprob
        return y

    def threshold(self, cost):
        """The threshold for this policy."""
        assert self.thresh is not None, "Need to set threshold first"
        return self.thresh


# endregion


# -------------------------------------------------------------------------
# region A score based distribution that is uncalibrated
# -------------------------------------------------------------------------
class UncalibratedScore(BaseDistribution):
    """An distribution modelling an uncalibrated score."""

    def __init__(self, config):
        super().__init__(config)
        self.feature_dim = 2 if self.config["add_constant"] else 1
        self.type = "uncalibratedscore"
        params = config["uncalibrated_params"]
        self.bound = params["bound"]
        self.width = params["width"]
        self.height = params["height"]
        self.shift = params["shift"]
        self.thresh = None
        self.is_1d = True

    def set_threshold_get_cost(self, threshold):
        """Find the cost corresponding to a given score threshold."""
        self.thresh = float(threshold)
        x = np.linspace(-self.bound, self.bound, 300)
        return utils.find_x_for_y(self.pdf(x), x, threshold)

    def pdf(self, x):
        """Get the probability of repayment."""
        num = (
            np.tan(x)
            + np.tan(self.bound)
            + self.height
            * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def sample_features(self, n, **kwargs):

        if self.config["protected_fraction"] is not None:
            s = (
                np.random.rand(n, 1) < self.config["protected_fraction"]
            ).astype(int)
            shifts = s - 0.5
            x = truncnorm.rvs(
                -self.bound + shifts, self.bound + shifts, loc=-shifts
            ).reshape(-1, 1)
        else:
            s = np.full(n, np.nan)
            x = truncnorm.rvs(-self.bound, self.bound, size=n).reshape(-1, 1)

        if self.config["protected_as_feature"]:
            x = np.concatenate((x, s.reshape(-1, 1)), axis=1)
        if self.config["add_constant"]:
            x = np.hstack([np.ones([n, 1]), x])
        return x, s.ravel()

    def sample_labels(self, x, s, yproba=False):
        if x.shape[1] == 2:
            yprob = self.pdf(x[:, 1])
        else:
            yprob = self.pdf(x)
        y = np.random.binomial(1, yprob)
        if yproba:
            return y, yprob
        return y

    def threshold(self, cost):
        """The threshold for this policy."""
        assert self.thresh is not None, "Need to set threshold first"
        return self.thresh


# endregion


# -------------------------------------------------------------------------
# region Real data resampling distribution
# -------------------------------------------------------------------------
class ResamplingDistribution(BaseDistribution):
    """Resample from a finite dataset."""

    def __init__(self, config):
        super().__init__(config)
        self.type = "resampling"
        self.feature_dim = None
        self._load_data()

    def _load_data(self):
        """Load the specified dataset."""
        test_size = self.config["test_size"]
        x, y, s = get_data(self.config)
        x, xte, y, yte, s, ste = train_test_split(x, y, s, test_size=test_size)

        self.x, self.y, self.s = x, y, s
        self.xtest, self.ytest, self.stest = xte, yte, ste

        self.n_examples = self.x.shape[0]
        self.allindices = np.arange(self.n_examples)
        self.n_group = {}
        self.indices_group = {}
        if self.config["add_constant"]:
            self.x = np.hstack([np.ones([self.n_examples, 1]), self.x])
            self.xtest = np.hstack(
                [np.ones([self.xtest.shape[0], 1]), self.xtest]
            )
        self.feature_dim = self.x.shape[1]

    def sample_features(self, n, **kwargs):
        raise NotImplementedError("Only use `sample_all` for Resampling.")

    def sample_labels(self, x, **kwargs):
        raise NotImplementedError("Only use `sample_all` for Resampling.")

    def sample_all(self, n, yproba=False):
        assert not yproba, "Cannot compute probabilities for real data."
        n = min(self.n_examples, n)
        indices = np.random.choice(self.allindices, n, replace=True)
        return self.x[indices], self.y[indices], self.s[indices]

    def sample_by_group(self, n, group=None):
        group = int(group)
        if group not in self.indices_group:
            self.indices_group[group] = self.allindices[self.s == group]
            self.n_group[group] = len(self.indices_group[group])
        n = min(self.n_group[group], n)
        indices = np.random.choice(self.indices_group[group], n, replace=True)
        return self.x[indices], self.y[indices]

    def get_test(self, n=None):
        return self.xtest, self.ytest, self.stest


# endregion
