"""
This module contains utility functions.
"""

import os
from datetime import datetime

import numpy as np
import statsmodels.api as sm
from logzero import logger

import featuremaps
import truedistribution


# -------------------------------------------------------------------------
# region Initialization and setup
# -------------------------------------------------------------------------
def setup_directories(config):
    """Create the directory structure needed to collect results and output."""
    logger.info("Setup directories for results...")
    result_dir = os.path.abspath(config["results"]["result_dir"])
    if config["results"]["name"] is None:
        dir_name = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + config["true_distribution"]["type"]
        )
    else:
        dir_name = config["results"]["name"]
    result_dir = os.path.join(result_dir, dir_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    config["results"]["resolved"] = result_dir

    fig_dir = os.path.join(result_dir, "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    config["results"]["figure_resolved"] = fig_dir

    result_data_prefix = os.path.join(result_dir, "data_")
    return result_data_prefix


def get_list_of_seeds(num):
    """Create a random list of integer seeds."""
    max_value = 2 ** 32 - 1
    data_seeds = np.random.randint(
        0, max_value, size=num, dtype=np.dtype("int64")
    )
    return data_seeds


def initialize_true_distribution(config):
    """Initialize the ground truth distribution to be used."""
    logger.info("Setup the true distribution...")
    td_config = config["true_distribution"]
    td_type = td_config["type"]
    cost = None
    if td_type == "1d":
        td = truedistribution.DummyDistribution1D(td_config)
    elif td_type == "fico":
        td = truedistribution.InverseCDF(td_config)
        logger.info(f"Setting threshold {td_config['threshold']}...")
        cost = td.set_threshold_get_cost(td_config["threshold"])
        logger.info(f"Resetting corresponding cost: {cost}...")
        config["utility"]["cost"] = cost
    elif td_type == "uncalibrated":
        td = truedistribution.UncalibratedScore(td_config)
        logger.info(f"Setting threshold {td_config['threshold']}...")
        cost = td.set_threshold_get_cost(td_config["threshold"])
        logger.info(f"Resetting corresponding cost: {cost}...")
        config["utility"]["cost"] = cost
    else:
        td = truedistribution.ResamplingDistribution(td_config)
    return td, cost


def initialize_featuremap(config, td):
    """
    Initialize the feature map to be used.

    Args:
        config: configuration dictionary
        td: Ground truth distribution.

    Returns:
        featuremap
    """
    fm_type = config["feature_map"]["type"]
    logger.info(f"Setup the feature map using {fm_type}...")
    if fm_type == "identity":
        fm = featuremaps.FeatureMapIdentity()
    elif fm_type == "rbf":
        fm = featuremaps.FeatureMapRBF()
    else:
        raise RuntimeError(f"Unknown feature map type {fm_type}")

    logger.info("Fit feature map from data...")
    x0, y0, _ = td.sample_all(config["feature_map"]["n_fit"])
    fm.fit(x0, y0)
    return fm


def get_initial_parameters(config, td, cost):
    """
    Get the initial parameters for a predicitve model and the policy.

    The issue here is that for the predictive model, we use the cost parameter
    as a threshold to get binary decisions: P(y|x,s) > c or not.
    However, the policy directly outputs binary decisions.
    In practice we use a logistic model for both, i.e., P(y|x,s) is the output
    of a logistic regression in [0,1] and we compare it to the cost for the
    predictive model.
    However, for the policy, we always just use 0.5 as a threshold as we want
    to learn binary outputs directly.

    For the predictive model, we may obtain initial weights by pre-training
    a logistic model on a small subset of the data.
    However, initializing the policy with the same parameters is a bit unfair,
    because depending on the cost parameter, either the predictive model or the
    policy will have different acceptance thresholds.

    This function calls `theta_y_to_d` that accounts for this difference.

    Args:
        config: configuration dictionary
        td: true distribution
        cost: cost parameters

    Returns:
        (init_y, init_d): arrays, the initial parameters for the predictive
        model and the policy
    """
    policy_init = config["policy"]["initialization"]
    logger.info(f"Setup parameter initialization: {policy_init}...")
    if policy_init == "pre_trained":
        logger.info(
            f"Pretrain on {config['policy']['n_pre_training']} iid "
            "samples form ground truth distribution..."
        )
        x0, y0, _ = td.sample_all(config["policy"]["n_pre_training"])
        init_y = fit_logit(x0, y0)
        logger.info(f"Initial parameter values for predictive {init_y}")
        init_d = theta_y_to_d(init_y, cost)
        logger.info(f"Initial parameter values for decisions {init_d}")
    elif isinstance(policy_init, str):
        init_y = policy_init
        init_d = policy_init
    elif isinstance(policy_init, (list, tuple)):
        init_y = np.array(policy_init).astype(float)
        logger.info(f"Initial parameter values for predictive {init_y}")
        init_d = theta_y_to_d(init_y, cost)
        logger.info(f"Initial parameter values for decisions {init_d}")
    else:
        raise RuntimeError(f"Unknown initialization {policy_init}")
    return init_y, init_d


# endregion


# -------------------------------------------------------------------------
# region Logistic regression helper functions
# -------------------------------------------------------------------------
def sigmoid(x):
    """
    Stable (vectorized) implementation of the sigmoid function.

    Args:
        x: Inputs.

    Returns:
        sigmoid(x).
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def fit_logit(x, y):
    """
    Fit a logistic regression classifier.

    Args:
        x: Input features.
        y: Binary labels.

    Returns:
        Fitted model (using statsmodels).
    """
    model = sm.Logit(y, x)
    return model.fit(disp=0).params


# endregion


# -------------------------------------------------------------------------
# region Data collection
# -------------------------------------------------------------------------
def collect_data(
    distribution, n, policy=None, fix_proposed=True, random_state=None
):
    """
    Collect data from a given distribution under a given policy.

    Args:
        distribution: The true underlying distribution (TrueDistribution).
        n: The number of examples to draw.
        policy: A policy for making decisions (see policies). If the policy is
            `None`, return unfiltered examples from the true distribution.
        fix_proposed: Whether n is the number of proposed examples, or the
            number of accepted examples.
        random_state: Random state for sample collection.

    Returns:
        4-tuple:
            [0] xprop: features of proposed examples
            [1] yprop: labels of proposed examples
            [2] sprop: protected attributes of proposed examples
            [4]: d: decisions made by the policy on the existing examples
    """
    if random_state is not None:
        np.random.seed(random_state)

    xprop, yprop, sprop = distribution.sample_all(n, yproba=False)
    if policy is None:
        d = np.full(len(yprop), True)
    elif fix_proposed:
        d = policy.sample(xprop) == 1
    else:
        raise ValueError(
            "Fixing the number of accepted (instead of the "
            "number of proposed is not supported anymore at this "
            "point. Please set `fix_proposed` to True."
        )
    return xprop, yprop, sprop, d


# endregion


# -------------------------------------------------------------------------
# region Utility and policy helpers
# -------------------------------------------------------------------------
def utility(distribution, policy, cost, n=1000, acceptance_rate=False):
    """
    Estimate the utility of a policy under a given true distribution.

    Args:
        distribution: The true underlying distribution (TrueDistribution).
        policy: A decision making policy (see policies).
        cost: The cost parameter of the utility in [0,1].
        n: The number of monte carlo samples used to estimate the utility.
        acceptance_rate: Whether to also return the acceptance rate.

    Returns:
        The utility estimate (float).
    """
    x, y, _ = distribution.get_test(n)
    d = policy.sample(x)
    util = np.sum(d * (y - cost)) / len(d)
    if acceptance_rate:
        return util, float(np.sum(d)) / len(d)
    else:
        return util


def extract_parameters(policies):
    """
    Extract the parameters from a collection of policies.

    Args:
        policies: An iterable collection of policies.

    Returns:
        np.ndarray of dimension (n, d) where n is the number of policies in the
            input and d is the dimension of the parameters of all policies.
    """
    n = len(policies)
    d = len(policies[0].theta.squeeze())
    thetas = np.zeros((n, d))
    for i, pi in enumerate(policies):
        thetas[i, :] = pi.theta.squeeze()
    return thetas


# endregion


# -------------------------------------------------------------------------
# region Misc
# -------------------------------------------------------------------------
def find_x_for_y(xs, ys, y):
    """Find the x value for a given y value."""
    xs = np.array(xs)
    ys = np.array(ys)
    for i in range(len(xs) - 1):
        (xold, yold) = xs[i], ys[i]
        (xnew, ynew) = xs[i + 1], ys[i + 1]
        if (yold - y) * (ynew - y) < 0:
            x = xold + ((y - yold) / (ynew - yold)) * (xnew - xold)
            return x
    return None


def get_threshold(theta, cost):
    """Get the x threshold for a given theta and cost."""
    if theta is None or len(theta) > 2:
        return
    theta = theta.squeeze()
    if cost is not None:
        return (-theta[0] + np.log(cost / (1 - cost))) / theta[1]
    else:
        return -theta[0] / theta[1]


def theta_y_to_d(init_y, cost):
    """Convert an predictive model to an euqivlant decision model."""
    init_d = init_y.copy()
    init_d[0] += np.log((1.0 - cost) / cost)
    return init_d


# endregion
