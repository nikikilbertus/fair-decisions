"""
This module contains learning strategies to update a policy under sequential
observations.
"""

import numpy as np
from logzero import logger
from numpy.linalg.linalg import LinAlgError
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from tqdm import tqdm

import utils
from metrics import FairnessHistory
from policies import LogisticPolicy, DeterministicThreshold


# -------------------------------------------------------------------------
# region (Abstract) Base Strategy
# -------------------------------------------------------------------------
class BaseStrategy:
    """An (abstract) base strategy."""

    def __init__(self, td, config):
        """Instantiate a strategy."""
        self.config = config
        self.td = td
        self.utility_opt = None
        self._init_parameters()
        self.deployed = {
            "pis": [],
            "thresholds": [],
            "utilities": [],
            "reaped_utilities": [],
        }
        monitored = self.config["results"]["monitored"]
        self.test_history = FairnessHistory(monitored)
        self.reaped_history = FairnessHistory(monitored)
        self.data_type = None
        self.policy_type = None

    def _init_parameters(self):
        """Initialize shortcuts for often used parameters from the config."""
        # Optimization parameters
        config_opt = self.config["optimization"]
        self.config_opt = config_opt
        self.minibatches = config_opt["minibatches"]
        self.batchsize = config_opt["batchsize"]
        self.n_samples = self.minibatches * self.batchsize
        self.time_steps = config_opt["time_steps"]
        self.epochs = config_opt["epochs"]
        self.lr_init = float(config_opt["learning_rate"])
        self.fix_prop = config_opt["fix_proposed"]
        # Misc parameters
        self.cost = self.config["utility"]["cost"]
        self.n_util_estim = self.config["utility"]["n_samples_estimate"]
        self.data_seeds = config_opt["data_seeds"]
        self.max_data = config_opt["max_data"]

    def _next_learning_rate_timestep(self, lr, t):
        """Return new learning rate for time steps."""
        if (t + 1) % self.config_opt["lr_frequency"] == 0:
            return lr * float(self.config_opt["lr_factor"])
        else:
            return lr

    def _next_learning_rate_epoch(self, lr, t):
        """Return new learning rate for epoch."""
        if (t + 1) % self.config_opt["e_lr_frequency"] == 0:
            return lr * float(self.config_opt["e_lr_factor"])
        else:
            return lr

    def _record_snapshot(self, pi, yprop, sprop, d):
        """Compute snapshot of policy and append to current history."""
        # policy
        if self.policy_type == "logistic":
            self.deployed["pis"].append(LogisticPolicy(pi, self.cost))
        elif self.policy_type == "deterministic":
            self.deployed["pis"].append(DeterministicThreshold(pi, self.cost))
        elif self.policy_type == "fixed":
            self.deployed["pis"].append(pi)
        else:
            raise RuntimeError(
                f"Cannot record full snapshot for policy {self.policy_type}"
            )
        # utility
        if pi is not None:
            utility = utils.utility(self.td, pi, self.cost, self.n_util_estim)
        else:
            utility = np.nan
        self.deployed["utilities"].append(utility)
        # threshold
        if pi is not None:
            if "deterministic" in pi.type:
                tmp_cost = self.cost
            else:
                tmp_cost = None
            threshold = utils.get_threshold(pi.theta, tmp_cost)
            if threshold is not None:
                self.deployed["thresholds"].append(threshold)
        # test metrics
        xtest, ytest, stest = self.td.sample_all(self.n_util_estim)
        if pi is not None:
            dtest = pi.sample(xtest)
        else:
            dtest = ytest.copy()
        self.test_history.snapshot(ytest, dtest, stest)
        # reaped utility and metrics
        if yprop is not None and sprop is not None and d is not None:
            # reaped utility
            reaped_utility = np.sum(yprop[d] - self.cost) / self.n_samples
            self.deployed["reaped_utilities"].append(reaped_utility)
            # reaped metrics
            self.reaped_history.snapshot(yprop, d, sprop)

    def _initialize_data_buffers(self):
        """Allocate memory buffers to keep data when training on all data."""
        if self.data_type == "all":
            if self.fix_prop:
                self.x_buf = np.empty((0, self.td.feature_dim), dtype=float)
                self.y_buf = np.empty(0, dtype=float)
                self.s_buf = np.empty(0, dtype=int)
                self.w_buf = np.empty(0, dtype=float)
            else:
                self.x_buf = np.zeros(
                    (self.n_samples * self.time_steps, self.td.feature_dim),
                    dtype=float,
                )
                self.y_buf = np.zeros(
                    self.n_samples * self.time_steps, dtype=float
                )
                self.s_buf = np.zeros(
                    self.n_samples * self.time_steps, dtype=int
                )
                self.w_buf = np.zeros(
                    self.n_samples * self.time_steps, dtype=float
                )
        elif self.data_type == "recent":
            self.x_buf, self.y_buf, self.s_buf = None, None, None
        else:
            raise RuntimeError(f"Invalid data_type {self.data_type}")

    def _update_data_buffers(self, x, y, s, weights, t, accepted):
        """Update the internal databuffers with data to be used next."""
        if self.data_type == "all":
            if self.fix_prop:
                self.x_buf = np.concatenate((self.x_buf, x), axis=0)
                self.y_buf = np.concatenate((self.y_buf, y), axis=0)
                self.s_buf = np.concatenate((self.s_buf, s), axis=0)
                if weights is not None:
                    self.w_buf = np.concatenate((self.w_buf, weights), axis=0)
                if self.x_buf.shape[0] > self.max_data:
                    self.x_buf = self.x_buf[-self.max_data :]
                    self.y_buf = self.y_buf[-self.max_data :]
                    self.s_buf = self.s_buf[-self.max_data :]
                    if weights is not None:
                        self.w_buf = self.w_buf[-self.max_data :]
                n_total = self.x_buf.shape[0]

            else:
                self.x_buf[t * accepted : (t + 1) * accepted, :] = x
                self.y_buf[t * accepted : (t + 1) * accepted] = y
                self.s_buf[t * accepted : (t + 1) * accepted] = s
                if weights is not None:
                    self.w_buf[t * accepted : (t + 1) * accepted] = weights
                n_total = (t + 1) * accepted
            if self.policy_type == "logistic":
                self._warn_if_few_minibatches(n_total)
        elif self.data_type == "recent":
            self.x_buf, self.y_buf, self.s_buf = x, y, s
            if self.policy_type == "logistic":
                self._warn_if_few_minibatches(accepted)
        else:
            raise RuntimeError(f"Invalid data_type {self.data_type}")

    def _warn_if_few_minibatches(self, n_data):
        """Log a warning message if we are working with little data."""
        minibatches = float(n_data) / self.batchsize
        if minibatches < 1:
            logger.warning("single minibatch:" f"{n_data} / {self.batchsize}")

    def _merge_and_convert_results(self):
        """Convert the stored list of snapshot to data into numpy arrays."""
        for k, v in self.deployed.items():
            if k != "pis":
                self.deployed[k] = np.array(v)
        for k, v in self.test_history.history.items():
            self.deployed["test_" + k] = v
        for k, v in self.reaped_history.history.items():
            self.deployed["reaped_" + k] = v
        return self.deployed

    def train(self, pi):
        """Train the given policy according to the strategy."""
        raise NotImplementedError("Subclass must override `train`.")


# endregion


# -------------------------------------------------------------------------
# region Merely roll out a fixed policy
# -------------------------------------------------------------------------
class UnrollStaticPolicy(BaseStrategy):
    def __init__(self, td, config):
        """
        Initialize a strategy that just rolls out a non-learning policy.

        If there the policy fed into the train function is None, the labels
        from the ground truth distribution will be used, i.e., the
        UnrollStaticPolicy unrolls a ground truth oracle that knows the outcome
        ahead of the decision.

        Args:
            td: Ground truth distribution.
            config: The configuration dictionary.
        """
        super().__init__(td, config)
        self.policy_type = "fixed"
        self.data_type = "recent"

    def train(self, pi):
        self._record_snapshot(pi, None, None, None)

        for t in tqdm(range(self.time_steps)):

            xprop, yprop, sprop, d = utils.collect_data(
                self.td,
                self.n_samples,
                policy=pi,
                fix_proposed=self.fix_prop,
                random_state=self.data_seeds[t],
            )
            _, y, _ = xprop[d], yprop[d], sprop[d]

            # the oracle
            if pi is None:
                d = y == 1
                y = y[d]

            accepted = len(y)
            if accepted < 1:
                logger.warning(f"0 accepted; continue")

            self._record_snapshot(pi, yprop, sprop, d)
        return self._merge_and_convert_results()


# endregion


# -------------------------------------------------------------------------
# region Learn a policy (learn to decide) using inverse propensity scores
# -------------------------------------------------------------------------
class IPSStrategy(BaseStrategy):
    """
    Update the policy with inverse propensity score matching.
    """

    def __init__(self, td, config, data_type):
        """Initialize the strategy."""
        super().__init__(td, config)
        self.data_type = data_type
        self.policy_type = "logistic"

    def train(self, pi):
        lr_timestep = self.lr_init
        self._initialize_data_buffers()
        self._record_snapshot(pi, None, None, None)

        for t in tqdm(range(self.time_steps)):
            lr_timestep = self._next_learning_rate_timestep(lr_timestep, t)

            # collect new data
            xprop, yprop, sprop, d = utils.collect_data(
                self.td,
                self.n_samples,
                policy=self.deployed["pis"][t],
                fix_proposed=self.fix_prop,
                random_state=self.data_seeds[t],
            )
            x, y, s = xprop[d], yprop[d], sprop[d]

            if self.data_type == "all":
                w = self._get_weights(x, self.deployed["pis"][t])
            else:
                w = None
            accepted = len(y)

            if accepted < 1:
                # didn't get any data, continue
                logger.warning(f"0 accepted; continue")
            else:
                self._update_data_buffers(x, y, s, w, t, accepted)
                train_size = len(self.y_buf)
                use_data = min(train_size, self.batchsize * self.minibatches)

                # epochs
                lr_epoch = lr_timestep
                for e in range(self.epochs):
                    lr_epoch = self._next_learning_rate_epoch(lr_epoch, e)
                    perm = np.random.permutation(train_size)[:use_data]
                    xp = self.x_buf[perm]
                    yp = self.y_buf[perm]
                    sp = self.s_buf[perm]
                    if self.data_type == "all":
                        wp = self.w_buf[perm]
                    # minibatches
                    for i1 in range(0, use_data, self.batchsize):
                        i2 = min(i1 + self.batchsize, use_data)
                        xb, yb, sb = xp[i1:i2], yp[i1:i2], sp[i1:i2]
                        wb = None if self.data_type != "all" else wp[i1:i2]
                        # gradient step
                        grad = self._grad_utility(
                            (xb, yb, sb),
                            pi,
                            self.deployed["pis"][t],
                            weights=wb,
                        )
                        pi.theta += lr_epoch * grad
            self._record_snapshot(pi, yprop, sprop, d)
        return self._merge_and_convert_results()

    def _grad_utility(self, sample, cur_policy, sample_policy, weights=None):
        """
        Estimate the gradient of the objective (i.e., utility plus fairness term
        if present) wrt the parameters of a logistic or semi_logistic policy
        from a given sample.

        Args:
            sample: A data sample tuple (x, y, s) consisting of features, labels,
                and protected attribute
            cur_policy: The current policy with respect to which the gradient of the
                utility is computed.
            sample_policy: The policy under which the data `sample` was collected.
            weights: Either None or weights from sampling policy are provided.

        Returns:
            The gradient (np.ndarray)
        """
        x, y, s = sample
        phi = cur_policy.fm(x)
        d = cur_policy.sample(x)

        # Common denominator from the score function of the current policy
        denom = 1.0 + np.exp(np.matmul(phi, cur_policy.theta))

        # If recent: numerator from reweighting by previous induced policy
        if weights is None:
            weights = np.ones_like(denom)
            sample_exp = np.exp(-np.matmul(phi, sample_policy.theta))
            if cur_policy.type == "semi_logistic":
                weights[sample_exp >= 1] *= 1.0 + sample_exp[sample_exp >= 1]
            else:
                weights = 1.0 + sample_exp

        # Each gradient term has d / denom in it
        tmp = d / denom
        if cur_policy.type == "semi_logistic":
            # Checking whether p >= 0.5 is same as x >= 0 is same as 1 + exp >= 2
            tmp[denom >= 2] = 0.0

        # Gradient of utility
        grad_util = (y - self.cost) * tmp
        grad_util *= weights
        grad_util = np.sum(phi * grad_util[:, np.newaxis], axis=0) / x.shape[0]

        if self.config["fairness"] is None:
            return grad_util

        # Difference of benefit terms themselves (with weights, no denom)
        benefit_difference = self._mean_difference(d * weights, s)

        # Gradient of benefit term
        if self.config["fairness"] == "demographic_parity":
            tmp_ben = tmp
        elif self.config["fairness"] == "equal_opportunity":
            tmp_ben = y * tmp
        else:
            raise ValueError(f"Unknown fairness: {self.config['fairness']}.")
        # The difference of the gradients
        grad_ben = tmp_ben * weights
        grad_ben = phi * grad_ben[:, np.newaxis]
        grad_ben = benefit_difference * self._mean_difference(grad_ben, s)

        return grad_util - self.config["lambda"] * grad_ben

    @staticmethod
    def _mean_difference(val, s):
        mask0, mask1 = s == 0, s == 1
        n0, n1 = np.sum(mask0), np.sum(mask1)
        if n0 == 0 or n1 == 0:
            return 0.0 if val.ndim == 1 else np.zeros(val.shape[-1])
        return (
            np.sum(val[mask0], axis=0) / n0 - np.sum(val[mask1], axis=0) / n1
        )

    @staticmethod
    def _get_weights(x, sample_policy):
        """Get the weight factors for some examples under a given policy."""
        phi = sample_policy.fm(x)
        sample_exp = np.exp(-np.matmul(phi, sample_policy.theta))
        weights = np.ones_like(sample_exp)
        if sample_policy.type == "semi_logistic":
            weights[sample_exp >= 1] *= 1.0 + sample_exp[sample_exp >= 1]
        else:
            weights = 1.0 + sample_exp
        return weights


# endregion


# -------------------------------------------------------------------------
# region Learn a predictive threshold model agnostic to distribution shifts
# -------------------------------------------------------------------------
class PredictiveStrategy(BaseStrategy):
    """
    An update strategy for a thresholded predictive model (unaware of data
    distribution shifts).
    """

    def __init__(self, td, config, data_type):
        """Initialize the strategy."""
        super().__init__(td, config)
        self.data_type = data_type
        self.policy_type = "deterministic"

    def train(self, pi):
        self._initialize_data_buffers()
        self._record_snapshot(pi, None, None, None)

        for t in tqdm(range(self.time_steps)):
            # collect new data
            xprop, yprop, sprop, d = utils.collect_data(
                self.td,
                self.n_samples,
                policy=self.deployed["pis"][t],
                fix_proposed=self.fix_prop,
                random_state=self.data_seeds[t],
            )
            x, y, s = xprop[d], yprop[d], sprop[d]

            accepted = len(y)
            if accepted < 1:
                # didn't get any data, continue
                logger.warning(f"0 accepted; continue")
            else:
                self._update_data_buffers(x, y, s, None, t, accepted)
                train_size = len(self.y_buf)
                use_data = min(train_size, self.batchsize * self.minibatches)
                perm = np.random.choice(train_size, use_data, replace=False)
                # update logistic model in pi
                try:
                    pi.set_theta(
                        np.array(
                            utils.fit_logit(self.x_buf[perm], self.y_buf[perm])
                        )
                    )
                except (PerfectSeparationError, LinAlgError) as err:
                    logger.info(f"Error in LogReg: {err}")

            self._record_snapshot(pi, yprop, sprop, d)
        return self._merge_and_convert_results()


# endregion
