"""Various classification metrics for binary outcomes and protected attributes.

This module contains a collection of functions to evaluate binary predictions
of a model on a given dataset potentially conditioned on membership in a
protected group. This bears resemblance to
https://github.com/Trusted-AI/AIF360/blob/master/aif360/metrics/classification_metric.py
but was implemented independently.
"""
from typing import Union, Optional, Sequence, Text, Tuple, Dict, Callable, Any

import numpy as np


# ==============================================================================
# FAIRNESS HISTORY APPEND ONLY RECORDER
# ==============================================================================


class FairnessHistory(object):
    """Keep track of various fairness and classification metrics."""

    def __init__(self, quantities: Sequence[Text]):
        """Initialize a fairness history.

        Possible quantities to record are:
          Stored as 3-tuples for the two groups and the entire dataset:
            "A": num_individuals,
            "N": num_negatives,
            "P": num_positives,
            "TP": true_positives
            "TN": true_negatives
            "FP": false_positives
            "FN": false_negatives
            "TPR": true_positive_rate
            "FPR": false_positive_rate
            "TNR": true_negative_rate
            "FNR": false_negative_rate
            "PPV": positive_predictive_value
            "NPV": negative_predictive_value
            "FDR": false_discovery_rate
            "FOR": false_omission_rate
            "ACC": accuracy
            "ERR": error_rate
            "SEL": selection_rate
            "F1": f1_score
          Stored as single numbers:
            "AVG_ODDS_DIFF": average of the difference of FPR and TPR
            "AVG_ABS_ODDS_DIFF": average of the absolute difference of FPR and
                TPR
            "ERR_DIFF": average of the absolute difference of FPR and TPR
            "DI": disparate impact: ratio of the fraction of positive outcomes
            "DP": demographic parity: diff between fractions of positive
                outcomes
            "EOP": equal opportunity: diff in true positive rates

        Args:
          quantities: A list of strings indicating which measures to record.
        """
        self._history = {quantity: [] for quantity in quantities}

    @property
    def history(self) -> Dict[Text, np.ndarray]:
        """The history of values recorded so far.

        Returns:
          A dictionary with the requested quantities as keys and arrays as
          values.
        """
        return {key: np.array(val) for key, val in self._history.items()}

    def snapshot(self, y: np.ndarray, yhat: np.ndarray, protected: np.ndarray):
        """Add a snapshot.

        Args:
          y: An array of binary (0/1) true outcomes.
          yhat: An array of binary (0/1) predictions/decisions.
          protected: A binary (0/1) array of for the protected attribute.
        """
        metrics = ClassificationMetrics(y, yhat, protected)
        for quantity, history in self._history.items():
            if quantity in metrics.label_to_fun:
                history.append(
                    metrics.all_groups(metrics.label_to_fun[quantity])
                )
            elif quantity in metrics.label_to_fair:
                history.append(metrics.label_to_fair[quantity]())
            else:
                raise ValueError("Unknown metric {}".format(quantity))


# ==============================================================================
# COLLECTION OF PERFORMANCE AND FAIRNESS METRICS FOR BINARY CLASSIFICATION
# ==============================================================================


class ClassificationMetrics(object):
    """A collection of fairness metrics."""

    def __init__(self, y: np.ndarray, yhat: np.ndarray, protected: np.ndarray):
        """Initialize fairness metrics.

        Args:
          y: An array of binary (0/1) true outcomes.
          yhat: An array of binary (0/1) predictions/decisions.
          protected: A binary (0/1) array for the protected attribute.
        """
        self._y = y
        self._yhat = yhat
        self._protected = protected.astype(bool)
        self._unprotected = np.logical_not(self._protected)
        self._results = {"all": {}, "protected": {}, "unprotected": {}}
        self._check_validity()

        self.label_to_fun = {
            "A": self.num_individuals,
            "N": self.num_negatives,
            "P": self.num_positives,
            "TP": self.num_true_positives,
            "TN": self.num_true_negatives,
            "FP": self.num_false_positives,
            "FN": self.num_false_negatives,
            "TPR": self.true_positive_rate,
            "FPR": self.false_positive_rate,
            "TNR": self.true_negative_rate,
            "FNR": self.false_negative_rate,
            "PPV": self.positive_predictive_value,
            "NPV": self.negative_predictive_value,
            "FDR": self.false_discovery_rate,
            "FOR": self.false_omission_rate,
            "ACC": self.accuracy,
            "ERR": self.error_rate,
            "SEL": self.selection_rate,
            "F1": self.f1_score,
        }
        self.label_to_fair = {
            "AVG_ODDS_DIFF": self.average_odds_difference,
            "AVG_ABS_ODDS_DIFF": self.average_abs_odds_difference,
            "ERR_DIFF": self.error_rate_difference,
            "DI": self.disparate_impact,
            "DP": self.demographic_parity_difference,
            "EOP": self.equal_opportunity_difference,
        }

    # ==========================================================================
    # Private utility functions
    # ==========================================================================

    def _check_validity(self):
        """Check whether the length of the inputs match.

        Raises:
          ValueError: Wrong input dimensions.
        """
        if len(self._y) != len(self._yhat):
            raise ValueError("y and yhat must be same length.")
        if self._protected is not None and len(self._y) != len(
            self._protected
        ):
            raise ValueError("y and protected must be same length.")

    def _get_condition(self, protected: Union[bool, None]) -> np.ndarray:
        """Get the indices for the individuals in the specified group.

        Args:
          protected: Whether to get the indices of the protected group (True),
              the unprotected group (False), or the entire dataset (None).

        Returns:
          An index array of the same length as the input data.
        """
        if protected is None:
            return np.full(len(self._y), True)
        elif protected:
            return self._protected
        else:
            return self._unprotected

    def _get_key(self, protected: Union[bool, None]) -> Text:
        """Convert bool or None into hashable key for dictionary indexing.

        Args:
          protected: Whether to address the protected group (True), the
              unprotected group (False), or the entire dataset (None).

        Returns:
          String, the key for a dictionary.
        """
        if protected is None:
            return "all"
        elif protected:
            return "protected"
        else:
            return "unprotected"

    def _num_confusionquadrant(
        self,
        true_val: int,
        predicted_val: int,
        protected: Optional[bool] = None,
    ) -> int:
        """Compute the number of instances in a quadrant of the confusion table.

        Args:
          true_val: The value for the true outcomes (0/1).
          predicted_val: the value for the predicted outcomes (0/1).
          protected: Boolean whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          Integer, the number of instances satisfying the selection criterium.
        """
        idx = self._get_condition(protected)
        return int(
            np.sum(
                ((self._y == true_val) & (self._yhat == predicted_val))[idx]
            )
        )

    @staticmethod
    def _difference(metric_function) -> float:
        """Compute the difference of the given metric between the two groups.

        Args:
          metric_function: The metric to be computed on the groups.

        Returns:
          float, the `metric_function` for the protected group minus the
          `metric_function` for the unprotected group.
        """
        return metric_function(protected=True) - metric_function(
            protected=False
        )

    @staticmethod
    def _ratio(metric_function) -> float:
        """Compute the ratio of the given metric between the two groups.

        Args:
          metric_function: The metric to be computed on the groups.

        Returns:
          float, the `metric_function` for the protected group divided by the
          `metric_function` for the unprotected group.
        """
        denominator = metric_function(protected=False)
        if np.isclose(denominator, 0.0):
            return np.inf
        return metric_function(protected=True) / denominator

    # ==========================================================================
    # Public utility functions
    # ==========================================================================

    def metrics(self) -> Any:
        """Compute all available metrics for all groups.

        Returns:
          A tuple of dictionaries:
              performance: Keys are "all", "protected", "unprotected" and the
                  values are dictionaries containing all available metrics.
              fairness: Keys are fairness measure labels and keys are numbers.
        """
        performance = {}
        for protected in [True, False, None]:
            group_label = self._get_key(protected)
            performance[group_label] = {}
            for label, function in self.label_to_fun.items():
                performance[group_label][label] = function(protected)
        fairness = {}
        for label, function in self.label_to_fair.items():
            fairness[label] = function()
        return performance, fairness

    @staticmethod
    def all_groups(
        metric_function: Callable[..., Union[float, int]]
    ) -> Tuple[Union[float, int], Union[float, int], Union[float, int]]:
        """Compute given metric for all possible groups (including all data).

        Args:
          metric_function: The metric to be computed on the groups.

        Returns:
          3-tuple: the `metric_function` for
            [0] the entire dataset
            [1] the protected group
            [2] the unprotected group.
        """
        return (
            metric_function(protected=None),
            metric_function(protected=True),
            metric_function(protected=False),
        )

    # ==========================================================================
    # Standard classification metrics
    # ==========================================================================

    def num_individuals(self, protected: Optional[bool] = None) -> int:
        """Get the number of individuals in a given class or the entire dataset.

        Args:
          protected: Whether to count the protected group (True), the
              unprotected group (False), or the entire dataset (None).

        Returns:
          integer, the number of individuals in the specified group.
        """
        return int(np.sum(self._get_condition(protected)))

    def num_positives(self, protected: Optional[bool] = None) -> int:
        """Number of real positive cases in the data.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).
        Returns:
          Integer, the number of real positive cases in the data.
        """
        key = self._get_key(protected)
        if "P" not in self._results[key]:
            idx = self._get_condition(protected)
            self._results[key]["P"] = int(np.sum(self._y[idx]))
        return self._results[key]["P"]

    def num_negatives(self, protected: Optional[bool] = None) -> int:
        """Number of real negative cases in the data.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).
        Returns:
          Integer, the number of real negative cases in the data.
        """
        key = self._get_key(protected)
        if "N" not in self._results[key]:
            idx = self._get_condition(protected)
            self._results[key]["N"] = int(np.sum(1 - self._y[idx]))
        return self._results[key]["N"]

    def num_predicted_positives(self, protected: Optional[bool] = None) -> int:
        """Number of positive predictions.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).
        Returns:
          Integer, the number of positive predictions.
        """
        key = self._get_key(protected)
        if "PPRED" not in self._results[key]:
            idx = self._get_condition(protected)
            self._results[key]["PPRED"] = int(np.sum(self._yhat[idx]))
        return self._results[key]["PPRED"]

    def num_predicted_negatives(self, protected: Optional[bool] = None) -> int:
        """Number of negative predictions.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).
        Returns:
          Integer, the number of negative predictions.
        """
        key = self._get_key(protected)
        if "NPRED" not in self._results[key]:
            idx = self._get_condition(protected)
            self._results[key]["NPRED"] = int(np.sum(1 - self._yhat[idx]))
        return self._results[key]["NPRED"]

    def num_true_positives(self, protected: Optional[bool] = None) -> int:
        """Number of true positives.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          Integer, the number of true positives.
        """
        key = self._get_key(protected)
        if "TP" not in self._results[key]:
            self._results[key]["TP"] = self._num_confusionquadrant(
                1, 1, protected
            )
        return self._results[key]["TP"]

    def num_true_negatives(self, protected: Optional[bool] = None) -> int:
        """Number of true negatives.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          Integer, the number of true negatives.
        """
        key = self._get_key(protected)
        if "TN" not in self._results[key]:
            self._results[key]["TN"] = self._num_confusionquadrant(
                0, 0, protected
            )
        return self._results[key]["TN"]

    def num_false_positives(self, protected: Optional[bool] = None) -> int:
        """Number of false positives.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          Integer, the number of false positives.
        """
        key = self._get_key(protected)
        if "FP" not in self._results[key]:
            self._results[key]["FP"] = self._num_confusionquadrant(
                0, 1, protected
            )
        return self._results[key]["FP"]

    def num_false_negatives(self, protected: Optional[bool] = None) -> int:
        """Number of false negatives.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          Integer, the number of false negatives.
        """
        key = self._get_key(protected)
        if "FN" not in self._results[key]:
            self._results[key]["FN"] = self._num_confusionquadrant(
                1, 0, protected
            )
        return self._results[key]["FN"]

    def true_positive_rate(self, protected: Optional[bool] = None) -> float:
        """True positive rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the true positive rate.
        """
        denom = self.num_positives(protected)
        if denom > 0:
            return self.num_true_positives(protected) / denom
        else:
            return np.inf

    def true_negative_rate(self, protected: Optional[bool] = None) -> float:
        """True negative rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the true negative rate.
        """
        denom = self.num_negatives(protected)
        if denom > 0:
            return self.num_true_negatives(protected) / denom
        else:
            return np.inf

    def false_positive_rate(self, protected: Optional[bool] = None) -> float:
        """False positive rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the false positive rate.
        """
        denom = self.num_negatives(protected)
        if denom > 0:
            return self.num_false_positives(protected) / denom
        else:
            return np.inf

    def false_negative_rate(self, protected: Optional[bool] = None) -> float:
        """false negative rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the false negative rate.
        """
        denom = self.num_negatives(protected)
        if denom > 0:
            return self.num_false_negatives(protected) / denom
        else:
            return np.inf

    def positive_predictive_value(
        self, protected: Optional[bool] = None
    ) -> float:
        """Positive predictive value.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the positive predictive value.
        """
        denom = self.num_true_positives(protected) + self.num_false_positives(
            protected
        )
        if denom > 0:
            return self.num_true_positives(protected) / denom
        else:
            return np.inf

    def negative_predictive_value(
        self, protected: Optional[bool] = None
    ) -> float:
        """Negative predictive value.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the negative predictive value.
        """
        denom = self.num_true_negatives(protected) + self.num_false_negatives(
            protected
        )
        if denom > 0:
            return self.num_true_negatives(protected) / denom
        else:
            return np.inf

    def false_discovery_rate(self, protected: Optional[bool] = None) -> float:
        """False discovery rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the false discovery rate.
        """
        denom = self.num_false_positives(protected) + self.num_true_positives(
            protected
        )
        if denom > 0:
            return self.num_false_positives(protected) / denom
        else:
            return np.inf

    def false_omission_rate(self, protected: Optional[bool] = None) -> float:
        """False omission rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the false omission rate.
        """
        denom = self.num_false_negatives(protected) + self.num_true_negatives(
            protected
        )
        if denom > 0:
            return self.num_false_negatives(protected) / denom
        else:
            return np.inf

    def accuracy(self, protected: Optional[bool] = None) -> float:
        """Accuracy.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the accuracy.
        """
        denom = self.num_positives(protected) + self.num_negatives(protected)
        if denom > 0:
            return (
                self.num_true_positives(protected)
                + self.num_true_negatives(protected)
            ) / denom
        else:
            return np.inf

    def error_rate(self, protected: Optional[bool] = None) -> float:
        """Error rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the error rate.
        """
        return 1.0 - self.accuracy(protected)

    def selection_rate(self, protected: Optional[bool] = None) -> float:
        """Selection rate.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the selection rate.
        """
        denom = self.num_individuals(protected)
        if denom > 0:
            return self.num_predicted_positives(protected) / denom
        else:
            return np.inf

    def f1_score(self, protected: Optional[bool] = None) -> float:
        """F1 score.

        Args:
          protected: Whether to condition on the protected group (True),
              the unprotected group (False) or compute on the entire dataset
              (None).

        Returns:
          float in [0, 1], the F1 score.
        """
        denom = (
            2 * self.num_true_positives(protected)
            + self.num_false_positives(protected)
            + self.num_false_negatives(protected)
        )
        if denom > 0:
            return 2 * self.num_true_positives(protected) / denom
        else:
            return np.inf

    # ==========================================================================
    # Aliases
    # ==========================================================================

    def precision(self, protected: Optional[bool] = None) -> float:
        """Alias of `positive_predictive_value`."""
        return self.positive_predictive_value(protected)

    def recall(self, protected: Optional[bool] = None) -> float:
        """Alias of `true_positive_rate`."""
        return self.true_positive_rate(protected)

    def fallout(self, protected: Optional[bool] = None) -> float:
        """Alias of `false_positive_rate`."""
        return self.false_positive_rate(protected)

    def sensitivity(self, protected: Optional[bool] = None) -> float:
        """Alias of `true_positive_rate`."""
        return self.true_positive_rate(protected)

    def specificity(self, protected: Optional[bool] = None) -> float:
        """Alias of `true_negative_rate`."""
        return self.true_negative_rate(protected)

    def selectivity(self, protected: Optional[bool] = None) -> float:
        """Alias of `true_negative_rate`."""
        return self.true_negative_rate(protected)

    # ==========================================================================
    # Fairness metrics
    # ==========================================================================

    def average_odds_difference(self) -> float:
        """Average of the difference in FPR and TPR."""
        return 0.5 * (
            self._difference(self.false_positive_rate)
            + self._difference(self.true_positive_rate)
        )

    def average_abs_odds_difference(self) -> float:
        """Average of the difference in FPR and TPR."""
        return 0.5 * (
            np.abs(self._difference(self.false_positive_rate))
            + np.abs(self._difference(self.true_positive_rate))
        )

    def error_rate_difference(self) -> float:
        """Difference in the error rate."""
        return self._difference(self.error_rate)

    def disparate_impact(self) -> float:
        """Disparate impact: ratio of the fraciton of positive outcomes."""
        return self._ratio(self.selection_rate)

    def demographic_parity_difference(self) -> float:
        """Demographic parity: difference of fractions of positive outcomes."""
        return self._difference(self.selection_rate)

    def equal_opportunity_difference(self) -> float:
        """Difference in true positive rates."""
        return self._difference(self.true_positive_rate)
