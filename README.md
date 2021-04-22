# Overview

This is the code for the paper

**Fair decisions despite imperfect predictions**

*Niki Kilbertus, Manuel Gomez-Rodriguez, Bernhard Schölkopf, Krikamol Muandet, Isabel Valera*

[https://arxiv.org/abs/1902.02979](https://arxiv.org/abs/1902.02979)

It allows to train different kinds of policies (logistic, semi-logistic, deterministic thresholded predictive model) via various strategies (update on most recent data batch, update on all data).

## Prerequisites

The dependencies can be found in the `requirements.txt`.
Please use your favorite python environment management system to install them.
We tested this code with Python 3.9.

## Run

The main file is `run.py` which only takes a configuration json file as input, via

```shell
python run.py --path path/to/file/config.json
```

We try to keep a single config file for each relevant example. Currently, they reside in the `experiments` folder:

* `config_uncalibrated.json`: This is the config file for the first synthetic 1D example presented in the paper with monotonically increasing $P(y|x)$, which is not calibrated however.
* `config_split.json`: This is the config file for the second synthetic 1D example presented in the paper with two disjoint green regions.
* `config_compas.json`: This is the config file for the real-world COMPAS dataset example.

In effect, each run of `run.py` is specific to a single dataset. However, it runs multiple different policies and training strategy combinations in a single run as specified by the `perform` option in the config file.

A standard setting would be

```json
"perform": [
  ["recent", "logistic"],
  ["all", "logistic"],
  ["recent", "semi_logistic"],
  ["all", "semi_logistic"],
  ["recent", "deterministic_threshold"]
  ["all", "deterministic_threshold"],
]
```

This is a list of pairs consisting of `strategy` and `policy`. In this example we would run all combinations that we also showed in the paper.

The strategies basically refer to the data on which the model is updated at each time step: *recent* means that we only update on the most recently observed batch of data, whereas *all* stands for updating on all data that has been seen so far.

For the policies, *logistic* and *semi-logistic* are the policies to directly learn decisions as described in the paper. They both use *IPS* (inverse propensity score weighting), i.e., the training strategy is aware of the data shift induced by the existing policy and corrects for it via inverse propensity score weighting as described in the paper.
The *deterministic_threshold* policy internally learns a predictive model for the labels $y$ (not for decisions!), i.e., $Q(y|x, s)$ and then thresholds by the cost parameters, i.e., $d = \mathbb{I}[Q(y|x, s) \ge c]$.
