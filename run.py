#!/usr/bin/env python3
"""The main script running fair policy learning given a config file."""

import argparse
import json
import os
from copy import deepcopy

import logzero
import numpy as np
from logzero import logger

import plotters
import strategies
import utils
from policies import get_optimal_policy, LogisticPolicy, DeterministicThreshold


# -------------------------------------------------------------------------
# Read configuration json file specific to the dataset
# -------------------------------------------------------------------------
formatter_class = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter_class)

parser.add_argument(
    "--path",
    help="The path to the config file.",
    type=str,
    default="experiments/config_split.json",
)

args = parser.parse_args().__dict__
config_path = os.path.abspath(args["path"])

# -------------------------------------------------------------------------
# Load config file and parameters
# -------------------------------------------------------------------------
logger.info(f"Read config file from {config_path}")
with open(config_path, "r") as f:
    config = json.load(f)

# -------------------------------------------------------------------------
# Setup up directory structure for result files
# -------------------------------------------------------------------------
result_data_prefix = utils.setup_directories(config)
fig_dir = config["results"]["figure_resolved"]

# -------------------------------------------------------------------------
# Setup logger
# -------------------------------------------------------------------------
logger.info(f"Add file handle to logger...")
log_file_path = os.path.join(config["results"]["resolved"], "logs.log")
logzero.logfile(log_file_path)
logger.info(f"Logging at {log_file_path}...")

# -------------------------------------------------------------------------
# Global values / settings
# -------------------------------------------------------------------------
logger.info(f"Setting global variables...")
cost = config["utility"]["cost"]
logger.info(f"cost: {cost}")
debug = config["debug"]
logger.info(f"debug: {debug}")
# Not for repeated utility estimation; large number for accurate estimates
n_util_estim_single = config["utility"]["n_samples_estimate_single"]
np.random.seed(config["seed"])
logger.info(f"seed: {config['seed']}")

# -------------------------------------------------------------------------
# Initialize a list of seeds so each method trains on the same proposals
# -------------------------------------------------------------------------
logger.info(f"Create a list of seeds for consistent proposals...")
config["optimization"]["data_seeds"] = utils.get_list_of_seeds(
    config["optimization"]["time_steps"]
)

# -------------------------------------------------------------------------
# Initialize the ground truth distribution
# -------------------------------------------------------------------------
td, new_cost = utils.initialize_true_distribution(config)
if new_cost is not None:
    cost = new_cost
    logger.info(f"cost hast been updated to {cost}")

if debug > 0 and td.is_1d:
    logger.info("Plot true distribution and samples...")
    plotters.plot_td_samples_1d(td, cost, path=fig_dir)

# -------------------------------------------------------------------------
# Construct the feature map (always use identity for now)
# -------------------------------------------------------------------------
fm = utils.initialize_featuremap(config, td)

# -------------------------------------------------------------------------
# Compute optimal deterministic policy and utility
# -------------------------------------------------------------------------
logger.info("Try to compute optimal policy and utility...")
utility_opt = None
rate_opt = None
threshold_opt = None
if hasattr(td, "threshold"):
    threshold_opt = td.threshold(cost)
pi_det_opt = get_optimal_policy(td, cost, fm)
if pi_det_opt is not None:
    utility_opt, rate_opt = utils.utility(
        td, pi_det_opt, cost, n=n_util_estim_single, acceptance_rate=True
    )
    logger.info(f"Found optimal policy and utility ({utility_opt})...")
else:
    logger.info("Could not compute optimal policy and utility...")

# -------------------------------------------------------------------------
# Get Policy Initialization
# -------------------------------------------------------------------------
init_y, init_d = utils.get_initial_parameters(config, td, cost)

# -------------------------------------------------------------------------
# Which policy-strategy combinations to run?
# -------------------------------------------------------------------------
logger.info(f"Run the following combinations {config['perform']}...")
to_run = sorted(list(set([tuple(x) for x in config["perform"]])))

# -------------------------------------------------------------------------
# Train policies using chosen strategies
# -------------------------------------------------------------------------
# collect results for plotting
training_results = {}
for data_strategy, policy_type in to_run:
    logger.info(f"Setup the policy of type {policy_type}...")
    if "logistic" in policy_type:
        # config will get changed, so we better make a deepcopy for each run
        tmp_config = deepcopy(config["policy"])
        # this decides whether it's a logistic or semi-logistic policy
        tmp_config["keep_positive"] = "semi" in policy_type
        pi = LogisticPolicy(init_d, featuremap=fm, config=tmp_config)
        logger.info(f"Use {data_strategy}...")
        strat = strategies.IPSStrategy(td, config, data_strategy)
    elif policy_type == "deterministic_threshold":
        pi = DeterministicThreshold(init_y, cost, featuremap=fm)
        logger.info(f"Use {data_strategy}...")
        strat = strategies.PredictiveStrategy(td, config, data_strategy)
    else:
        raise RuntimeError(f"Unknown policy type {policy_type}")

    # Train the policy with the chosen strategy
    deployed = strat.train(pi)
    # Try to extrac the parameters theta of all deployed policies (if existent)
    all_theta = utils.extract_parameters(deployed["pis"])
    # Policies cannot be stored as arrays in .npz and are no longer needed
    del deployed["pis"]

    logger.info(
        f"Write results and plot {policy_type} trained on {data_strategy}..."
    )
    suffix = policy_type + "_" + data_strategy
    fname = result_data_prefix + suffix + ".npz"
    np.savez(
        fname,
        thetas=all_theta,
        data_seeds=config["optimization"]["data_seeds"],
        **deployed,
    )
    # Collect results for plotting
    training_results[(policy_type, data_strategy)] = deployed
    # For the synthetic examples, plot the policies themselves
    if debug > 0 and td.is_1d:
        if "deterministic" in policy_type:
            eff_cost = cost
        else:
            eff_cost = None
        plotters.plot_models(
            all_theta, td, cost=eff_cost, save=True, path=fig_dir, suff=suffix
        )

# -------------------------------------------------------------------------
# Evaluate optimal policy if known
# This is the optimal implementable policy, i.e., the deterministic threshold
# policy that has access to a perfect predictive model.
# -------------------------------------------------------------------------
fixed_results = {}
if pi_det_opt is not None:
    logger.info(f"Evaluate optimal policy...")
    strat = strategies.UnrollStaticPolicy(td, config)
    deployed = strat.train(pi_det_opt)
    deployed["thresholds"] = utils.get_threshold(pi_det_opt.theta, cost)
    del deployed["pis"]
    logger.info(f"Write results for optimal policy...")
    fname = result_data_prefix + "optimal.npz"
    np.savez(
        fname, data_seeds=config["optimization"]["data_seeds"], **deployed
    )
    fixed_results["optimal"] = deployed

# -------------------------------------------------------------------------
# Evaluate oracle policy
# This is a practically impossible policy that can foresee the true labels
# of individuals before they are realized.
# -------------------------------------------------------------------------
logger.info(f"Evaluate oracle policy...")
strat = strategies.UnrollStaticPolicy(td, config)
# Setting pi=None is the flag for using the oracle policy
deployed = strat.train(None)
del deployed["pis"]
logger.info(f"Write results for oracle...")
fname = result_data_prefix + "oracle.npz"
np.savez(fname, data_seeds=config["optimization"]["data_seeds"], **deployed)
fixed_results["oracle"] = deployed

# -------------------------------------------------------------------------
# Plot the results. This is a preliminary plotting function for some early
# visualization of what is going on. The publication ready plots are generated
# from the stored data in a separate notebook.
# -------------------------------------------------------------------------
if debug > 0:
    plotters.plot_results(
        training_results, fixed_results, save=True, path=fig_dir
    )

# -------------------------------------------------------------------------
# Write out the (updated) config file and wrap up the run
# -------------------------------------------------------------------------
logger.info("Write out used config file...")
res_config_path = os.path.join(config["results"]["resolved"], "config.json")
# remove non serializable entries
del config["optimization"]["data_seeds"]
with open(res_config_path, "w") as f:
    json.dump(config, f, indent=2)

logger.info("Finished run.")
