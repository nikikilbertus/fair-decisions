"""This module contains plotting and debugging functionality.

This is only used to produce some plots on the fly while running the code, i.e.,
only for single runs.
"""

import os
from timeit import default_timer as time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import sigmoid, utility, get_threshold

# -------------------------------------------------------------------------
# region Global variables
# -------------------------------------------------------------------------
# This dictionary is a remnant from early times when the names for policies and
# strategies were still much more obscure. It's useful still to be able to
# change the keys without changing the labels in the plots.
LABELNAMES = {
    "recent": "recent",
    "all": "all",
    "deterministic_threshold": "deterministic",
    "logistic": "logistic",
    "semi_logistic": "semi-logistic",
    "bernoulli": "bernoulli",
    "optimal": "optimal",
    "oracle": "oracle",
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

STYLEMAP = {
    "oracle": dict(c="k", ls=":"),
    "optimal": dict(c="r", ls="-"),
    "logistic": dict(c=COLORS[0]),
    "semi_logistic": dict(c=COLORS[1]),
    "deterministic_threshold": dict(c=COLORS[2]),
    "recent": dict(ls="-"),
    "all": dict(ls="--"),
}
# endregion


# -------------------------------------------------------------------------
# region Plotters
# -------------------------------------------------------------------------
def plot_models(
    thetas, td, cost=None, save=True, path=None, suff="", points=100
):
    """Plot the decision functions of various logistic models in 1D.

    Args:
        thetas: List of parameters of various models.
        td: The true distribution.
        cost: Cost factor (if deterministic policy).
        save: Whether to store the figure to disk.
        path: The path where to save the figure (only if `save=True`).
        suff: Optional suffix for the file name.
        points: How many points to use for the plot.
    """
    if save and path is None:
        raise RuntimeError("Need path to save figure.")
    if thetas.shape[1] != 2:
        raise RuntimeError("Can only plot decision functions in 1D.")
    if cost is None:
        cost = 0.5

    plt.figure()

    n_models = thetas.shape[0]
    skip = int(n_models // 20)
    if skip == 0:
        skip = 1
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    x, _, _ = td.sample_all(points)
    perm = np.argsort(x[:, 1])
    x = x[perm]
    for i in range(n_models):
        plt.plot(
            x[:, 1],
            sigmoid(thetas[i, :] @ x.T),
            color=colors[i],
            lw=0.7,
            label=(f"t={i}" if i in [0, n_models - 1] else None),
        )
        if i % skip == 0 or i in [0, n_models - 1]:
            plt.axvline(
                x=get_threshold(thetas[i], cost),
                ls="dashed",
                alpha=0.5,
                c=colors[i],
                lw=0.7,
            )
    if hasattr(td, "threshold"):
        opt_thresh = td.threshold(cost)
        plt.axvline(x=opt_thresh, ls="dashed", c="crimson", label="optimal")
    plt.legend()
    plt.tight_layout()
    if save:
        if suff:
            suff = "_" + suff
        figname = "func" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches="tight")
    else:
        plt.show()


def plot_td_samples_1d(
    td, cost=None, save=True, path=None, suff="", samples=300
):
    """Plot some samples from the true distribution and their probabilities."""
    if save and path is None:
        raise RuntimeError("Need path to save figure.")
    plt.figure()
    x0, y0, s0, yprob0 = td.sample_all(samples, yproba=True)
    perm = np.argsort(x0[:, 1])
    xs = x0[perm]
    ys = y0[perm]
    yprobs = yprob0[perm]
    if s0 is not None:
        ss = s0[perm]
        plt.plot(
            xs[ss == 0, 1], ys[ss == 0], ".", label="samples, s=0", alpha=0.5
        )
        plt.plot(
            xs[ss == 1, 1], ys[ss == 1], ".", label="samples, s=1", alpha=0.5
        )
    else:
        plt.plot(xs[:, 1], ys, ".", xs[:, 1], yprobs, ".", label="samples")
    plt.plot(xs[:, 1], yprobs, label="true probabilities")
    plt.axhline(y=cost, ls="dashed", c="k", label=f"c = {cost:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.title("Samples from the ground truth distribution")
    if save:
        if suff:
            suff = "_" + suff
        figname = "td_data" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches="tight")
    else:
        plt.show()


def plot_td_samples(
    td, names=None, save=True, path=None, suff="", samples=300
):
    """Plot some samples from the true distribution and their probabilities."""
    if save and path is None:
        raise RuntimeError("Need path to save figure.")
    plt.figure()
    x, y, _, yprob = td.sample_all(samples, yproba=True)
    df = pd.DataFrame(
        np.hstack((x, yprob[:, np.newaxis])), columns=names, figsize=(20, 20)
    )
    pd.plotting.scatter_matrix(df)
    plt.tight_layout()
    if save:
        if suff:
            suff = "_" + suff
        figname = "td_data" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches="tight")
    else:
        plt.show()


def plot_results(learned, fixed, save=True, path=None, suff=""):
    """Plot various characteristics over the training for various settings.

    Args:
        learned: A dictionary containing the results (dict values: arrays)
            for various settings (dict keys: (str, str)).
        fixed: A dictionary containing fixed result values.
        save: Whether to save the figure.
        path: Where to save the figure (only if `save=True`.
        suff: Optional suffix for the file name.
    """
    plt.figure()

    rows = 7
    fig, axs = plt.subplots(rows, 2, figsize=(20, rows * 4))

    # -------------------------------------------------------------------------
    # fixed policies
    for policy, results in fixed.items():
        tmp_label = LABELNAMES[policy]
        # utility (test)
        axs[0, 0].plot(
            results["utilities"], label=tmp_label, **STYLEMAP[policy]
        )
        timesteps = np.arange(1, len(results["utilities"]) + 1)
        axs[0, 1].plot(
            np.cumsum(results["utilities"]) / timesteps,
            label=tmp_label,
            **STYLEMAP[policy],
        )
        # utility (reaped)
        axs[1, 0].plot(
            results["reaped_utilities"], label=tmp_label, **STYLEMAP[policy]
        )
        timesteps = np.arange(1, len(results["reaped_utilities"]) + 1)
        axs[1, 1].plot(
            np.cumsum(results["reaped_utilities"]) / timesteps,
            label=tmp_label,
            **STYLEMAP[policy],
        )
        # selection rate (test)
        axs[2, 0].plot(
            results["test_SEL"][:, 0], label=tmp_label, **STYLEMAP[policy]
        )
        # selection rate (reaped)
        axs[2, 1].plot(
            results["reaped_SEL"][:, 0], label=tmp_label, **STYLEMAP[policy]
        )
        # disparate impact (test)
        axs[3, 0].plot(results["test_DI"], label=tmp_label, **STYLEMAP[policy])
        # disparate impact (reaped)
        axs[3, 1].plot(
            results["reaped_DI"], label=tmp_label, **STYLEMAP[policy]
        )
        # demogrpahic parity (test)
        axs[4, 0].plot(results["test_DP"], label=tmp_label, **STYLEMAP[policy])
        # demographic parity (reaped)
        axs[4, 1].plot(
            results["reaped_DP"], label=tmp_label, **STYLEMAP[policy]
        )
        # equal opportunity (test)
        axs[5, 0].plot(
            results["test_EOP"], label=tmp_label, **STYLEMAP[policy]
        )
        # equal opportunity (reaped)
        axs[5, 1].plot(
            results["reaped_EOP"], label=tmp_label, **STYLEMAP[policy]
        )

        if results["thresholds"] and len(results["thresholds"]) > 0:
            ls_stoch = "dotted" if "logistic" in policy else None
            axs[6, 0].plot(results["thresholds"], ls=ls_stoch, label=tmp_label)

    # -------------------------------------------------------------------------
    # learned policies
    for (policy, strategy), results in learned.items():
        tmp_label = f"{LABELNAMES[policy]} {LABELNAMES[strategy]}"
        # utility (test)
        axs[0, 0].plot(
            results["utilities"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        timesteps = np.arange(1, len(results["utilities"]) + 1)
        axs[0, 1].plot(
            np.cumsum(results["utilities"]) / timesteps,
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # utility (reaped)
        axs[1, 0].plot(
            results["reaped_utilities"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        timesteps = np.arange(1, len(results["reaped_utilities"]) + 1)
        axs[1, 1].plot(
            np.cumsum(results["reaped_utilities"]) / timesteps,
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # selection rate (test)
        axs[2, 0].plot(
            results["test_SEL"][:, 0],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # selection rate (reaped)
        axs[2, 1].plot(
            results["reaped_SEL"][:, 0],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # disparate impact (test)
        axs[3, 0].plot(
            results["test_DI"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # disparate impact (reaped)
        axs[3, 1].plot(
            results["reaped_DI"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # demogrpahic parity (test)
        axs[4, 0].plot(
            results["test_DP"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # demographic parity (reaped)
        axs[4, 1].plot(
            results["reaped_DP"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # equal opportunity (test)
        axs[5, 0].plot(
            results["test_EOP"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )
        # equal opportunity (reaped)
        axs[5, 1].plot(
            results["reaped_EOP"],
            label=tmp_label,
            **STYLEMAP[policy],
            **STYLEMAP[strategy],
        )

        if len(results["thresholds"]) > 0:
            ls_stoch = "dotted" if "logistic" in policy else None
            axs[6, 0].plot(results["thresholds"], ls=ls_stoch, label=tmp_label)

    for i in range(rows):
        axs[i, 0].set_xlabel(r"time step $t$")
        axs[i, 1].set_xlabel(r"time step $t$")
    axs[0, 0].set_ylabel(r"utility $u(\pi_{\theta_t}, c)$")
    axs[0, 1].set_ylabel(r"cumulative utility")
    axs[1, 0].set_ylabel(
        r"reaped utility $\sum_{y \in \mathcal{D}_t} (y - c)$"
    )
    axs[1, 1].set_ylabel(r"cumulative reaped utility")
    axs[2, 0].set_ylabel(r"selection rate")
    axs[2, 1].set_ylabel(r"reaped selection rate")
    axs[3, 0].set_ylabel(r"disparate impact")
    axs[3, 1].set_ylabel(r"reaped disparate impact")
    axs[4, 0].set_ylabel(r"demographic parity")
    axs[4, 1].set_ylabel(r"reaped demographic parity")
    axs[5, 0].set_ylabel(r"equal opportunity")
    axs[5, 1].set_ylabel(r"reaped equal opportunity")
    axs[6, 0].set_ylabel(r"decision boundary in input space")
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    if save:
        if suff:
            suff = "_" + suff
        figname = "utilities" + suff + ".pdf"
        figname = os.path.abspath(os.path.join(path, figname))
        plt.savefig(figname, bbox_inches="tight")
    else:
        plt.show()

# endregion