"""Visualizations for the hierarchically-local similarity results.

All functions take the long-format results DataFrame written by the experiment
(see ``repsim.experiment.run_experiment``) and save a figure, returning its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_RELATION_ORDER = ["descendant", "self", "ancestor"]


def _binned_trend(
    sub: pd.DataFrame, n_bins: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean held-out R^2 within equal-width log2(n_classes) bins.

    Returns the bin-centre breadth (linear scale), mean R^2, and standard error
    per bin, dropping empty bins. Used to draw an aggregate granularity trend.
    """
    log_breadth = np.log2(sub["n_classes"].to_numpy())
    edges = np.linspace(log_breadth.min(), log_breadth.max() + 1e-9, n_bins + 1)
    which = np.digitize(log_breadth, edges) - 1
    centres, means, ses = [], [], []
    for b in range(n_bins):
        vals = sub["r2_eval"].to_numpy()[which == b]
        if vals.size:
            centres.append(2 ** ((edges[b] + edges[b + 1]) / 2))
            means.append(vals.mean())
            ses.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(centres), np.array(means), np.array(ses)


def plot_r2_vs_granularity(results: pd.DataFrame, out_dir: Path) -> Path:
    """Held-out R^2 vs node breadth for the hierarchical nodes (linear).

    Each target contributes one faint line through its descendant -> self ->
    ancestor chain (ordered by breadth); a bold binned-mean trend aggregates
    across all targets and model pairs. Tests the core claim: alignment degrades
    as the region of the hierarchy broadens (coarsens).
    """
    hier = results[(results["grouping"] == "hierarchical") & (results["transform"] == "linear")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, chain in hier.groupby("target"):
        line = chain.groupby("n_classes")["r2_eval"].mean().sort_index()
        ax.plot(line.index, line.values, color="0.7", lw=0.7, alpha=0.5, zorder=1)
    centres, means, ses = _binned_trend(hier)
    ax.errorbar(centres, means, yerr=ses, color="C3", lw=2.5, marker="o",
                capsize=3, zorder=3, label="binned mean +/- SE")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("node breadth: #ImageNet classes (log scale; finer $\\to$ coarser)")
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("Cross-model alignment vs hierarchy granularity")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "r2_vs_granularity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_real_vs_null(results: pd.DataFrame, out_dir: Path) -> Path:
    """Held-out R^2 vs breadth for real subtrees vs size-matched random sets.

    The locality test: if coherent subtrees aligned better than random class sets
    of equal size, the hierarchical trend would sit above the random trend. In
    practice the two trends overlap -- alignment tracks breadth (class count),
    not semantic coherence.
    """
    linear = results[results["transform"] == "linear"]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"hierarchical": "C0", "random": "C1"}
    labels = {"hierarchical": "hierarchical subtree", "random": "random class set"}
    for grouping, sub in linear.groupby("grouping"):
        ax.scatter(sub["n_classes"], sub["r2_eval"], s=8, alpha=0.15,
                   color=colors.get(grouping, "0.5"))
        centres, means, ses = _binned_trend(sub)
        ax.errorbar(centres, means, yerr=ses, lw=2.5, marker="o", capsize=3,
                    color=colors.get(grouping, "0.5"), label=labels.get(grouping, grouping))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("node breadth: #ImageNet classes (log scale)")
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("Locality control: coherent subtrees vs random class sets")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "real_vs_null.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overfitting(results: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter in-sample vs held-out R^2 for the hierarchical nodes.

    Points are sized by ``n_classes`` (node breadth, since fit-set size is fixed)
    and coloured by relation. Points far below the diagonal indicate overfitting
    (high in-sample, low held-out).
    """
    hier = results[results["grouping"] == "hierarchical"]
    kinds = [k for k in ["linear", "rigid"] if k in hier["transform"].unique()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        sub = hier[hier["transform"] == kind]
        sns.scatterplot(
            data=sub, x="r2_train", y="r2_eval", hue="relation",
            hue_order=_RELATION_ORDER, size="n_classes", sizes=(15, 200),
            alpha=0.6, ax=ax, legend=(kind == kinds[-1]),
        )
        lims = [min(sub["r2_eval"].min(), 0), 1]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_title(f"{kind} transform")
        ax.set_xlabel("in-sample $R^2$ (fit split)")
        ax.set_ylabel("held-out $R^2$")
    fig.suptitle("Overfitting: in-sample vs held-out alignment $R^2$")
    fig.tight_layout()
    path = out_dir / "overfitting.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pair_heatmap(results: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of mean held-out R^2 for each ordered model pair (linear)."""
    linear = results[(results["transform"] == "linear") & (results["grouping"] == "hierarchical")]
    grid = linear.pivot_table(
        index="source", columns="target_model", values="r2_eval", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Mean held-out $R^2$ (linear): source $\\to$ target")
    ax.set_xlabel("target model")
    ax.set_ylabel("source model")
    fig.tight_layout()
    path = out_dir / "pair_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_all(results_csv: Path, out_dir: Path | None = None) -> list[Path]:
    """Generate every figure from a results CSV, saved next to it by default."""
    results = pd.read_csv(results_csv)
    out_dir = out_dir or results_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_r2_vs_granularity(results, out_dir),
        plot_real_vs_null(results, out_dir),
        plot_overfitting(results, out_dir),
        plot_pair_heatmap(results, out_dir),
    ]
