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

# Nature-style rcParams: Helvetica-like sans (Nimbus Sans), thin spines, no top/
# right frame, restrained type sizes, gridlines drawn per-axes (off by default).
_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "regular",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Okabe-Ito colourblind-safe qualitative palette (drops hard-to-read yellow).
_OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def _pair_label(df: pd.DataFrame) -> pd.Series:
    """Directional model-pair label ``source -> target_model`` (e.g. dinov2 → mae)."""
    return df["source"] + " → " + df["target_model"]


def _primary_transform(results: pd.DataFrame) -> str:
    """The transform to visualise: ``linear`` if present, else the first available.

    The figures show one fitted/similarity score per node; with several transforms
    present we plot ``linear`` (the default analysis), and otherwise fall back to
    whatever single transform was run (e.g. ``rbf_cka``).
    """
    kinds = set(results["transform"].unique())
    return "linear" if "linear" in kinds else sorted(kinds)[0]


def _metric_label(transform: str) -> str:
    """Axis label for the score column, which holds CKA for ``rbf_cka`` else R^2."""
    return "RBF CKA" if transform == "rbf_cka" else "$R^2$"


def _light_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Apply a faint background gridline on one axis, Nature-style."""
    ax.grid(axis=axis, color="0.92", linewidth=0.6, zorder=0)


def _annotate_example_nodes(
    ax: plt.Axes, sub: pd.DataFrame, xcol: str, ycol: str, n: int = 3
) -> None:
    """Label ``n`` hierarchical nodes with their synset name to anchor intuition.

    Splits the points into ``n`` equal-count bins along ``log2(xcol)`` (breadth)
    and, in each bin, labels the single node whose ``ycol`` deviates most from the
    overall mean. The labelled points are therefore y-axis outliers spread across
    granularity scales (fine → coarse), illustrating what the breadth axis means.
    Random null nodes are excluded -- only real synsets carry interpretable names.
    """
    real = sub[sub["grouping"] == "hierarchical"].dropna(subset=[ycol, xcol])
    if real.empty:
        return
    logx = np.log2(real[xcol].to_numpy())
    edges = np.quantile(logx, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9  # include the widest node in the last bin
    which = np.clip(np.digitize(logx, edges) - 1, 0, n - 1)
    ymean = real[ycol].to_numpy().mean()
    for b in range(n):
        bin_rows = real[which == b]
        if bin_rows.empty:
            continue
        pick = bin_rows.iloc[(bin_rows[ycol] - ymean).abs().to_numpy().argmax()]
        ax.annotate(
            pick["node_label"], (pick[xcol], pick[ycol]),
            xytext=(0, 9 if b % 2 == 0 else -12), textcoords="offset points",
            fontsize=6.5, ha="center", color="0.15", zorder=5,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"),
        )


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
    """Held-out R^2 vs granularity, one trendline per directional model pair.

    Each ordered model pair (source -> target_model) gets its own colour: the
    scatter shows every hierarchical node (with horizontal jitter so coincident
    breadths separate) and a least-squares trendline fit in log2(breadth) space.
    Tests the core claim per pair: alignment degrades as the region of the
    hierarchy broadens (coarsens). The random null is excluded.
    """
    transform = _primary_transform(results)
    hier = results[
        (results["grouping"] == "hierarchical") & (results["transform"] == transform)
    ].copy()
    hier["pair"] = _pair_label(hier)
    pairs = sorted(hier["pair"].unique())
    palette = {p: _OKABE_ITO[i % len(_OKABE_ITO)] for i, p in enumerate(pairs)}
    rng = np.random.default_rng(0)

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        log_all = np.log2(hier["n_classes"].to_numpy())
        xgrid = np.linspace(log_all.min(), log_all.max(), 100)
        for pair in pairs:
            sub = hier[hier["pair"] == pair]
            x = np.log2(sub["n_classes"].to_numpy())
            y = sub["r2_eval"].to_numpy()
            jitter = rng.uniform(-0.13, 0.13, size=x.size)  # multiplicative on log2 axis
            ax.scatter(2 ** (x + jitter), y, s=9, color=palette[pair],
                       alpha=0.30, linewidths=0, zorder=2)
            if x.size >= 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(2 ** xgrid, slope * xgrid + intercept, color=palette[pair],
                        lw=1.8, solid_capstyle="round", zorder=3, label=pair)
        _annotate_example_nodes(ax, hier, "n_classes", "r2_eval")
        ax.set_xscale("log", base=2)
        _light_grid(ax, "y")
        ax.set_xlabel("Granularity — no. of ImageNet classes (finer → coarser)")
        ax.set_ylabel(f"Alignment ({_metric_label(transform)}, held-out)")
        ax.set_title("Cross-model alignment vs hierarchy granularity")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  title="Model pair", title_fontsize=8, handlelength=1.3)
        path = out_dir / "r2_vs_granularity.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_real_vs_null(results: pd.DataFrame, out_dir: Path) -> Path:
    """Held-out R^2 vs breadth for real subtrees vs size-matched random sets.

    The locality test: if coherent subtrees aligned better than random class sets
    of equal size, the hierarchical trend would sit above the random trend. In
    practice the two trends overlap -- alignment tracks breadth (class count),
    not semantic coherence.
    """
    transform = _primary_transform(results)
    chosen = results[results["transform"] == transform]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"hierarchical": "C0", "random": "C1"}
    labels = {"hierarchical": "hierarchical subtree", "random": "random class set"}
    for grouping, sub in chosen.groupby("grouping"):
        ax.scatter(sub["n_classes"], sub["r2_eval"], s=8, alpha=0.15,
                   color=colors.get(grouping, "0.5"))
        centres, means, ses = _binned_trend(sub)
        ax.errorbar(centres, means, yerr=ses, lw=2.5, marker="o", capsize=3,
                    color=colors.get(grouping, "0.5"), label=labels.get(grouping, grouping))
    _annotate_example_nodes(ax, chosen, "n_classes", "r2_eval")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("node breadth: #ImageNet classes (log scale)")
    ax.set_ylabel(f"held-out {_metric_label(transform)}")
    ax.set_title("Locality control: coherent subtrees vs random class sets")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "real_vs_null.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_overfitting(results: pd.DataFrame, out_dir: Path) -> Path:
    """Per-model-pair scatter of in-sample vs held-out R^2 (linear, hierarchical).

    One panel per directional model pair (source -> target_model); points are
    coloured by node breadth on a log scale (shared colourbar). Points far below
    the dashed identity line indicate overfitting (high in-sample, low held-out),
    which the colour reveals concentrates at the finest, narrowest nodes.
    """
    transform = _primary_transform(results)
    hier = results[
        (results["grouping"] == "hierarchical") & (results["transform"] == transform)
    ].copy()
    hier["pair"] = _pair_label(hier)
    pairs = sorted(hier["pair"].unique())
    log_b = np.log2(hier["n_classes"].to_numpy())
    norm = plt.Normalize(log_b.min(), log_b.max())

    ncol = 3
    nrow = -(-len(pairs) // ncol)
    with plt.rc_context(_NATURE_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.55 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        lo = min(hier["r2_train"].min(), hier["r2_eval"].min(), 0.0)
        lims = [lo, 1.0]
        sc = None
        for ax, pair in zip(axes.flat, pairs):
            sub = hier[hier["pair"] == pair]
            ax.plot(lims, lims, ls="--", color="0.6", lw=0.8, zorder=1)
            sc = ax.scatter(sub["r2_train"], sub["r2_eval"],
                            c=np.log2(sub["n_classes"]), cmap="viridis", norm=norm,
                            s=14, alpha=0.85, linewidths=0, zorder=2)
            ax.set_title(pair)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")
        for ax in axes.flat[len(pairs):]:
            ax.set_visible(False)

        metric = _metric_label(transform)
        fig.supxlabel(f"In-sample {metric} (fit split)", fontsize=9)
        fig.supylabel(f"Held-out {metric}", fontsize=9)
        fig.suptitle(f"Overfitting by model pair: in-sample vs held-out {metric}",
                     fontsize=10)
        ticks = list(range(int(np.ceil(log_b.min())), int(np.floor(log_b.max())) + 1))
        cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02, ticks=ticks)
        cbar.set_label("Granularity (no. classes)", fontsize=8)
        cbar.ax.set_yticklabels([2 ** t for t in ticks])
        cbar.ax.tick_params(labelsize=7)
        path = out_dir / "overfitting.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_pair_heatmap(results: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of mean held-out score for each ordered model pair."""
    transform = _primary_transform(results)
    chosen = results[(results["transform"] == transform) & (results["grouping"] == "hierarchical")]
    grid = chosen.pivot_table(
        index="source", columns="target_model", values="r2_eval", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Mean held-out {_metric_label(transform)} ({transform}): source $\\to$ target")
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
