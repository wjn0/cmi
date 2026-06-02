"""Smoke tests for the results visualizations."""

import numpy as np
import pandas as pd

from repsim.plots import plot_all


def _synthetic_results(seed: int = 0, kinds=("linear", "rigid")) -> pd.DataFrame:
    """Build a small results-shaped DataFrame spanning granularity and groupings."""
    rng = np.random.default_rng(seed)
    rows = []
    # Hierarchical nodes across a range of breadths, for several targets.
    for t in range(4):
        for depth, relation, k in [(1, "descendant", 4), (0, "self", 12), (-1, "ancestor", 60)]:
            for source, target in [("a", "b"), ("b", "a")]:
                for kind in kinds:
                    rows.append({
                        "target": f"t{t}", "node": f"t{t}.{relation}",
                        "node_label": relation, "relation": relation,
                        "grouping": "hierarchical", "depth": depth, "n_classes": k,
                        "n_train": 3840, "n_eval": 250,
                        "source": source, "target_model": target, "transform": kind,
                        "r2_train": float(rng.uniform(0.4, 0.9)),
                        "r2_eval": float(rng.uniform(0.2, 0.7)),
                    })
    # Size-matched random control nodes.
    for k in [4, 12, 60]:
        for source, target in [("a", "b"), ("b", "a")]:
            for kind in kinds:
                rows.append({
                    "target": "__random__", "node": f"random_k{k}_r0",
                    "node_label": f"random(k={k})", "relation": "random",
                    "grouping": "random", "depth": 0, "n_classes": k,
                    "n_train": 3840, "n_eval": 250,
                    "source": source, "target_model": target, "transform": kind,
                    "r2_train": float(rng.uniform(0.4, 0.9)),
                    "r2_eval": float(rng.uniform(0.2, 0.7)),
                })
    return pd.DataFrame(rows)


def test_plot_all_writes_all_figures(tmp_path):
    csv = tmp_path / "results.csv"
    _synthetic_results().to_csv(csv, index=False)
    paths = plot_all(csv)
    assert len(paths) == 4
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_plot_all_handles_rbf_cka_only_results(tmp_path):
    # No `linear` rows: the plots must fall back to the rbf_cka score, not crash.
    csv = tmp_path / "results.csv"
    _synthetic_results(kinds=("rbf_cka",)).to_csv(csv, index=False)
    paths = plot_all(csv)
    assert len(paths) == 4
    for path in paths:
        assert path.exists() and path.stat().st_size > 0
