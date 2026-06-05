"""Tests for the local_similarity_performance experiment's stages.

These exercise the pure-numpy pieces (probe, asymmetric pairwise similarity, and
the out-predictivity aggregation) without loading models or downloading datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from repsim.experiments.local_similarity_performance import (
    _pairwise_similarity,
    _probe_accuracy,
)
from repsim.plots import _out_predictivity


def _cfg(**probe):
    """A minimal stand-in config exposing only the ``probe`` block used by the probe."""
    from omegaconf import OmegaConf

    return OmegaConf.create({"probe": {"kind": "logistic", **probe}})


def test_probe_recovers_a_linearly_separable_task():
    """A standardised logistic probe scores ~1.0 on a separable held-out split."""
    rng = np.random.default_rng(0)
    train = np.vstack([rng.normal(-3, 1, (100, 8)), rng.normal(3, 1, (100, 8))])
    train_y = np.array([0] * 100 + [1] * 100)
    test = np.vstack([rng.normal(-3, 1, (50, 8)), rng.normal(3, 1, (50, 8))])
    test_y = np.array([0] * 50 + [1] * 50)
    acc = _probe_accuracy(train, train_y, test, test_y, _cfg())
    assert acc > 0.95


def test_pairwise_similarity_is_asymmetric_for_linear():
    """When B is a rank-deficient linear image of A, A->B predicts far better than B->A.

    A is full-rank; B = A @ M with M rank-2, so B is exactly linear in A (R^2 ~ 1)
    but A is not recoverable from the rank-2 B (R^2 < 1). The directional R^2s must
    therefore differ -- the asymmetry the experiment relies on.
    """
    rng = np.random.default_rng(1)
    a_train, a_test = rng.normal(size=(300, 4)), rng.normal(size=(120, 4))
    m = rng.normal(size=(4, 2)) @ rng.normal(size=(2, 4))  # rank-2 map
    train = {"A": a_train, "B": a_train @ m}
    test = {"A": a_test, "B": a_test @ m}
    scores = _pairwise_similarity(train, test, transforms=["linear"], whiten=False)

    assert set(scores) == {("A", "B"), ("B", "A")}
    r2_a_to_b = scores[("A", "B")]["linear"][1]
    r2_b_to_a = scores[("B", "A")]["linear"][1]
    assert r2_a_to_b > 0.99
    assert r2_a_to_b > r2_b_to_a + 0.05


def test_out_predictivity_aggregates_source_r2_and_dedups_accuracy():
    """Out-predictivity is the per-(dataset, source) mean held-out R^2; accuracy dedups."""
    results = pd.DataFrame([
        {"dataset": "d", "source": "A", "target_model": "B", "transform": "linear",
         "r2_eval": 0.8, "acc_source": 0.9, "acc_target": 0.7},
        {"dataset": "d", "source": "A", "target_model": "C", "transform": "linear",
         "r2_eval": 0.6, "acc_source": 0.9, "acc_target": 0.5},
        {"dataset": "d", "source": "B", "target_model": "A", "transform": "linear",
         "r2_eval": 0.4, "acc_source": 0.7, "acc_target": 0.9},
    ])
    table = _out_predictivity(results, "linear").set_index("model")
    assert table.loc["A", "pred_out"] == 0.7  # mean(0.8, 0.6)
    assert table.loc["A", "accuracy"] == 0.9
    assert table.loc["B", "pred_out"] == 0.4
