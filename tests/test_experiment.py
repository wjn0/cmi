"""Tests for the experiment's per-node fit/held-out splitting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from repsim.experiment import _evaluate, _rows_by_class
from repsim.imagenet_hierarchy import HierarchyNode


def _fake_model(name: str) -> SimpleNamespace:
    """A stand-in exposing only the ``.spec.name`` that ``_evaluate`` reads."""
    return SimpleNamespace(spec=SimpleNamespace(name=name))


def _setup(n_fit: int, n_eval: int, n_jobs: int = 1):
    """Two models, two nodes (one broad, one too small), and random embeddings."""
    cfg = OmegaConf.create(
        {"seed": 0, "n_fit_samples": n_fit, "n_eval_samples": n_eval,
         "transforms": ["linear"], "n_jobs": n_jobs}
    )
    models = [_fake_model("a"), _fake_model("b")]
    # Broad node spans classes {0, 1} (plenty of rows); small node is class {2}.
    nodes = [
        HierarchyNode("broad", "broad", "ancestor", -1, (0, 1)),
        HierarchyNode("small", "small", "descendant", 1, (2,)),
    ]
    # 200 rows for the broad node's classes, only 3 rows for the small node's.
    sample_classes = np.array([0, 1] * 100 + [2] * 3)
    rng = np.random.default_rng(0)
    d = 3
    embeddings = {m.spec.name: rng.standard_normal((sample_classes.size, d)) for m in models}
    return cfg, {"t": nodes}, models, embeddings, sample_classes


def test_every_kept_node_uses_the_fixed_sample_count():
    """All records report exactly ``n_fit_samples`` fit and ``n_eval_samples`` held-out rows."""
    n_fit, n_eval = 10, 4
    records = _evaluate(*_setup(n_fit, n_eval))
    df = pd.DataFrame.from_records(records)
    assert (df["n_train"] == n_fit).all()
    assert (df["n_eval"] == n_eval).all()


def test_node_kept_only_when_it_supplies_fit_plus_holdout():
    """The broad node (200 rows) just fits n_fit + n_eval == 200; the small one cannot."""
    records = _evaluate(*_setup(n_fit=150, n_eval=50))
    assert {r["node"] for r in records} == {"broad"}


def test_nodes_without_enough_samples_are_skipped():
    """The small node (3 rows) cannot supply 10 fit + 4 held-out; only broad remains."""
    records = _evaluate(*_setup(n_fit=10, n_eval=4))
    assert {r["node"] for r in records} == {"broad"}


def test_parallel_eval_matches_serial():
    """Fanning the eval across workers gives identical records to the serial path.

    The per-node split is seeded from a spawned SeedSequence, so it must not depend
    on how many workers run the nodes.
    """
    serial = _evaluate(*_setup(n_fit=50, n_eval=20, n_jobs=1))
    parallel = _evaluate(*_setup(n_fit=50, n_eval=20, n_jobs=2))
    key = lambda r: (r["node"], r["source"], r["target_model"], r["transform"])
    assert sorted(serial, key=key) == sorted(parallel, key=key)


def test_rows_by_class_groups_every_row():
    """Each class maps to exactly its row positions, in ascending order."""
    sample_classes = np.array([2, 0, 1, 0, 2, 0])
    by_class = _rows_by_class(sample_classes)
    assert sorted(by_class) == [0, 1, 2]
    assert by_class[0].tolist() == [1, 3, 5]
    assert by_class[2].tolist() == [0, 4]
