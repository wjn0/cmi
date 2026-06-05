"""Tests for the experiment registry and the experiment-module interface."""

from __future__ import annotations

import pytest
from hydra import compose, initialize

from repsim.experiments import _EXPERIMENTS, get_experiment


def test_every_experiment_exposes_the_module_interface():
    """Each registered experiment provides run_experiment and plot_results."""
    for name in _EXPERIMENTS:
        module = get_experiment(name)
        assert callable(module.run_experiment)
        assert callable(module.plot_results)


def test_unknown_experiment_raises_with_available_names():
    with pytest.raises(KeyError, match="granularity_similarity"):
        get_experiment("no_such_experiment")


@pytest.mark.parametrize(
    "choice, expected_name",
    [
        ("granularity_similarity/cross_model", "granularity_similarity"),
        ("granularity_similarity/dinov2_scale", "granularity_similarity"),
        ("local_similarity_performance", "local_similarity_performance"),
    ],
)
def test_every_experiment_config_composes_and_dispatches(choice, expected_name):
    """Each conf/experiment option composes cleanly and names a registered experiment."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose("config", overrides=[f"experiment={choice}"])
    assert cfg.experiment_name == expected_name
    get_experiment(cfg.experiment_name)  # must not raise
    assert len(cfg.models) >= 2  # similarity needs at least one model pair


def test_granularity_configs_inherit_the_shared_design():
    """The _shared design keys land in both granularity configurations."""
    with initialize(version_base=None, config_path="../conf"):
        cross = compose("config", overrides=["experiment=granularity_similarity/cross_model"])
        scale = compose("config", overrides=["experiment=granularity_similarity/dinov2_scale"])
    for cfg in (cross, scale):
        assert cfg.dataset.hf_id == "benjamin-paine/imagenet-1k-256x256"
        assert cfg.n_eval_samples == 250
    # Configuration-specific keys still differ.
    assert cross.n_fit_samples != scale.n_fit_samples


def test_local_similarity_performance_config_has_datasets_and_probe():
    """The experiment config exposes the PCam/CelebA dataset list and a probe block."""
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose("config", overrides=["experiment=local_similarity_performance"])
    names = {d.name for d in cfg.datasets}
    assert {"pcam", "celeba"} <= names
    for d in cfg.datasets:
        assert d.hf_id and d.label_column and d.train_split and d.test_split
    assert cfg.probe.kind == "logistic"
    assert cfg.n_train_per_class > 0 and cfg.n_test_per_class > 0
