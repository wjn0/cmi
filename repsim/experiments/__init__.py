"""Experiment registry: one module per experiment, dispatched by name.

Each experiment module exposes:

- ``run_experiment(cfg, output_dir) -> pd.DataFrame`` -- run end-to-end and
  write ``results.csv`` to ``output_dir``.
- ``plot_results(results_csv) -> list[Path]`` -- regenerate the experiment's
  figures from a results CSV.

The Hydra config selects the experiment via the ``experiment_name`` key (set by
every config under ``conf/experiment/``); :func:`get_experiment` maps that name
to its module.
"""

from __future__ import annotations

from types import ModuleType

from repsim.experiments import granularity_similarity, local_similarity_performance

_EXPERIMENTS: dict[str, ModuleType] = {
    "granularity_similarity": granularity_similarity,
    "local_similarity_performance": local_similarity_performance,
}


def get_experiment(name: str) -> ModuleType:
    """Return the experiment module registered under ``name``.

    Args:
        name: The ``experiment_name`` from the composed config.

    Raises:
        KeyError: If no experiment is registered under ``name``.
    """
    try:
        return _EXPERIMENTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown experiment {name!r}; available: {sorted(_EXPERIMENTS)}"
        ) from None
