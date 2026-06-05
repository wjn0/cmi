"""Hydra entrypoint: ``uv run python -m repsim.run``."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from repsim.experiments import get_experiment
from repsim.log import setup_logging
from repsim.tracking import log_run


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment selected by the ``experiment`` config group.

    Each config under ``conf/experiment/`` sets ``experiment_name``, which picks
    the matching ``repsim/experiments/<experiment_name>.py`` module. Sweep, e.g.::

        uv run python -m repsim.run -m \\
          experiment=granularity_similarity/cross_model,granularity_similarity/dinov2_scale \\
          similarity=linear,rbf_cka
    """
    setup_logging()
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)
    experiment = get_experiment(cfg.experiment_name)
    results = experiment.run_experiment(cfg, output_dir=out_dir)
    figures = experiment.plot_results(out_dir / "results.csv")
    log_run(
        cfg, results,
        artifacts=[out_dir / "results.csv", *figures],
        experiment=hydra_cfg.runtime.choices.get("experiment"),
    )


if __name__ == "__main__":
    main()
