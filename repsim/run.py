"""Hydra entrypoint: ``uv run python -m repsim.run``."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from repsim.experiment import run_experiment
from repsim.log import setup_logging
from repsim.plots import plot_all
from repsim.tracking import log_run


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the experiment selected by the ``experiment`` config group.

    Defaults to ``experiment=hierarchically_local_similarity``; override with
    ``experiment=dinov2_scale_similarity`` or sweep both with
    ``-m experiment=hierarchically_local_similarity,dinov2_scale_similarity``.
    """
    setup_logging()
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)
    results = run_experiment(cfg, output_dir=out_dir)
    figures = plot_all(out_dir / "results.csv")
    experiment = hydra_cfg.runtime.choices.get("experiment")
    log_run(
        cfg, results,
        artifacts=[out_dir / "results.csv", *figures],
        run_name=experiment,
        tags={"experiment": experiment} if experiment else None,
    )


if __name__ == "__main__":
    main()
