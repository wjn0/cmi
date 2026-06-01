"""Hydra entrypoint: ``uv run python -m repsim.run``."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from repsim.experiment import run_experiment
from repsim.plots import plot_all


@hydra.main(version_base=None, config_path="../conf", config_name="hierarchically_local_similarity")
def main(cfg: DictConfig) -> None:
    """Run the experiment named by ``--config-name`` (default: this experiment)."""
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    run_experiment(cfg, output_dir=out_dir)
    plot_all(out_dir / "results.csv")


if __name__ == "__main__":
    main()
