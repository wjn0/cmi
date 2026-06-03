"""Optional MLflow logging of experiment params, summary metrics, and artifacts.

Pulling results off the cluster's shared filesystem to inspect them is painful, so
every run is also logged to MLflow: the config as params, a handful of aggregate
scores as metrics, and ``results.csv`` plus the figures as artifacts. Browsing the
MLflow UI then beats hunting through ``outputs/`` by hand. Tracking is best-effort
-- if MLflow is missing or disabled, the experiment still runs and writes its CSV.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _summary_metrics(results: pd.DataFrame) -> dict[str, float]:
    """Aggregate held-out / in-sample scores per transform for at-a-glance tracking."""
    metrics: dict[str, float] = {"n_rows": float(len(results))}
    for transform, sub in results.groupby("transform"):
        metrics[f"{transform}/heldout_r2_mean"] = float(sub["r2_eval"].mean())
        metrics[f"{transform}/insample_r2_mean"] = float(sub["r2_train"].mean())
    return metrics


def _params(cfg: DictConfig) -> dict[str, object]:
    """Flatten the config knobs worth tracking into MLflow params."""
    c = OmegaConf.to_container(cfg, resolve=True)
    return {
        "seed": c["seed"],
        "n_fit_samples": c["n_fit_samples"],
        "n_eval_samples": c["n_eval_samples"],
        "per_class_limit": c["per_class_limit"],
        "transforms": ",".join(c["transforms"]),
        "whiten": c["whiten"],
        "device": c.get("device"),
        "models": ",".join(m["name"] for m in c["models"]),
        "dataset": c["dataset"]["hf_id"],
        "split": c["dataset"]["split"],
        "n_random_targets": c["n_random_targets"],
        "max_ancestor_levels": c["max_ancestor_levels"],
        "max_descendant_levels": c["max_descendant_levels"],
    }


def log_run(
    cfg: DictConfig,
    results: pd.DataFrame,
    artifacts: Sequence[Path],
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Log one experiment run to MLflow (params, summary metrics, artifacts).

    No-op when ``cfg.mlflow.enabled`` is false or MLflow is not installed, so the
    experiment never fails just because tracking is unavailable.

    Args:
        cfg: The run config; its ``mlflow`` block selects the tracking URI and
            experiment name (``tracking_uri: null`` uses the local ``./mlruns`` store
            or ``$MLFLOW_TRACKING_URI``).
        results: The long-format results DataFrame.
        artifacts: Files to attach to the run (``results.csv``, figures).
        run_name: Optional MLflow run name (e.g. the experiment group).
        tags: Optional MLflow tags.
    """
    mlflow_cfg = cfg.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping experiment tracking.")
        return

    uri = mlflow_cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "repsim"))
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(dict(tags))
        mlflow.log_params(_params(cfg))
        mlflow.log_metrics(_summary_metrics(results))
        for artifact in artifacts:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))
    log.info("Logged run to MLflow experiment %r (tracking_uri=%s).",
             mlflow_cfg.get("experiment_name", "repsim"), mlflow.get_tracking_uri())
