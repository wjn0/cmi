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
    if "transform" not in results.columns:
        return metrics
    for transform, sub in results.groupby("transform"):
        metrics[f"{transform}/heldout_r2_mean"] = float(sub["r2_eval"].mean())
        metrics[f"{transform}/insample_r2_mean"] = float(sub["r2_train"].mean())
    return metrics


# Scalar config knobs worth tracking; experiment-specific ones are simply absent
# from other experiments' configs and are skipped.
_PARAM_KEYS = (
    "seed", "n_fit_samples", "n_eval_samples", "per_class_limit", "whiten",
    "device", "n_random_targets", "max_ancestor_levels", "max_descendant_levels",
)


def _params(cfg: DictConfig) -> dict[str, object]:
    """Flatten the config knobs worth tracking into MLflow params."""
    c = OmegaConf.to_container(cfg, resolve=True)
    params: dict[str, object] = {k: c[k] for k in _PARAM_KEYS if k in c}
    params["transforms"] = ",".join(c["transforms"])
    params["models"] = ",".join(m["name"] for m in c["models"])
    if "dataset" in c:  # granularity_similarity: a single dataset + split
        params["dataset"] = c["dataset"]["hf_id"]
        params["split"] = c["dataset"]["split"]
    if "datasets" in c:  # local_similarity_performance: a dataset list
        params["datasets"] = ",".join(d["hf_id"] for d in c["datasets"])
    return params


def _run_name_and_tags(
    cfg: DictConfig, experiment: str | None
) -> tuple[str, dict[str, str]]:
    """Compose a unique run name + filterable tags from the experiment and metric.

    The run name and the ``transforms``/``whiten`` tags encode the similarity
    metric, so a swept ``linear`` and ``rbf_cka`` land as distinct, comparable
    MLflow runs instead of colliding under one name.
    """
    transforms = "+".join(cfg.transforms)
    whiten = bool(cfg.get("whiten", False))
    metric = f"{transforms}{', whiten' if whiten else ''}"
    run_name = f"{experiment} [{metric}]" if experiment else metric
    tags = {"transforms": transforms, "whiten": str(whiten)}
    if experiment:
        tags["experiment"] = experiment
    return run_name, tags


def log_run(
    cfg: DictConfig,
    results: pd.DataFrame,
    artifacts: Sequence[Path],
    experiment: str | None = None,
    extra_tags: Mapping[str, str] | None = None,
) -> None:
    """Log one experiment run to MLflow (params, summary metrics, artifacts).

    No-op when ``cfg.mlflow.enabled`` is false or MLflow is not installed, so the
    experiment never fails just because tracking is unavailable. The run name and
    tags encode the experiment and similarity metric, so swept runs stay distinct.

    Args:
        cfg: The run config; its ``mlflow`` block selects the tracking URI and
            experiment name (``tracking_uri: null`` uses ``$MLFLOW_TRACKING_URI``).
        results: The long-format results DataFrame.
        artifacts: Files to attach to the run (``results.csv``, figures).
        experiment: The experiment config group choice (e.g.
            ``granularity_similarity/dinov2_scale``),
            used to name and tag the run.
        extra_tags: Optional extra MLflow tags.
    """
    mlflow_cfg = cfg.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping experiment tracking.")
        return

    run_name, tags = _run_name_and_tags(cfg, experiment)
    if extra_tags:
        tags.update(extra_tags)
    uri = mlflow_cfg.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "repsim"))
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params(_params(cfg))
        mlflow.log_metrics(_summary_metrics(results))
        for artifact in artifacts:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))
    log.info("Logged MLflow run %r to experiment %r (tracking_uri=%s).",
             run_name, mlflow_cfg.get("experiment_name", "repsim"), mlflow.get_tracking_uri())
