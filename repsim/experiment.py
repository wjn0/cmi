"""The ``hierarchically_local_similarity`` experiment.

Pretrained vision models are embedded once over the full 1000-class ImageNet pool.
For each of several target WordNet nodes (an explicit list plus randomly sampled
ones) we build a connected chain of hierarchy nodes -- a few descendant levels
below the target, the target, and a few ancestors above it -- spanning a range of
subtree breadths. At each node we fit the optimal linear (and optionally rigid)
transform between every ordered pair of models and score R^2 in-sample and on a
held-out split. Because R^2 depends on fit-set size, each node is subsampled to a
fixed ``n_fit_samples`` fit and ``n_eval_samples`` held-out images (a disjoint
random split of that node's images), so alignment quality is comparable across
nodes of differing breadth; nodes that cannot supply both counts are skipped.

To separate semantic locality from sheer breadth (class count), the same scoring
is applied to size-matched "null" nodes whose classes are drawn at random from
the full pool. The result shows how cross-model alignment quality varies with
hierarchy granularity, and whether coherent subtrees align better than random
class sets of equal size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from repsim.imagenet_hierarchy import (
    HierarchyNode,
    all_class_indices,
    build_nodes,
    random_grouping_nodes,
    sample_target_nodes,
)
from repsim.inference import Embeddings, build_sample_index, extract_embeddings
from repsim.models import LoadedModel, ModelSpec
from repsim.transforms import evaluate_transform

_NULL_KEY = "__random__"


def _resolve_device(requested: str) -> str:
    """Validate and return the requested device.

    Fails loudly if CUDA was requested but is unavailable, rather than silently
    falling back to CPU (which would make inference ~100x slower). Pass
    ``device=cpu`` explicitly to run on CPU.
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda requested but torch.cuda.is_available() is False. "
            "The allocated GPU node's driver may be incompatible with this torch "
            "build, or no GPU was allocated. Request a modern GPU (e.g. "
            "--gres=gpu:L40S:1) or pass device=cpu explicitly."
        )
    return requested


def _resolve_targets(cfg: DictConfig) -> list[str]:
    """Return the explicit target nodes plus the randomly sampled ones."""
    explicit = list(cfg.target_nodes)
    sampled = sample_target_nodes(
        n=cfg.n_random_targets,
        exclude=explicit,
        min_classes=cfg.random_target_min_classes,
        max_classes=cfg.random_target_max_classes,
        seed=cfg.seed,
    )
    return explicit + sampled


def _build_all_nodes(cfg: DictConfig) -> dict[str, list[HierarchyNode]]:
    """Build every target's hierarchy chain plus the size-matched null nodes."""
    target_nodes: dict[str, list[HierarchyNode]] = {}
    for target in _resolve_targets(cfg):
        target_nodes[target] = build_nodes(
            target, cfg.max_ancestor_levels, cfg.max_descendant_levels
        )
    sizes = [len(n.class_indices) for nodes in target_nodes.values() for n in nodes]
    target_nodes[_NULL_KEY] = random_grouping_nodes(
        sizes, cfg.null_replicates, cfg.seed
    )
    return target_nodes


def run_experiment(cfg: DictConfig, output_dir: Path | None = None) -> pd.DataFrame:
    """Run the hierarchically-local similarity experiment and persist results.

    Args:
        cfg: Hydra configuration (see ``conf/hierarchically_local_similarity.yaml``).
        output_dir: Directory to write ``results.csv`` to (defaults to cwd).

    Returns:
        A long-format DataFrame with one row per (target, node, model pair,
        direction, transform kind) and its in-sample and held-out R^2.
    """
    device = _resolve_device(cfg.device)
    cache_dir = Path(cfg.cache_dir)

    models = [LoadedModel(ModelSpec(**OmegaConf.to_container(m)), device) for m in cfg.models]
    max_dim = max(m.dim for m in models)
    per_class_limit = cfg.per_class_limit or max_dim
    if cfg.n_fit_samples <= max_dim:
        raise ValueError(
            f"n_fit_samples={cfg.n_fit_samples} must exceed the largest model dim "
            f"({max_dim}), else the linear transform is under-determined."
        )

    target_nodes = _build_all_nodes(cfg)
    n_real = sum(len(v) for k, v in target_nodes.items() if k != _NULL_KEY)
    print(f"{len(target_nodes) - 1} targets, {n_real} hierarchy nodes, "
          f"{len(target_nodes[_NULL_KEY])} null nodes. "
          f"Inference pool: all {len(all_class_indices())} classes.")
    print(f"Per-class limit: {per_class_limit}; fixed fit/eval samples per node: "
          f"{cfg.n_fit_samples}/{cfg.n_eval_samples} (d_model={max_dim}).")

    dataset = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
    labels = np.array(dataset["label"])
    index = build_sample_index(labels, all_class_indices(), per_class_limit, cfg.seed)
    embeddings = extract_embeddings(
        models, dataset, index, cfg.batch_size,
        cache_dir, cfg.dataset.hf_id, cfg.dataset.split,
    )

    records = _evaluate(cfg, target_nodes, models, embeddings, index.classes)
    results = pd.DataFrame.from_records(records)

    out_dir = output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    print(f"Wrote {len(results)} rows to {out_dir / 'results.csv'}")
    return results


def _evaluate(
    cfg: DictConfig,
    target_nodes: dict[str, list[HierarchyNode]],
    models,
    embeddings: Embeddings,
    sample_classes: np.ndarray,
) -> list[dict]:
    """Fit and score transforms for every target, node, ordered pair and kind.

    Each node's images are split into a fixed ``n_fit_samples`` fit set and a
    disjoint ``n_eval_samples`` held-out set (shared across all model pairs and
    transforms at that node), so the fit-set size is constant -- removing the
    sample-size confound when comparing R^2 across nodes of differing breadth.
    Nodes that cannot supply both counts are skipped.
    """
    n_fit, n_eval = cfg.n_fit_samples, cfg.n_eval_samples
    rng = np.random.default_rng(cfg.seed)
    records: list[dict] = []
    skipped = 0
    for target, nodes in target_nodes.items():
        for node in nodes:
            rows = np.flatnonzero(np.isin(sample_classes, node.class_indices))
            if rows.size < n_fit + n_eval:
                skipped += 1
                continue
            shuffled = rng.permutation(rows)
            fit_idx, eval_idx = shuffled[:n_fit], shuffled[n_fit : n_fit + n_eval]
            for src in models:
                for tgt in models:
                    if src.spec.name == tgt.spec.name:
                        continue
                    xs_tr, ys_tr = embeddings[src.spec.name][fit_idx], embeddings[tgt.spec.name][fit_idx]
                    xs_ev, ys_ev = embeddings[src.spec.name][eval_idx], embeddings[tgt.spec.name][eval_idx]
                    for kind in cfg.transforms:
                        r2_train, r2_eval = evaluate_transform(kind, xs_tr, ys_tr, xs_ev, ys_ev)
                        records.append({
                            "target": target,
                            "node": node.key,
                            "node_label": node.label,
                            "relation": node.relation,
                            "grouping": node.grouping,
                            "depth": node.depth,
                            "n_classes": len(node.class_indices),
                            "n_train": n_fit,
                            "n_eval": n_eval,
                            "source": src.spec.name,
                            "target_model": tgt.spec.name,
                            "transform": kind,
                            "r2_train": r2_train,
                            "r2_eval": r2_eval,
                        })
    print(f"Skipped {skipped} nodes below the minimum-sample threshold.")
    return records
