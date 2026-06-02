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

import logging
import time
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
from repsim.transforms import centered_rbf_gram, cka_from_grams, evaluate_transform, fit_whitening

log = logging.getLogger(__name__)

_NULL_KEY = "__random__"
_LOG_EVERY = 25  # log eval progress every this many nodes


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
    """Build every target's hierarchy chain plus the size-matched null nodes.

    A synset is frequently reached by several targets -- broad ancestors such as
    ``entity.n.01`` especially -- so each synset is kept only under the first
    target that introduces it. Evaluating the same synset once (rather than
    redundantly per target) avoids over-weighting coarse nodes in the trends and
    wasting compute. Null nodes have unique keys and are unaffected.
    """
    target_nodes: dict[str, list[HierarchyNode]] = {}
    seen: set[str] = set()
    for target in _resolve_targets(cfg):
        chain = build_nodes(target, cfg.max_ancestor_levels, cfg.max_descendant_levels)
        fresh = [n for n in chain if n.key not in seen]
        seen.update(n.key for n in fresh)
        target_nodes[target] = fresh
    sizes = [len(n.class_indices) for nodes in target_nodes.values() for n in nodes]
    target_nodes[_NULL_KEY] = random_grouping_nodes(
        sizes, cfg.null_replicates, cfg.seed
    )
    return target_nodes


def run_experiment(cfg: DictConfig, output_dir: Path | None = None) -> pd.DataFrame:
    """Run the hierarchically-local similarity experiment and persist results.

    Args:
        cfg: Hydra configuration (``conf/config.yaml`` + the selected
            ``conf/experiment/*.yaml``).
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
    if cfg.get("whiten", False) and "rbf_cka" in cfg.transforms:
        raise ValueError(
            "whiten=true is incompatible with the rbf_cka transform: RBF CKA "
            "double-centres its kernel, so it must see raw (un-whitened) "
            "embeddings. Run rbf_cka with whiten=false."
        )

    target_nodes = _build_all_nodes(cfg)
    n_real = sum(len(v) for k, v in target_nodes.items() if k != _NULL_KEY)
    log.info(
        "%d targets, %d hierarchy nodes, %d null nodes. Inference pool: all %d classes.",
        len(target_nodes) - 1, n_real, len(target_nodes[_NULL_KEY]), len(all_class_indices()),
    )
    log.info(
        "Per-class limit: %d; fixed fit/eval samples per node: %d/%d (d_model=%d); whiten=%s.",
        per_class_limit, cfg.n_fit_samples, cfg.n_eval_samples, max_dim, cfg.get("whiten", False),
    )

    dataset = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
    labels = np.array(dataset["label"])
    index = build_sample_index(labels, all_class_indices(), per_class_limit, cfg.seed)
    embeddings = extract_embeddings(
        models, dataset, index, cfg.batch_size,
        cache_dir, cfg.dataset.hf_id, cfg.dataset.split,
        num_workers=cfg.get("num_workers", 8),
    )

    records = _evaluate(cfg, target_nodes, models, embeddings, index.classes)
    results = pd.DataFrame.from_records(records)

    out_dir = output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    log.info("Wrote %d rows to %s", len(results), out_dir / "results.csv")
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

    When ``cfg.whiten`` is set, each model's embeddings are PCA-whitened per node
    (mean-centred, decorrelated, scaled to unit variance) with the whitening fitted
    on that node's fit split and applied to both splits. R^2 is then measured in the
    target's whitened space, where every dimension contributes equally to SS_tot
    rather than high-variance directions dominating.

    For the ``rbf_cka`` transform each model's centred RBF Gram matrix is built once
    per node (per split) and reused across every model pair, rather than rebuilt for
    each ordered pair.
    """
    n_fit, n_eval = cfg.n_fit_samples, cfg.n_eval_samples
    whiten = cfg.get("whiten", False)
    use_cka = "rbf_cka" in cfg.transforms
    rng = np.random.default_rng(cfg.seed)
    records: list[dict] = []
    skipped = 0
    all_nodes = [(target, node) for target, nodes in target_nodes.items() for node in nodes]
    log.info("Evaluating %d nodes (whiten=%s); logging every %d nodes.",
             len(all_nodes), whiten, _LOG_EVERY)
    start = time.monotonic()
    for i, (target, node) in enumerate(all_nodes, 1):
        rows = np.flatnonzero(np.isin(sample_classes, node.class_indices))
        if rows.size < n_fit + n_eval:
            skipped += 1
            continue
        shuffled = rng.permutation(rows)
        fit_idx, eval_idx = shuffled[:n_fit], shuffled[n_fit : n_fit + n_eval]
        # Fit/apply whitening once per model per node (it depends only on the
        # model's own fit split), not redundantly inside every ordered pair.
        fit_emb = {m.spec.name: embeddings[m.spec.name][fit_idx] for m in models}
        eval_emb = {m.spec.name: embeddings[m.spec.name][eval_idx] for m in models}
        if whiten:
            for name in list(fit_emb):
                w = fit_whitening(fit_emb[name])
                fit_emb[name], eval_emb[name] = w.apply(fit_emb[name]), w.apply(eval_emb[name])
        # Build each model's centred RBF Gram once per node (reused across pairs).
        gram_fit = {n: centered_rbf_gram(e) for n, e in fit_emb.items()} if use_cka else {}
        gram_eval = {n: centered_rbf_gram(e) for n, e in eval_emb.items()} if use_cka else {}
        for src in models:
            for tgt in models:
                if src.spec.name == tgt.spec.name:
                    continue
                xs_tr, ys_tr = fit_emb[src.spec.name], fit_emb[tgt.spec.name]
                xs_ev, ys_ev = eval_emb[src.spec.name], eval_emb[tgt.spec.name]
                for kind in cfg.transforms:
                    if kind == "rbf_cka":
                        r2_train = cka_from_grams(gram_fit[src.spec.name], gram_fit[tgt.spec.name])
                        r2_eval = cka_from_grams(gram_eval[src.spec.name], gram_eval[tgt.spec.name])
                    else:
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
                        "whiten": bool(whiten),
                        "r2_train": r2_train,
                        "r2_eval": r2_eval,
                    })
        if i % _LOG_EVERY == 0 or i == len(all_nodes):
            elapsed = time.monotonic() - start
            log.info("  %d/%d nodes (%d evaluated, %d skipped, %d rows, %.0fs elapsed).",
                     i, len(all_nodes), i - skipped, skipped, len(records), elapsed)
    log.info("Skipped %d nodes below the minimum-sample threshold; %d rows total.",
             skipped, len(records))
    return records
