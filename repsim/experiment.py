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
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from joblib import Parallel, delayed
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


def _resolve_n_jobs(requested: int) -> int:
    """Resolve a joblib ``n_jobs`` value, respecting the SLURM CPU allocation.

    ``-1`` (or any non-positive value) means "use every available CPU", but the
    machine's total core count is the wrong number under SLURM -- the job is only
    allocated ``--cpus-per-task`` cores. ``os.sched_getaffinity`` returns the cores
    the process is actually pinned to, which is exactly that allocation, so we use
    it to avoid oversubscribing the node.
    """
    if requested > 0:
        return requested
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms
        return os.cpu_count() or 1


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


def _rows_by_class(sample_classes: np.ndarray) -> dict[int, np.ndarray]:
    """Map each class index to its (ascending) row positions in ``sample_classes``.

    Built once so each node selects its rows by concatenating a handful of per-class
    row arrays, instead of rescanning the whole sample with ``np.isin`` for every one
    of the hundreds of nodes.
    """
    order = np.argsort(sample_classes, kind="stable")
    sorted_classes = sample_classes[order]
    boundaries = np.flatnonzero(np.diff(sorted_classes)) + 1
    keys = sorted_classes[np.concatenate(([0], boundaries))]
    return {int(k): rows for k, rows in zip(keys, np.split(order, boundaries))}


def _node_rows(class_indices, rows_by_class: dict[int, np.ndarray]) -> np.ndarray:
    """Return the sorted sample-row positions belonging to a node's classes."""
    parts = [rows_by_class[c] for c in class_indices if c in rows_by_class]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def _evaluate_node(
    target: str,
    node: HierarchyNode,
    model_names: list[str],
    embeddings: Embeddings,
    rows_by_class: dict[int, np.ndarray],
    n_fit: int,
    n_eval: int,
    whiten: bool,
    use_cka: bool,
    transforms: list[str],
    seed: np.random.SeedSequence,
) -> list[dict] | None:
    """Fit and score every ordered model pair at one node -- one parallel task.

    Returns the node's records, or ``None`` if it cannot supply ``n_fit + n_eval``
    images. Self-contained so it runs cleanly in a worker process: it takes model
    *names* (not the loaded GPU models, which cannot be pickled) and reads small row
    slices from the shared memory-mapped embeddings. The per-node split is seeded
    from a spawned :class:`~numpy.random.SeedSequence`, so it is reproducible and
    independent of the order nodes happen to run in.

    Whitening (when enabled) and the centred RBF Gram matrices are each computed
    once per model per split and reused across all ordered pairs, rather than
    rebuilt for every pair.
    """
    rows = _node_rows(node.class_indices, rows_by_class)
    if rows.size < n_fit + n_eval:
        return None
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(rows)
    fit_idx, eval_idx = shuffled[:n_fit], shuffled[n_fit : n_fit + n_eval]
    # Materialise the row slices into RAM (np.asarray detaches them from the
    # read-only memmap) once per model, then reuse across whitening, Grams and pairs.
    fit_emb = {name: np.asarray(embeddings[name][fit_idx]) for name in model_names}
    eval_emb = {name: np.asarray(embeddings[name][eval_idx]) for name in model_names}
    if whiten:
        for name in model_names:
            w = fit_whitening(fit_emb[name])
            fit_emb[name], eval_emb[name] = w.apply(fit_emb[name]), w.apply(eval_emb[name])
    gram_fit = {n: centered_rbf_gram(e) for n, e in fit_emb.items()} if use_cka else {}
    gram_eval = {n: centered_rbf_gram(e) for n, e in eval_emb.items()} if use_cka else {}
    records: list[dict] = []
    for src in model_names:
        for tgt in model_names:
            if src == tgt:
                continue
            for kind in transforms:
                if kind == "rbf_cka":
                    r2_train = cka_from_grams(gram_fit[src], gram_fit[tgt])
                    r2_eval = cka_from_grams(gram_eval[src], gram_eval[tgt])
                else:
                    r2_train, r2_eval = evaluate_transform(
                        kind, fit_emb[src], fit_emb[tgt], eval_emb[src], eval_emb[tgt]
                    )
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
                    "source": src,
                    "target_model": tgt,
                    "transform": kind,
                    "whiten": bool(whiten),
                    "r2_train": r2_train,
                    "r2_eval": r2_eval,
                })
    return records


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

    The nodes are embarrassingly parallel (each fits and scores its own pairs
    independently), so they are distributed across CPU workers with joblib
    (``cfg.n_jobs``; ``-1`` uses the whole SLURM allocation). The large embeddings
    are memory-mapped and shared read-only across workers, so fanning out costs no
    extra RAM. See :func:`_evaluate_node` for the per-node work.
    """
    n_fit, n_eval = cfg.n_fit_samples, cfg.n_eval_samples
    whiten = cfg.get("whiten", False)
    use_cka = "rbf_cka" in cfg.transforms
    transforms = list(cfg.transforms)
    model_names = [m.spec.name for m in models]
    all_nodes = [(target, node) for target, nodes in target_nodes.items() for node in nodes]
    rows_by_class = _rows_by_class(sample_classes)
    seeds = np.random.SeedSequence(cfg.seed).spawn(len(all_nodes))
    n_jobs = _resolve_n_jobs(int(cfg.get("n_jobs", -1)))

    log.info("Evaluating %d nodes across %d worker(s) (whiten=%s, transforms=%s).",
             len(all_nodes), n_jobs, whiten, transforms)
    start = time.monotonic()
    per_node = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_node)(
            target, node, model_names, embeddings, rows_by_class,
            n_fit, n_eval, whiten, use_cka, transforms, seed,
        )
        for (target, node), seed in zip(all_nodes, seeds)
    )
    records = [r for node_records in per_node if node_records for r in node_records]
    skipped = sum(node_records is None for node_records in per_node)
    log.info("Evaluated %d nodes (%d skipped below the %d-sample threshold) in %.0fs; %d rows.",
             len(all_nodes) - skipped, skipped, n_fit + n_eval,
             time.monotonic() - start, len(records))
    return records
