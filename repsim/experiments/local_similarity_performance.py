"""The ``local_similarity_performance`` experiment.

Hypothesis: on a dataset occupying a *small region* of the training manifold, the
model that is the best one-way (asymmetric) linear *predictor* of the others on
that region also classifies it best. Concretely, for models A, B, C and a region
D, let the out-predictivity of A be the mean R^2 of linearly mapping A's features
onto each other model's features on D:

    pred_out(A; D) = mean_{M != A} R^2(A -> M; D).

The claim is that ``argmax_M pred_out(M; D)`` coincides with ``argmax_M acc(M, D)``
-- the model the others are most expressible from is the one that has captured the
region best. Two narrow regions are used: PCam (histopathology patches, tumour
vs. normal) and CelebA (face attribute, e.g. smiling), each a small, specialised
slice far from the broad ImageNet pool the models were pre/self-trained on.

For each dataset D and model M_i: (1) embed D's train+test splits with M_i's
native pooled features (cached, shared seed), (2) fit a linear probe on the train
features and record held-out test accuracy ``acc(M_i, D)``, (3) score every
*ordered* model pair (M_i -> M_j) on D under the selected ``similarity`` metric
(asymmetric linear R^2, or symmetric RBF CKA) -- fit on train, scored on the same
held-out test split as the probe. Results are written long-format, one row per
(dataset, ordered pair, transform), carrying the pair's similarity and each
member's probe accuracy; per-model out-predictivity is derived in the plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from repsim.inference import Embeddings, SampleIndex, build_sample_index, extract_embeddings
from repsim.models import LoadedModel, ModelSpec
from repsim.transforms import (
    centered_rbf_gram,
    cka_from_grams,
    evaluate_transform,
    fit_whitening,
)

log = logging.getLogger(__name__)


def _split_labels(dataset, label_column: str) -> Int[np.ndarray, " total"]:
    """Return the integer class label of every row in a split.

    The PCam (``label``) and CelebA-attribute columns are boolean; both are cast
    to ``{0, 1}`` so the same balanced per-class sampling works for either.
    """
    return np.asarray(dataset[label_column], dtype=np.int64)


def _embed_split(
    models: list[LoadedModel],
    dataset_cfg: DictConfig,
    split: str,
    per_class_limit: int,
    cfg: DictConfig,
    cache_dir: Path,
) -> tuple[Embeddings, SampleIndex]:
    """Embed one dataset split with every model on a shared seeded sample.

    A single balanced per-class sample (``per_class_limit`` rows per class, shared
    seed) is drawn once and embedded by every model, so all models see the same
    images in the same order; the result is cached per (dataset, split, sample).

    Returns:
        The per-model embeddings and the :class:`SampleIndex` whose ``classes`` are
        the probe labels aligned with the embedding rows.
    """
    config = dataset_cfg.get("config")
    dataset = load_dataset(dataset_cfg.hf_id, config)[split] if config else load_dataset(dataset_cfg.hf_id)[split]
    labels = _split_labels(dataset, dataset_cfg.label_column)
    classes = sorted(int(c) for c in np.unique(labels))
    index = build_sample_index(labels, classes, per_class_limit, cfg.seed)
    embeddings = extract_embeddings(
        models, dataset, index, cfg.batch_size,
        cache_dir, dataset_cfg.hf_id, split,
        num_workers=cfg.get("num_workers", 8),
    )
    return embeddings, index


def _probe_accuracy(
    train_emb: Float[np.ndarray, "n_train d"],
    train_labels: Int[np.ndarray, " n_train"],
    test_emb: Float[np.ndarray, "n_test d"],
    test_labels: Int[np.ndarray, " n_test"],
    cfg: DictConfig,
) -> float:
    """Fit a standardised logistic-regression linear probe and return test accuracy.

    Features are standardised (the probe is fit on frozen embeddings, whose scale
    differs across models) and an L2-regularised logistic regression is fit on the
    train split, then scored on the held-out test split shared with the similarity
    transforms.
    """
    probe_cfg = cfg.probe
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=probe_cfg.get("C", 1.0), max_iter=probe_cfg.get("max_iter", 1000)),
    )
    clf.fit(train_emb, train_labels)
    return float(clf.score(test_emb, test_labels))


def _pairwise_similarity(
    train_emb: dict[str, np.ndarray],
    test_emb: dict[str, np.ndarray],
    transforms: list[str],
    whiten: bool,
) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Score every ordered model pair, fitting on train and evaluating on test.

    Linear R^2 is asymmetric (``A -> B`` differs from ``B -> A``); RBF CKA is
    symmetric but still recorded per ordered pair for a uniform schema. Whitening
    (when enabled) and the centred RBF Grams are computed once per model and reused
    across all pairs.

    Returns:
        ``{(source, target): {transform: (r2_train, r2_eval)}}``.
    """
    model_names = list(train_emb)
    fit_emb = {n: np.asarray(train_emb[n]) for n in model_names}
    eval_emb = {n: np.asarray(test_emb[n]) for n in model_names}
    if whiten:
        for name in model_names:
            w = fit_whitening(fit_emb[name])
            fit_emb[name], eval_emb[name] = w.apply(fit_emb[name]), w.apply(eval_emb[name])
    use_cka = "rbf_cka" in transforms
    gram_fit = {n: centered_rbf_gram(e) for n, e in fit_emb.items()} if use_cka else {}
    gram_eval = {n: centered_rbf_gram(e) for n, e in eval_emb.items()} if use_cka else {}

    scores: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for src in model_names:
        for tgt in model_names:
            if src == tgt:
                continue
            per_transform: dict[str, tuple[float, float]] = {}
            for kind in transforms:
                if kind == "rbf_cka":
                    per_transform[kind] = (
                        cka_from_grams(gram_fit[src], gram_fit[tgt]),
                        cka_from_grams(gram_eval[src], gram_eval[tgt]),
                    )
                else:
                    per_transform[kind] = evaluate_transform(
                        kind, fit_emb[src], fit_emb[tgt], eval_emb[src], eval_emb[tgt]
                    )
            scores[(src, tgt)] = per_transform
    return scores


def run_experiment(cfg: DictConfig, output_dir: Path | None = None) -> pd.DataFrame:
    """Run the local-similarity-vs-performance experiment and persist results.

    Args:
        cfg: Hydra configuration (``conf/config.yaml`` +
            ``conf/experiment/local_similarity_performance.yaml`` + a ``similarity``
            group).
        output_dir: Directory to write ``results.csv`` to (defaults to cwd).

    Returns:
        A long-format DataFrame with one row per (dataset, ordered model pair,
        transform) carrying the pair's similarity (``r2_train``/``r2_eval``) and
        each member's held-out probe accuracy (``acc_source``/``acc_target``).
    """
    cache_dir = Path(cfg.cache_dir)
    models = [LoadedModel(ModelSpec(**OmegaConf.to_container(m)), cfg.device) for m in cfg.models]
    max_dim = max(m.dim for m in models)
    transforms = list(cfg.transforms)
    whiten = bool(cfg.get("whiten", False))
    if whiten and "rbf_cka" in transforms:
        raise ValueError("whiten=true is incompatible with rbf_cka (it double-centres its kernel).")

    records: list[dict] = []
    for dataset_cfg in cfg.datasets:
        log.info("Dataset %s (%s) ...", dataset_cfg.name, dataset_cfg.hf_id)
        train_emb, train_idx = _embed_split(
            models, dataset_cfg, dataset_cfg.train_split, cfg.n_train_per_class, cfg, cache_dir
        )
        test_emb, test_idx = _embed_split(
            models, dataset_cfg, dataset_cfg.test_split, cfg.n_test_per_class, cfg, cache_dir
        )
        n_train = len(train_idx.rows)
        if n_train <= max_dim:
            raise ValueError(
                f"{dataset_cfg.name}: {n_train} train samples <= largest model dim "
                f"({max_dim}); the linear transform is under-determined. Raise "
                f"n_train_per_class."
            )
        accuracy = {
            m.spec.name: _probe_accuracy(
                np.asarray(train_emb[m.spec.name]), train_idx.classes,
                np.asarray(test_emb[m.spec.name]), test_idx.classes, cfg,
            )
            for m in models
        }
        log.info("  probe test accuracy: %s",
                 ", ".join(f"{n}={a:.3f}" for n, a in accuracy.items()))
        similarity = _pairwise_similarity(train_emb, test_emb, transforms, whiten)
        for (src, tgt), per_transform in similarity.items():
            for kind, (r2_train, r2_eval) in per_transform.items():
                records.append({
                    "dataset": dataset_cfg.name,
                    "hf_id": dataset_cfg.hf_id,
                    "n_train": n_train,
                    "n_test": len(test_idx.rows),
                    "source": src,
                    "target_model": tgt,
                    "transform": kind,
                    "whiten": whiten,
                    "r2_train": r2_train,
                    "r2_eval": r2_eval,
                    "acc_source": accuracy[src],
                    "acc_target": accuracy[tgt],
                })

    results = pd.DataFrame.from_records(records)
    out_dir = output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    log.info("Wrote %d rows to %s", len(results), out_dir / "results.csv")
    return results


def plot_results(results_csv: Path) -> list[Path]:
    """Generate this experiment's figures next to ``results_csv``."""
    from repsim.plots import plot_local_similarity_performance

    return plot_local_similarity_performance(results_csv)
