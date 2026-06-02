"""Sampling ImageNet images per class and extracting/caching model embeddings.

A single seeded sample of row indices is drawn per split and shared across all
models, so every model embeds exactly the same images in the same order. Each
model's embeddings for a split are cached to disk keyed by the sampling
configuration, making re-runs cheap.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from repsim.models import LoadedModel

log = logging.getLogger(__name__)

Embeddings = dict[str, Float[np.ndarray, "n d"]]


@dataclass(frozen=True)
class SampleIndex:
    """A fixed set of dataset rows and their class indices for one split.

    Attributes:
        rows: Dataset row indices, grouped by class in ``classes`` order.
        classes: Class index of each row (aligned with ``rows``).
    """

    rows: Int[np.ndarray, "n"]
    classes: Int[np.ndarray, "n"]


def build_sample_index(
    labels: Int[np.ndarray, "total"],
    class_indices: Sequence[int],
    per_class_limit: int,
    seed: int,
) -> SampleIndex:
    """Sample up to ``per_class_limit`` rows for each requested class.

    Args:
        labels: Class label of every row in the split.
        class_indices: Classes to sample from.
        per_class_limit: Maximum rows to draw per class.
        seed: Seed for reproducible sampling.

    Returns:
        A :class:`SampleIndex` over the sampled rows.
    """
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    classes: list[int] = []
    for cls in class_indices:
        candidates = np.flatnonzero(labels == cls)
        if candidates.size > per_class_limit:
            candidates = rng.choice(candidates, per_class_limit, replace=False)
        rows.extend(int(r) for r in candidates)
        classes.extend([cls] * candidates.size)
    return SampleIndex(rows=np.array(rows, dtype=np.int64), classes=np.array(classes, dtype=np.int64))


def _cache_key(dataset_id: str, split: str, index: SampleIndex, model_name: str) -> str:
    """Return a deterministic cache filename stem for one model/split."""
    payload = json.dumps(
        {
            "dataset": dataset_id,
            "split": split,
            "model": model_name,
            "rows": index.rows.tolist(),
        }
    ).encode()
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{model_name}_{split}_{digest}"


class _PreprocessDataset(TorchDataset):
    """Decode one image and apply every model's preprocessing, in DataLoader workers.

    Decoding a JPEG and running the per-model resize/normalize is pure-CPU work
    that otherwise starves the GPU; a multi-worker ``DataLoader`` over this
    dataset parallelises it across cores so the GPU stays fed. ``__getitem__``
    returns one preprocessed tensor per model (in ``models`` order); the default
    collate stacks each position into a per-model batch. Workers only touch the
    models' CPU preprocessors, never their GPU weights.
    """

    def __init__(self, dataset: Dataset, rows: Sequence[int], models: Sequence[LoadedModel]) -> None:
        self._dataset = dataset
        self._rows = rows
        self._models = models

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> list[Float[torch.Tensor, "c h w"]]:
        image = self._dataset[int(self._rows[i])]["image"].convert("RGB")
        return [m.preprocess(image) for m in self._models]


def extract_embeddings(
    models: Sequence[LoadedModel],
    dataset: Dataset,
    index: SampleIndex,
    batch_size: int,
    cache_dir: Path,
    dataset_id: str,
    split: str,
    num_workers: int = 8,
) -> Embeddings:
    """Extract (or load cached) pooled embeddings for several models at once.

    Each image is decoded once and preprocessed for every model inside a
    multi-worker :class:`~torch.utils.data.DataLoader` (decode + preprocess, not
    the forward pass, is the bottleneck, so it is parallelised across
    ``num_workers`` CPU workers to keep the GPU fed). Per-model embeddings are
    aligned with ``index.rows`` and cached under ``cache_dir``; if every model is
    already cached, nothing is decoded.

    Args:
        num_workers: DataLoader worker processes for parallel decode/preprocess.

    Returns:
        A mapping from model name to its embedding matrix.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        m.spec.name: cache_dir / f"{_cache_key(dataset_id, split, index, m.spec.name)}.npy"
        for m in models
    }
    if all(p.exists() for p in paths.values()):
        log.info("Loading cached embeddings for %d models from %s.", len(paths), cache_dir)
        return {name: np.load(p) for name, p in paths.items()}

    log.info(
        "Cache miss: embedding %d images for %d models (%s) with %d workers. This is the slow path.",
        len(index.rows), len(models),
        ", ".join(p.name for p in paths.values() if not p.exists()), num_workers,
    )
    loader = DataLoader(
        _PreprocessDataset(dataset, index.rows, models),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    chunks: dict[str, list[np.ndarray]] = {m.spec.name: [] for m in models}
    # mininterval=10: in a file (no TTY) tqdm writes one line per update, so cap
    # updates to every ~10s rather than flooding the SLURM log with thousands.
    for batch in tqdm(loader, desc=f"embed/{split}", mininterval=10.0):
        for model, tensors in zip(models, batch):
            chunks[model.spec.name].append(model.embed(tensors).float().cpu().numpy())

    embeddings = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    for name, matrix in embeddings.items():
        np.save(paths[name], matrix)
    log.info("Embedded and cached %d images for %d models.", len(index.rows), len(models))
    return embeddings
