"""Sampling ImageNet images per class and extracting/caching model embeddings.

A single seeded sample of row indices is drawn per split and shared across all
models, so every model embeds exactly the same images in the same order. Each
model's embeddings for a split are cached to disk keyed by the sampling
configuration, making re-runs cheap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from jaxtyping import Float, Int
from tqdm import tqdm

from repsim.models import LoadedModel

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


def extract_embeddings(
    models: Sequence[LoadedModel],
    dataset: Dataset,
    index: SampleIndex,
    batch_size: int,
    cache_dir: Path,
    dataset_id: str,
    split: str,
) -> Embeddings:
    """Extract (or load cached) pooled embeddings for several models at once.

    Each image is decoded only once and fed to every model (decoding, not the
    forward pass, is the bottleneck). Per-model embeddings are aligned with
    ``index.rows`` and cached under ``cache_dir``; if every model is already
    cached, nothing is decoded.

    Returns:
        A mapping from model name to its embedding matrix.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        m.spec.name: cache_dir / f"{_cache_key(dataset_id, split, index, m.spec.name)}.npy"
        for m in models
    }
    if all(p.exists() for p in paths.values()):
        return {name: np.load(p) for name, p in paths.items()}

    rows = index.rows.tolist()
    chunks: dict[str, list[np.ndarray]] = {m.spec.name: [] for m in models}
    for start in tqdm(range(0, len(rows), batch_size), desc=f"embed/{split}"):
        images = [img.convert("RGB") for img in dataset[rows[start : start + batch_size]]["image"]]
        for model in models:
            tensors = torch.stack([model.preprocess(img) for img in images])
            chunks[model.spec.name].append(model.embed(tensors).float().cpu().numpy())

    embeddings = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    for name, matrix in embeddings.items():
        np.save(paths[name], matrix)
    return embeddings
