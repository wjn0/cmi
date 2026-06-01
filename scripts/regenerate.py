"""Regenerate results.csv + figures from CACHED embeddings (no GPU, no decode).

The embedding cache is keyed by the sampling config, so once embeddings exist we
can re-run the evaluation/plotting at new transform/n_fit settings without
touching a model or decoding an image. Bypasses model construction entirely.

Usage: srun -c 8 --mem=48G --time=1:00:00 uv run python scripts/regenerate.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf

from repsim.experiment import _build_all_nodes, _evaluate
from repsim.imagenet_hierarchy import all_class_indices
from repsim.inference import _cache_key, build_sample_index
from repsim.plots import plot_all

cfg = OmegaConf.load("conf/hierarchically_local_similarity.yaml")
names = [m.name for m in cfg.models]

print("loading labels...", flush=True)
ds = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
labels = np.array(ds["label"])
index = build_sample_index(labels, all_class_indices(), cfg.per_class_limit, cfg.seed)

print("loading cached embeddings...", flush=True)
cache = Path(cfg.cache_dir)
emb = {
    n: np.load(cache / f"{_cache_key(cfg.dataset.hf_id, cfg.dataset.split, index, n)}.npy")
    for n in names
}
models = [SimpleNamespace(spec=SimpleNamespace(name=n)) for n in names]

print("building nodes + evaluating...", flush=True)
target_nodes = _build_all_nodes(cfg)
records = _evaluate(cfg, target_nodes, models, emb, index.classes)

import pandas as pd  # noqa: E402

out_dir = Path("outputs/regenerated")
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame.from_records(records).to_csv(out_dir / "results.csv", index=False)
print(f"wrote {len(records)} rows to {out_dir / 'results.csv'}", flush=True)
paths = plot_all(out_dir / "results.csv")
print("figures:", [p.name for p in paths], flush=True)
