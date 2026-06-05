"""Regenerate results.csv + figures from CACHED embeddings (no GPU, no decode).

The embedding cache is keyed by the sampling config, so once embeddings exist we
can re-run the evaluation/plotting at new transform/n_fit settings without
touching a model or decoding an image. Bypasses model construction entirely.
Specific to the granularity_similarity experiment (it rebuilds its hierarchy
nodes and per-node eval).

Usage: srun -c 32 --mem=120G --time=1:00:00 \
           uv run python scripts/regenerate.py [experiment] [overrides...]

  experiment  config group option under conf/experiment (default
              granularity_similarity/cross_model).
  overrides   extra Hydra overrides, e.g. similarity=rbf_cka whiten=true.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from datasets import load_dataset
from hydra import compose, initialize
from omegaconf import OmegaConf

from repsim.experiments.granularity_similarity import _build_all_nodes, _evaluate, plot_results
from repsim.imagenet_hierarchy import all_class_indices
from repsim.inference import _cache_key, build_sample_index
from repsim.tracking import log_run

# Compose the same config Hydra would: shared base + the experiment's defaults
# chain, plus any overrides passed on the command line.
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "granularity_similarity/cross_model"
with initialize(version_base=None, config_path="../conf"):
    cfg = compose("config", overrides=[f"experiment={EXPERIMENT}", *sys.argv[2:]])
names = [m.name for m in cfg.models]

print("loading labels...", flush=True)
ds = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
labels = np.array(ds["label"])
index = build_sample_index(labels, all_class_indices(), cfg.per_class_limit, cfg.seed)

print("loading cached embeddings (memory-mapped)...", flush=True)
cache = Path(cfg.cache_dir)
emb = {
    n: np.load(
        cache / f"{_cache_key(cfg.dataset.hf_id, cfg.dataset.split, index, n)}.npy",
        mmap_mode="r",
    )
    for n in names
}
models = [SimpleNamespace(spec=SimpleNamespace(name=n)) for n in names]

print("building nodes + evaluating...", flush=True)
target_nodes = _build_all_nodes(cfg)
records = _evaluate(cfg, target_nodes, models, emb, index.classes)

out_dir = Path("outputs/regenerated") / EXPERIMENT
out_dir.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame.from_records(records)
results.to_csv(out_dir / "results.csv", index=False)
print(f"wrote {len(records)} rows to {out_dir / 'results.csv'}", flush=True)
paths = plot_results(out_dir / "results.csv")
print("figures:", [p.name for p in paths], flush=True)

log_run(cfg, results, artifacts=[out_dir / "results.csv", *paths],
        experiment=EXPERIMENT, extra_tags={"source": "regenerate"})
print("logged run to MLflow", flush=True)
