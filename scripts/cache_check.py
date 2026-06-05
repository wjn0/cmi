"""Verify the embedding cache hits for the current config (no model loading).

Loads only the dataset labels, rebuilds the seeded sample index, and checks
whether each model's cached ``.npy`` exists -- so a CPU run won't silently fall
into the slow full-pool embedding path on a cache miss.

Usage: uv run python scripts/cache_check.py [experiment]
  experiment  config group option (default granularity_similarity/cross_model).
"""

import logging
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from hydra import compose, initialize

from repsim.imagenet_hierarchy import all_class_indices
from repsim.inference import _cache_key, build_sample_index
from repsim.log import setup_logging

setup_logging()
log = logging.getLogger("cache_check")
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "granularity_similarity/cross_model"
with initialize(version_base=None, config_path="../conf"):
    cfg = compose("config", overrides=[f"experiment={EXPERIMENT}"])
log.info("Loading labels for %s split=%s ...", cfg.dataset.hf_id, cfg.dataset.split)
ds = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
labels = np.array(ds["label"])
index = build_sample_index(labels, all_class_indices(), cfg.per_class_limit, cfg.seed)
log.info("Sample index: %d rows.", index.rows.size)

cache_dir = Path(cfg.cache_dir)
all_hit = True
for name in (m.name for m in cfg.models):
    path = cache_dir / f"{_cache_key(cfg.dataset.hf_id, cfg.dataset.split, index, name)}.npy"
    hit = path.exists()
    all_hit &= hit
    log.info("%-9s -> %s  [%s]", name, path.name, "HIT" if hit else "MISS")
log.info("ALL CACHED: %s", all_hit)
