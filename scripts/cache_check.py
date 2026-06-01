"""Verify the embedding cache hits for the current config (no model loading).

Loads only the dataset labels, rebuilds the seeded sample index, and checks
whether each model's cached ``.npy`` exists -- so a CPU run won't silently fall
into the slow full-pool embedding path on a cache miss.
"""

import logging
from pathlib import Path

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf

from repsim.imagenet_hierarchy import all_class_indices
from repsim.inference import _cache_key, build_sample_index
from repsim.log import setup_logging

setup_logging()
log = logging.getLogger("cache_check")
cfg = OmegaConf.load("conf/hierarchically_local_similarity.yaml")
per_class_limit = cfg.per_class_limit or 768
log.info("Loading labels for %s split=%s ...", cfg.dataset.hf_id, cfg.dataset.split)
ds = load_dataset(cfg.dataset.hf_id)[cfg.dataset.split]
labels = np.array(ds["label"])
index = build_sample_index(labels, all_class_indices(), per_class_limit, cfg.seed)
log.info("Sample index: %d rows.", index.rows.size)

cache_dir = Path(cfg.cache_dir)
all_hit = True
for name in ("dinov2", "siglip", "mae"):
    path = cache_dir / f"{_cache_key(cfg.dataset.hf_id, cfg.dataset.split, index, name)}.npy"
    hit = path.exists()
    all_hit &= hit
    log.info("%-7s -> %s  [%s]", name, path.name, "HIT" if hit else "MISS")
log.info("ALL CACHED: %s", all_hit)
