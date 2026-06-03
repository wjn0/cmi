"""Regenerate results.csv + figures from CACHED embeddings (no GPU, no decode).

The embedding cache is keyed by the sampling config, so once embeddings exist we
can re-run the evaluation/plotting at new transform/n_fit settings without
touching a model or decoding an image. Bypasses model construction entirely.

Usage: srun -c 32 --mem=120G --time=1:00:00 \
           uv run python scripts/regenerate.py [experiment] [overrides...]

  experiment  config group under conf/experiment (default
              hierarchically_local_similarity).
  overrides   extra dotlist config overrides, e.g. transforms=[linear] whiten=true.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf

from repsim.experiment import _build_all_nodes, _evaluate
from repsim.imagenet_hierarchy import all_class_indices
from repsim.inference import _cache_key, build_sample_index
from repsim.plots import plot_all
from repsim.tracking import log_run

# Compose the same config Hydra would: shared base + one experiment's overrides,
# plus any dotlist overrides passed on the command line.
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else "hierarchically_local_similarity"
cfg = OmegaConf.merge(
    OmegaConf.load("conf/config.yaml"),
    OmegaConf.load(f"conf/experiment/{EXPERIMENT}.yaml"),
    OmegaConf.from_dotlist(sys.argv[2:]),
)
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

import pandas as pd  # noqa: E402

out_dir = Path("outputs/regenerated") / EXPERIMENT
out_dir.mkdir(parents=True, exist_ok=True)
results = pd.DataFrame.from_records(records)
results.to_csv(out_dir / "results.csv", index=False)
print(f"wrote {len(records)} rows to {out_dir / 'results.csv'}", flush=True)
paths = plot_all(out_dir / "results.csv")
print("figures:", [p.name for p in paths], flush=True)

log_run(cfg, results, artifacts=[out_dir / "results.csv", *paths],
        run_name=EXPERIMENT, tags={"experiment": EXPERIMENT, "source": "regenerate"})
print("logged run to MLflow", flush=True)
