"""Benchmark bf16 embedding throughput for all four DINOv2 scales.

Times extract_embeddings over a few thousand images and reports images/sec, to
confirm the bf16 + larger-batch path is markedly faster than the fp32 baseline
(~50 img/s at batch 64) before committing a full-pool run.
"""

import time

import numpy as np
import torch
from datasets import load_dataset

from repsim.inference import build_sample_index, extract_embeddings
from repsim.models import LoadedModel, ModelSpec

DATASET = "benjamin-paine/imagenet-1k-256x256"
specs = [
    ModelSpec("dinov2_s", "transformers", "facebook/dinov2-small", "pooler"),
    ModelSpec("dinov2_b", "transformers", "facebook/dinov2-base", "pooler"),
    ModelSpec("dinov2_l", "transformers", "facebook/dinov2-large", "pooler"),
    ModelSpec("dinov2_g", "transformers", "facebook/dinov2-giant", "pooler"),
]
device = "cuda" if torch.cuda.is_available() else "cpu"
models = [LoadedModel(s, device) for s in specs]
print(f"device={device} dtypes={[str(m._dtype) for m in models]}")

ds = load_dataset(DATASET)["train"]
labels = np.array(ds["label"])
index = build_sample_index(labels, list(range(8)), per_class_limit=400, seed=0)
n = len(index.rows)
print(f"benchmarking {n} images x {len(models)} models, batch=256, workers=16")

t0 = time.monotonic()
extract_embeddings(
    models, ds, index, batch_size=256,
    cache_dir=__import__("pathlib").Path("/tmp/bench_emb"),
    dataset_id=DATASET, split="bench", num_workers=16,
)
dt = time.monotonic() - t0
print(f"elapsed={dt:.1f}s  throughput={n / dt:.0f} img/s")
print(f"=> full pool (997,939 img) ETA ~ {997939 / (n / dt) / 60:.0f} min")
