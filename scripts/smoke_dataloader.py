"""Smoke-test the multi-worker DataLoader embedding path: no crash + row-aligned.

Embeds a small slice with two DINOv2 scales via extract_embeddings (workers > 0),
then recomputes a few rows serially and checks they match, confirming the
DataLoader preserves order and the fork+CUDA+workers combo is stable.
"""

import numpy as np
import torch
from datasets import load_dataset

from repsim.inference import build_sample_index, extract_embeddings
from repsim.models import LoadedModel, ModelSpec

DATASET = "benjamin-paine/imagenet-1k-256x256"
SPLIT = "train"
specs = [
    ModelSpec("dinov2_s", "transformers", "facebook/dinov2-small", "pooler"),
    ModelSpec("dinov2_b", "transformers", "facebook/dinov2-base", "pooler"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
models = [LoadedModel(s, device) for s in specs]

ds = load_dataset(DATASET)[SPLIT]
labels = np.array(ds["label"])
index = build_sample_index(labels, [0, 1, 2, 3], per_class_limit=64, seed=0)
print(f"rows={len(index.rows)}")

emb = extract_embeddings(
    models, ds, index, batch_size=32,
    cache_dir=__import__("pathlib").Path("/tmp/smoke_emb"),
    dataset_id=DATASET, split="smoke", num_workers=4,
)

# Recompute a handful of rows serially and compare to the DataLoader output.
for name, m in [(s.name, mm) for s, mm in zip(specs, models)]:
    for i in [0, 1, 37, len(index.rows) - 1]:
        img = ds[int(index.rows[i])]["image"].convert("RGB")
        ref = m.embed(m.preprocess(img).unsqueeze(0)).float().cpu().numpy()[0]
        got = emb[name][i]
        ok = np.allclose(ref, got, atol=1e-4)
        print(f"{name} row {i}: match={ok} maxdiff={np.abs(ref - got).max():.2e}")
        assert ok, f"MISALIGNED at {name} row {i}"
print("SMOKE OK: DataLoader path is crash-free and row-aligned.")
