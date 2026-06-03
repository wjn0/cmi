# README

Experiments in cross-model representation similarity, alignment, and identifiability.

## Running the experiments

The model set is the `experiment` config group, the similarity metric is the
`similarity` group (`linear` or `rbf_cka`); each lands as its own MLflow run.
Sweep both axes with Hydra multirun (`-m`) to cover the full 2×2:

```bash
uv run python -m repsim.run -m \
  experiment=hierarchically_local_similarity,dinov2_scale_similarity \
  similarity=linear,rbf_cka
```

- `experiment=hierarchically_local_similarity` — cross-model (DINOv2 / SigLIP / MAE)
- `experiment=dinov2_scale_similarity` — within-DINO (the four S/B/L/g scales)
- `similarity=linear` — affine least-squares R²; `similarity=rbf_cka` — RBF CKA
- **linear CKA**: `similarity=linear whiten=true` (whitening makes `linear` == linear CKA)

Compute is heavy — launch via SLURM, not on the head node (see `scripts/*.sbatch`).
Embeddings are cached on first run; re-eval from cache with `scripts/regenerate.py`.

## Viewing results (MLflow)

Every run logs params, summary scores, and `results.csv` + figures to a local
MLflow store. Browse them with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
