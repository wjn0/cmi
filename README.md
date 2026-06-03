# README

Experiments in cross-model representation similarity, alignment, and identifiability.

## Running the experiments

Each run computes **both** similarities in one pass — affine `linear` R² and RBF
`rbf_cka` — over the two model sets:

```bash
# cross-model: DINOv2 / SigLIP / MAE
uv run python -m repsim.run experiment=hierarchically_local_similarity

# within-DINO: the four DINOv2 scales (S/B/L/g)
uv run python -m repsim.run experiment=dinov2_scale_similarity
```

For **linear CKA** (whitened) instead of RBF CKA, add `transforms=[linear] whiten=true`
to either command. This gives the full 2×2 (linear / CKA × cross-model / within-DINO).

Compute is heavy — launch via SLURM, not on the head node (see `scripts/*.sbatch`).
Embeddings are cached on first run; re-eval from cache with `scripts/regenerate.py`.

## Viewing results (MLflow)

Every run logs params, summary scores, and `results.csv` + figures to a local
MLflow store. Browse them with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
