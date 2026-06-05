# README

Experiments in cross-model representation similarity, alignment, and identifiability.

## Experiments

Each experiment is a module in `repsim/experiments/` with matching configs in
`conf/experiment/`; configurations of the same experiment are files inside its
config directory:

- `granularity_similarity` — how representational similarity varies with the
  granularity of the ImageNet/WordNet hierarchy region it is measured over.
  - `experiment=granularity_similarity/cross_model` — DINOv2 / SigLIP / MAE
  - `experiment=granularity_similarity/dinov2_scale` — the four DINOv2 S/B/L/g scales
- `local_similarity_performance` — on a narrow data region (PCam, CelebA), does
  the model that best one-way (asymmetric) linearly predicts the others also
  classify that region best?
  - `experiment=local_similarity_performance`

The similarity metric is the `similarity` group (`linear` or `rbf_cka`); each
(experiment, similarity) pair lands as its own MLflow run. Sweep both axes with
Hydra multirun (`-m`):

```bash
uv run python -m repsim.run -m \
  experiment=granularity_similarity/cross_model,granularity_similarity/dinov2_scale \
  similarity=linear,rbf_cka
```

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
