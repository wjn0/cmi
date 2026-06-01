# Project Overview

Experiments in cross-model representation similarity, alignment, and identifiability.

## Files

- `pyproject.toml` — Project metadata and dependencies (uv-managed).
- `uv.lock` — Pinned dependency lockfile.
- `README.md` — Short project description.
- `CLAUDE.md` — Development guidelines for this project.
- `OVERVIEW.md` — This file; one-line description of every file in the project.
- `repsim/` — Main package.
  - `repsim/__init__.py` — Package root (currently empty).
  - `repsim/imagenet_hierarchy.py` — WordNet navigation of the ImageNet-1k label tree (class↔synset mapping, build a target's descendant→self→ancestor node chain, and size-matched random-class null nodes).
  - `repsim/models.py` — Load pretrained vision models (transformers/timm) and extract native pooled embeddings.
  - `repsim/transforms.py` — Linear (affine least-squares) and rigid (PCA + similarity Procrustes) cross-model transforms with R² scoring, plus optional PCA whitening of embeddings.
  - `repsim/inference.py` — Seeded per-class image sampling and cached embedding extraction.
  - `repsim/experiment.py` — Orchestrates the `hierarchically_local_similarity` experiment (full-1000-class embed once, per-target descendant→ancestor node chains + null nodes, fixed within-node fit/held-out split, in-sample + held-out R²); writes `results.csv`.
  - `repsim/plots.py` — Nature-styled figures from a results CSV (per-model-pair R² vs granularity trendlines with jittered points, real-vs-null locality control, per-model-pair overfitting scatter coloured by breadth, model-pair heatmap).
  - `repsim/log.py` — Flushing stdout logging setup so progress appears live in SLURM log files (bare `print` is block-buffered to files).
  - `repsim/run.py` — Hydra entrypoint (`uv run python -m repsim.run`).
- `conf/hierarchically_local_similarity.yaml` — Hydra config for the experiment.
- `scripts/cache_check.py` — Verify the embedding cache hits for the current config (labels only, no model loading) before a slow CPU run.
- `scripts/run_whiten.sbatch` — SLURM CPU batch script: re-eval the experiment with `whiten=true device=cpu` from cached embeddings.
- `tests/test_experiment.py` — Tests for per-node fixed fit/held-out splitting in the experiment.
- `tests/test_transforms.py` — Tests for transforms and R² scoring.
- `tests/test_imagenet_hierarchy.py` — Tests for WordNet/ImageNet hierarchy navigation and random control nodes.
- `tests/test_plots.py` — Smoke test that `plot_all` writes every figure.
