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
  - `repsim/transforms.py` — Linear (affine least-squares) and rigid (PCA + similarity Procrustes) cross-model transforms with R² scoring, nonlinear RBF CKA (median-bandwidth, double-centred kernel; needs un-whitened inputs), plus optional PCA whitening of embeddings.
  - `repsim/inference.py` — Seeded per-class image sampling and cached embedding extraction (multi-worker DataLoader: decode+preprocess parallelised across CPUs to keep the GPU fed).
  - `repsim/experiment.py` — Orchestrates the `hierarchically_local_similarity` experiment (full-1000-class embed once, per-target descendant→ancestor node chains deduplicated by synset + null nodes, fixed within-node fit/held-out split, in-sample + held-out R²); writes `results.csv`.
  - `repsim/plots.py` — Nature-styled figures from a results CSV (per-model-pair R² vs granularity trendlines with jittered points, real-vs-null locality control, per-model-pair overfitting scatter coloured by breadth, model-pair heatmap).
  - `repsim/log.py` — Flushing stdout logging setup so progress appears live in SLURM log files (bare `print` is block-buffered to files).
  - `repsim/run.py` — Hydra entrypoint (`uv run python -m repsim.run`).
- `conf/config.yaml` — Shared base config + `experiment` config group; sets `transforms: [linear, rbf_cka]` (both similarities in one pass) and selects an experiment. Sweep both via `-m experiment=hierarchically_local_similarity,dinov2_scale_similarity`.
- `conf/experiment/hierarchically_local_similarity.yaml` — Experiment overrides: DINOv2/SigLIP/MAE cross-model, n_fit=3840.
- `conf/experiment/dinov2_scale_similarity.yaml` — Experiment overrides: the four DINOv2 scales (S/B/L/g), single global n_fit=7680.
- `scripts/cache_check.py` — Verify the embedding cache hits for the current config (labels only, no model loading) before a slow CPU run.
- `scripts/smoke_dataloader.py` — Smoke test: the multi-worker DataLoader embedding path is crash-free (fork+CUDA+workers) and row-aligned vs a serial recompute.
- `scripts/bench_embed.py` — Benchmark bf16 embedding throughput for the four DINOv2 scales on a GPU (run on gpu100/H100, not debug_gpu).
- `scripts/regenerate.py` — Regenerate results.csv + figures from cached embeddings (no GPU/decode); composes base + one experiment config. Edit `EXPERIMENT` to switch experiments.
- `scripts/run_whiten.sbatch` — SLURM CPU batch script: linear-CKA re-eval of `hierarchically_local_similarity` (`transforms=[linear] whiten=true device=cpu`) from cached embeddings.
- `scripts/run_dinov2_scales.sbatch` — SLURM GPU batch script (L40S — ~4.3x cheaper than H100 for this bf16 embed): embed all four DINOv2 scales over the full pool, then run the `dinov2_scale_similarity` experiment end-to-end (linear + RBF CKA).
- `scripts/run_dinov2_scales_eval_cpu.sbatch` — SLURM CPU batch script: eval-only re-run of `dinov2_scale_similarity` (device=cpu) reusing the cached embeddings; the per-node lstsq + RBF-CKA eval is CPU-bound, so it runs off the GPU (computes both similarities).
- `tests/test_experiment.py` — Tests for per-node fixed fit/held-out splitting in the experiment.
- `tests/test_transforms.py` — Tests for transforms and R² scoring.
- `tests/test_imagenet_hierarchy.py` — Tests for WordNet/ImageNet hierarchy navigation and random control nodes.
- `tests/test_plots.py` — Smoke test that `plot_all` writes every figure.
