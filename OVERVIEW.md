# Project Overview

Experiments in cross-model representation similarity, alignment, and identifiability.

## Files

- `pyproject.toml` — Project metadata and dependencies (uv-managed).
- `uv.lock` — Pinned dependency lockfile.
- `README.md` — Project description + how to run the experiments (the `experiment` × `similarity` Hydra axes) and launch the MLflow UI.
- `CLAUDE.md` — Development guidelines for this project.
- `OVERVIEW.md` — This file; one-line description of every file in the project.
- `repsim/` — Main package.
  - `repsim/__init__.py` — Package root (currently empty).
  - `repsim/experiments/` — One module per experiment, dispatched by the config's `experiment_name`.
    - `repsim/experiments/__init__.py` — Experiment registry (`get_experiment`); each module exposes `run_experiment(cfg, output_dir)` and `plot_results(results_csv)`.
    - `repsim/experiments/granularity_similarity.py` — Orchestrates the `granularity_similarity` experiment (full-1000-class embed once, per-target descendant→ancestor node chains deduplicated by synset + null nodes, fixed within-node fit/held-out split, in-sample + held-out R²); the embarrassingly-parallel per-node eval fans out across CPU workers (joblib, `n_jobs`, SLURM-aware) with a precomputed class→rows index; writes `results.csv`.
    - `repsim/experiments/local_similarity_performance.py` — Tests whether, on a narrow data region (PCam, CelebA), the model that best one-way (asymmetric) linearly predicts the others (highest out-predictivity) also classifies it best: embeds each dataset's train/test per model (cached, balanced per-class), fits a standardised logistic linear probe for held-out accuracy, scores every ordered model pair under the selected `similarity` metric (fit on train, R²/CKA on the same held-out test split); writes long-format `results.csv` (per-model out-predictivity derived in the plots).
  - `repsim/imagenet_hierarchy.py` — WordNet navigation of the ImageNet-1k label tree (class↔synset mapping, build a target's descendant→self→ancestor node chain, and size-matched random-class null nodes).
  - `repsim/models.py` — Load pretrained vision models (transformers/timm) and extract native pooled embeddings.
  - `repsim/transforms.py` — Linear (affine least-squares) and rigid (PCA + similarity Procrustes) cross-model transforms with R² scoring, nonlinear RBF CKA (median-bandwidth, double-centred kernel; needs un-whitened inputs), plus optional PCA whitening of embeddings.
  - `repsim/inference.py` — Seeded per-class image sampling and cached embedding extraction (multi-worker DataLoader: decode+preprocess parallelised across CPUs to keep the GPU fed); returns memory-mapped embedding matrices so the per-node eval reads small row slices without loading ~15 GB into RAM.
  - `repsim/plots.py` — Nature-styled figures. granularity_similarity (`plot_all`): per-model-pair R² vs granularity trendlines with jittered points, real-vs-null locality control, per-model-pair overfitting scatter coloured by breadth, model-pair heatmap. local_similarity_performance (`plot_local_similarity_performance`): per-dataset out-predictivity-vs-accuracy scatter (agreement marker) and grouped out-predictivity/accuracy bars.
  - `repsim/tracking.py` — Optional MLflow logging of each run (config params, summary held-out/in-sample scores per transform, and `results.csv`/figures as artifacts); tolerant of per-experiment config keys; no-op if disabled or MLflow unavailable.
  - `repsim/log.py` — Flushing stdout logging setup so progress appears live in SLURM log files (bare `print` is block-buffered to files).
  - `repsim/run.py` — Hydra entrypoint (`uv run python -m repsim.run`); dispatches to the experiment module named by `cfg.experiment_name`.
- `conf/config.yaml` — Shared base config (seed, device, cache_dir, n_jobs, mlflow) + the `experiment` and `similarity` config groups; only knobs shared by every experiment live here.
- `conf/similarity/linear.yaml` — Similarity-metric group: affine least-squares linear R² (`transforms=[linear]`, `whiten=false`); its own MLflow run.
- `conf/similarity/rbf_cka.yaml` — Similarity-metric group: nonlinear RBF CKA (`transforms=[rbf_cka]`, `whiten=false`); its own MLflow run.
- `conf/experiment/granularity_similarity/_shared.yaml` — The granularity_similarity design shared by both configurations (targets, hierarchy chains, null controls, per-class sampling, ImageNet dataset, `experiment_name`).
- `conf/experiment/granularity_similarity/cross_model.yaml` — Configuration: DINOv2/SigLIP/MAE cross-model, n_fit=3840; full procedure description.
- `conf/experiment/granularity_similarity/dinov2_scale.yaml` — Configuration: the four DINOv2 scales (S/B/L/g), single global n_fit=7680.
- `conf/experiment/local_similarity_performance.yaml` — Experiment config: same DINOv2/SigLIP/MAE trio; PCam + CelebA datasets (binary tasks); standardised logistic linear-probe block; balanced per-class train/test sampling.
- `scripts/cache_check.py` — Verify the embedding cache hits for a composed experiment config (labels only, no model loading) before a slow CPU run.
- `scripts/smoke_dataloader.py` — Smoke test: the multi-worker DataLoader embedding path is crash-free (fork+CUDA+workers) and row-aligned vs a serial recompute.
- `scripts/bench_embed.py` — Benchmark bf16 embedding throughput for the four DINOv2 scales on a GPU (run on gpu100/H100, not debug_gpu).
- `scripts/regenerate.py` — Regenerate granularity_similarity results.csv + figures from cached embeddings (no GPU/decode); composes the config with Hydra (`scripts/regenerate.py [experiment] [overrides...]`).
- `scripts/run_whiten.sbatch` — SLURM CPU batch script: linear-CKA re-eval of `granularity_similarity/cross_model` (`transforms=[linear] whiten=true device=cpu`) from cached embeddings.
- `scripts/run_dinov2_scales.sbatch` — SLURM GPU batch script (L40S — ~4.3x cheaper than H100 for this bf16 embed): embed all four DINOv2 scales over the full pool, then run `granularity_similarity/dinov2_scale` end-to-end (linear + RBF CKA).
- `scripts/run_dinov2_scales_eval_cpu.sbatch` — SLURM CPU batch script: eval-only re-run of `granularity_similarity/dinov2_scale` (device=cpu) reusing the cached embeddings; the per-node lstsq + RBF-CKA eval is CPU-bound, so it runs off the GPU (computes both similarities).
- `scripts/run_both_eval_cpu.sbatch` — SLURM CPU batch script: eval-only sweep of both granularity_similarity configurations × both similarities from cached embeddings.
- `scripts/run_local_similarity.sbatch` — SLURM GPU batch script (L40S): embed PCam + CelebA train/test with the DINOv2/SigLIP/MAE trio, then run `local_similarity_performance` over both similarities (probe accuracy + ordered-pair similarity).
- `tests/test_granularity_similarity.py` — Tests for per-node fixed fit/held-out splitting in the granularity_similarity experiment.
- `tests/test_experiments_registry.py` — Tests that every experiment config composes, dispatches to a registered module, and that the local_similarity_performance config carries its dataset list and probe block.
- `tests/test_local_similarity_performance.py` — Tests the experiment's pure stages: separable-task probe accuracy, asymmetric linear pairwise similarity, and the out-predictivity aggregation.
- `tests/test_transforms.py` — Tests for transforms and R² scoring.
- `tests/test_imagenet_hierarchy.py` — Tests for WordNet/ImageNet hierarchy navigation and random control nodes.
- `tests/test_plots.py` — Smoke test that `plot_all` writes every figure.
