#!/bin/bash
#SBATCH --job-name=repsim_dinov2_scales
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_logs/dinov2_scales_%j.log

# Cost note: the H100 (gpu100) is only ~1.9x faster than an L40S for this bf16
# embed but costs 8x, so the L40S is ~4.3x cheaper per run -- we pin the L40S
# (Ada, full bf16 tensor cores). MUST request a modern card explicitly: the gpu
# partition is mostly 1080ti/2080ti, where bf16 is emulated and ~100x slower.
#
# Embed the full 1000-class pool once for each of the four DINOv2 scales
# (small/base/large/giant CLS poolers), cache them, then run the
# dinov2_scale_similarity experiment (per-node fixed fit/held-out splits, both the
# linear transform and RBF CKA between every ordered scale pair) and emit plots.
# Image decoding --
# not the forward pass -- is the bottleneck, so each image is decoded once and
# fed to all four scales; request plenty of CPUs for parallel decode. The full
# embeddings (~15 GB across the four scales) are held in RAM during eval, hence
# the large --mem. Re-runs reuse the cached embeddings (see scripts/cache_check.py).
# PYTHONUNBUFFERED keeps tqdm/stdout flushing live in the SLURM log.
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
srun uv run python -m repsim.run experiment=dinov2_scale_similarity
