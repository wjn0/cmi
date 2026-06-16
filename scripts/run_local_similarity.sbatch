#!/bin/bash
#SBATCH --job-name=repsim_local_sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_logs/local_similarity_%j.log

# local_similarity_performance: embed PCam + CelebA train/test with the
# DINOv2/SigLIP/MAE trio (balanced per-class sample), fit a logistic linear probe
# per (model, dataset) for held-out accuracy, and score every ordered model pair.
# Only `similarity=linear` is run: the hypothesis is about ONE-WAY (asymmetric)
# predictivity, and RBF CKA is symmetric, so it carries no directional signal here.
# As with the granularity runs, image decode (not the forward pass) is the
# bottleneck, so each image is decoded once and fed to all three models -- request
# plenty of CPUs. MUST pin a modern card (L40S, Ada): the gpu partition's
# 1080ti/2080ti emulate bf16 ~100x slower. PYTHONUNBUFFERED keeps tqdm live.
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
srun uv run python -m repsim.run experiment=local_similarity_performance similarity=linear
