#!/bin/bash
#SBATCH --job-name=repsim_whiten
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm_logs/whiten_%j.log

# Re-evaluate the hierarchically_local_similarity experiment with PCA whitening,
# on CPU, reusing the cached full-pool embeddings (verified by scripts/cache_check.py).
# PYTHONUNBUFFERED keeps any non-logging stdout (and tqdm on stderr) flushing live.
export PYTHONUNBUFFERED=1
# Dataset + embeddings are fully cached; skip HF network round-trips (faster start, quieter log).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
srun uv run python -m repsim.run whiten=true device=cpu
