#!/bin/bash
#SBATCH --job-name=repsim_both_eval
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=outputs/slurm_logs/both_eval_%j.log

# Eval-only re-run of BOTH experiments on CPU, reusing the cached full-pool
# embeddings (no model load, no image decode). Each run writes results.csv and
# regenerates every figure via plot_all. Both similarities (linear + RBF CKA)
# computed in one pass. All .npy caches verified present beforehand.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
srun uv run python -m repsim.run -m \
  experiment=hierarchically_local_similarity,dinov2_scale_similarity \
  device=cpu
