#!/bin/bash
#SBATCH --job-name=repsim_dinov2_scales_eval
#SBATCH --partition=defaultp
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm_logs/dinov2_scales_eval_%j.log

# Eval-only re-run of dinov2_scale_similarity on CPU, reusing the cached full-pool
# embeddings (the four DINOv2 scales, already embedded on GPU). The eval is pure
# CPU work (per-node affine least-squares lstsq + RBF-CKA Gram for 12 ordered
# scale pairs at n_fit=7680 -- both similarities run in this one pass), so it has
# no business holding an expensive GPU node -- run it here with a generous
# walltime so it can't hit a wall before writing results.csv.
# The per-node eval fans out across all 32 CPUs (n_jobs=-1). Each worker builds its
# own RBF-CKA Gram matrices, which are O(n_fit^2) (~1.9 GB across the four scales at
# n_fit=7680); 32 concurrent workers is why --mem is 120G, not 64G. The cached
# embeddings are memory-mapped (read-only, shared across workers via the page cache).
# All four .npy caches must exist under outputs/embeddings (verify w/ cache_check).
export PYTHONUNBUFFERED=1
# Dataset + embeddings are fully cached; skip HF network round-trips.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
srun uv run python -m repsim.run experiment=dinov2_scale_similarity device=cpu
