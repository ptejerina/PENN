#!/bin/bash
#SBATCH --job-name=penn_campaign
#SBATCH --array=0-119
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH -o output/campaign_%A_%a.o
#SBATCH -e output/campaign_%A_%a.e

# The whole campaign as one Slurm array: 3 geometries x 4 configurations x 10 seeds = 120 runs.
# Array index = 40*geom + 10*config + seed, with geom in (circle, torus, torus4) and
# config in (smooth, rawsoft, min, folded).  Submit from the repository root after
#   mkdir -p output
# Adapt the module/activation lines to your cluster.  Wall time: ~1 h (S^1) to ~16 h
# (min-of-charts on T^4, 16 charts per point) on an NVIDIA L40S; the other T^4 runs
# are latency-bound (~2-2.5 h) and run on any GPU.  Resubmitting does not resume: delete
# the run folder first, or append --resume to the command in its cmd.txt.

module purge
# source /path/to/venv/bin/activate
# module load cuda
set -euo pipefail
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

GEOMS=(circle torus torus4)
CONFIGS=(smooth rawsoft min folded)
i=$SLURM_ARRAY_TASK_ID
geom=${GEOMS[$((i / 40))]}; config=${CONFIGS[$(((i % 40) / 10))]}; seed=$((i % 10))
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}     # submit from the repository root
PY=python bash "$ROOT/launch/train_run.sh" "$geom" "$config" "$seed"
