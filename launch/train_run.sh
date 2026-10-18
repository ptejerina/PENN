#!/usr/bin/env bash
# Train one run of the campaign with the exact flags used in the paper.
#
#   bash launch/train_run.sh <geom> <config> <seed>
#       geom    : circle | torus | torus4
#       config  : smooth (Fourier-feature) | rawsoft (soft-periodic) | min (min-of-charts) | folded (geometric)
#       seed    : integer (the paper uses 0-9)
#
# The run is written to runs_fullhorizon/{S1,T2,T4}/<geom>_<config>_s<seed>/ (cmd.txt, train.log,
# checkpoint gpu_<geom>[_rawsoft|_charts|_charts1].pt and its .json history), the layout read by
# analysis/eval_all.py and the notebook.  Budget: 5e5 iterations on S^1, 1e6 on T^2 and T^4.
# Resume an interrupted run by re-running the command in cmd.txt with --resume <checkpoint> appended.
# Environment: PY (python executable, default python), ITERS (override the budget).
set -euo pipefail
geom=${1:?geom}; config=${2:?config}; seed=${3:?seed}
PY=${PY:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case $geom in
  circle) sub=S1; iters=500000 ;;
  torus)  sub=T2; iters=1000000 ;;
  torus4) sub=T4; iters=1000000 ;;
  *) echo "geom must be circle, torus or torus4" >&2; exit 1 ;;
esac
case $config in
  smooth)  flag="" ;;
  rawsoft) flag="--raw_soft" ;;
  min)     flag="--charts" ;;
  folded)  flag="--charts_single" ;;
  *) echo "config must be smooth, rawsoft, min or folded" >&2; exit 1 ;;
esac
iters=${ITERS:-$iters}
dir=$ROOT/runs_fullhorizon/$sub/${geom}_${config}_s${seed}
mkdir -p "$dir"; cd "$dir"
cmd="$PY -u $ROOT/gpu_critic.py --geom $geom $flag --iters $iters --width 128 --bc_margin 1.0 --restarts 1 --tau_min 0.0 --seed $seed"
echo "$cmd" > cmd.txt
echo "[$(date)] $cmd"
$cmd 2>&1 | tee train.log
