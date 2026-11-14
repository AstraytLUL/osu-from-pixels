#!/usr/bin/env bash
# Experiments 1 (B) and 2 (B), in the brief's priority order, each followed by its final evaluation.
# Safe to rerun after an interruption: finished runs and evaluations are skipped; an interrupted run
# is restarted from scratch (its directory is moved aside first).
#   bash rl/run_experiments.sh
set -u
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

run() {   # run <dir> <args...>: one PPO run unless final.pt exists
  local dir=$1; shift
  if [ -f "$dir/final.pt" ]; then return 0; fi
  if [ -d "$dir" ]; then mv "$dir" "$dir.interrupted.$(date +%s)"; fi
  $PY -m rl.ppo "$@" --out "$dir" || echo "FAILED $dir"
}

exp1() {  # exp1 <seed>
  local d=rl/runs/exp1_B_seed$1
  run $d --config rl/configs/exp1_hd.yaml --seed $1
  [ -f $d/final.pt ] && $PY -m rl.evaluate --label exp1_B --seed $1 --checkpoint $d/final.pt --maps heldout \
    --mods HD NM --out rl/results/exp1.csv
}

exp2() {  # exp2 <map> <seed>
  local d=rl/runs/exp2_B_$1_seed$2
  run $d --config rl/configs/exp2_map.yaml --map $1 --seed $2
  if [ -f $d/final.pt ]; then
    $PY -m rl.evaluate --label exp2_B_$1 --seed $2 --checkpoint $d/final.pt --maps $1 --mods NM --out rl/results/exp2.csv
    $PY -m rl.evaluate --label exp2_B_$1 --seed $2 --checkpoint $d/final.pt --maps heldout --mods NM --out rl/results/exp2.csv
  fi
}

exp1 0
for m in 4952940 555797 4377326; do exp2 $m 0; done
exp1 1
exp1 2
for s in 1 2; do for m in 4952940 555797 4377326; do exp2 $m $s; done; done
$PY -m rl.report
echo "experiment queue done"
