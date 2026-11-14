#!/usr/bin/env bash
# Control for the Exp 2 transfer check (addition, 15 Nov 2026). The transfer check found that PPO on
# one hard map also improves other hard maps. Is that specific to practising a HARD map, or does the
# PPO recipe improve hard maps whatever it trains on? Two controls, evaluated exactly as
# rl/run_transfer.sh (the ten maps of rl/maps.yaml exp2_transfer, NoMod):
#   1. the existing sanity run: PPO on the whole BC training split, 300k steps, one seed;
#   2. the Exp 2 recipe unchanged (rl/configs/exp2_map.yaml, 1M steps, seeds 0-2) on one map of the
#      training range: 1570203, the highest-rated jump map of the BC training split (6.32 stars).
#      The map was fixed by that rule before any control run.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=rl/results/exp2_control.csv
MAP=1570203
$PY -m rl.evaluate --label ctrl_trainsplit_300k --checkpoint rl/runs/sanity_nm_seed0/final.pt \
    --maps exp2_transfer --mods NM --out $OUT
for s in 0 1 2; do
  $PY -m rl.ppo --config rl/configs/exp2_map.yaml --map $MAP --seed $s --out rl/runs/exp2_ctrl_${MAP}_seed$s
  $PY -m rl.evaluate --label exp2_ctrl_$MAP --seed $s --checkpoint rl/runs/exp2_ctrl_${MAP}_seed$s/final.pt \
      --maps exp2_transfer --mods NM --out $OUT
done
# As for the Exp 2 maps (rl/run_experiments.sh): the control map itself, before and after, and the
# held-out maps after.
OWN=rl/results/exp2_control_own.csv
$PY -m rl.evaluate --label exp2_A --checkpoint runs/baseline_cap_seed0/best.pt --maps $MAP --mods NM --out $OWN
for s in 0 1 2; do
  for maps in $MAP heldout; do
    $PY -m rl.evaluate --label exp2_ctrl_$MAP --seed $s --checkpoint rl/runs/exp2_ctrl_${MAP}_seed$s/final.pt \
        --maps $maps --mods NM --out $OWN
  done
done
echo "control done"
