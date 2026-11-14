#!/usr/bin/env bash
# Exp 2 transfer check (addition): BC and the nine single-map PPO policies on the hard test maps
# (rl/maps.yaml exp2_transfer), NoMod. Does a specialist help other hard maps of its style?
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=rl/results/exp2_transfer.csv
$PY -m rl.evaluate --label exp2_A --checkpoint runs/baseline_cap_seed0/best.pt --maps exp2_transfer --mods NM --out $OUT
for m in 4952940 555797 4377326; do
  for s in 0 1 2; do
    $PY -m rl.evaluate --label exp2_B_$m --seed $s --checkpoint rl/runs/exp2_B_${m}_seed$s/final.pt \
        --maps exp2_transfer --mods NM --out $OUT
  done
done
echo "transfer done"
