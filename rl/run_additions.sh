#!/usr/bin/env bash
# Additions beyond the brief (rl/RESULTS.md): Exp 1 condition D (PPO from C on HD + NoMod) and the
# C-mix control (supervised fine-tune on the replays as recorded + HD re-renders). Seed 0 first;
# seeds 1 2 for D if time allows:  SEEDS="1 2" ONLY=D bash rl/run_additions.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SEEDS=${SEEDS:-0}
ONLY=${ONLY:-all}

if [ "$ONLY" = all ] || [ "$ONLY" = Cmix ]; then
  if [ ! -f rl/runs/exp1_Cmix_seed0/best.pt ] || ! grep -q "best val accuracy" rl/runs/exp1_Cmix_seed0/train.log 2>/dev/null; then
    $PY -m rl.sft_hd --seed 0 --epochs 20 --name exp1_Cmix --mix || exit 1
  fi
  $PY -m rl.evaluate --label exp1_Cmix --seed 0 --checkpoint rl/runs/exp1_Cmix_seed0/best.pt \
      --maps heldout --mods HD NM --out rl/results/exp1.csv
fi &

if [ "$ONLY" = all ] || [ "$ONLY" = D ]; then
  for s in $SEEDS; do
    d=rl/runs/exp1_D_seed$s
    if [ ! -f $d/final.pt ]; then
      [ -d $d ] && mv $d ${d}_interrupted_$(date +%s)
      $PY -m rl.ppo --config rl/configs/exp1_D.yaml --seed $s || exit 1
    fi
    $PY -m rl.evaluate --label exp1_D --seed $s --checkpoint $d/final.pt --maps heldout --mods HD NM \
        --out rl/results/exp1.csv
  done
fi
wait
echo "additions done"
