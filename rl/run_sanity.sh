#!/usr/bin/env bash
# Baselines with per-hit errors, condition C's evaluation, and the two PPO sanity runs (the brief:
# no experiment starts until both pass). Safe to rerun: finished steps are skipped.
#   bash rl/run_sanity.sh
set -u
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
BC=runs/baseline_cap_seed0/best.pt

# condition A for both experiments (rl/evaluate.py skips rows already written)
$PY -m rl.evaluate --label exp1_A --checkpoint $BC --maps heldout --mods HD NM --out rl/results/exp1.csv
$PY -m rl.evaluate --label exp2_A --checkpoint $BC --maps exp2 --mods NM --out rl/results/exp2.csv

# condition C (rl/sft_hd.py must have finished)
if grep -q "best val (HD)" rl/runs/exp1_C_seed0/train.log 2>/dev/null; then
  $PY -m rl.evaluate --label exp1_C --checkpoint rl/runs/exp1_C_seed0/best.pt --maps heldout --mods HD NM --out rl/results/exp1.csv
fi

# sanity 1: learning rate 0 must reproduce BC (curve on the val maps at step 0 and at the end)
[ -f rl/runs/sanity_lr0_seed0/final.pt ] || $PY -m rl.ppo --config rl/configs/sanity_nm.yaml --lr-scale 0 \
  --total-steps 20480 --out rl/runs/sanity_lr0_seed0
# sanity 2: a short NoMod fine-tune on the training maps must not degrade the policy
[ -f rl/runs/sanity_nm_seed0/final.pt ] || $PY -m rl.ppo --config rl/configs/sanity_nm.yaml --out rl/runs/sanity_nm_seed0
echo "sanity queue done"
