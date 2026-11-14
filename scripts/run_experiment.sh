#!/usr/bin/env bash
# One architecture experiment, start to finish: train, transfer matrix on the val maps, upload.
#
#     bash scripts/run_experiment.sh configs/arch/cnn3d.yaml 0          # seed 0
#     bash scripts/run_experiment.sh configs/baseline.yaml 1 --no-push  # keep it local
#
# Rerunning the same command after an interruption resumes training from last.pt, and skips the
# steps already done. Writes runs/<name>_seed<seed>/ (train/bc.py) and, unless --no-push, uploads
# it to the private repo sc4001-project/bc-<name>, folder seed<seed>/ (train/hub.py).
set -euo pipefail
cd "$(dirname "$0")/.."

config=${1:?usage: run_experiment.sh CONFIG SEED [--no-push]}
seed=${2:?usage: run_experiment.sh CONFIG SEED [--no-push]}
push=1
[[ "${3:-}" == "--no-push" ]] && push=0

name=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$config")
run="runs/${name}_seed${seed}"
echo "== $name seed $seed -> $run"

# 1. train (or finish training)
resume=()
[[ -f "$run/last.pt" ]] && resume=(--resume)
epochs=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['train']['epochs'])" "$config")
done_epochs=0
[[ -f "$run/metrics.jsonl" ]] && done_epochs=$(wc -l < "$run/metrics.jsonl")
if (( done_epochs < epochs )); then
    python -m train.bc --config "$config" --seed "$seed" "${resume[@]}"
fi

# 2. transfer matrix on the val maps (a diagnostic; test only for final results)
if [[ ! -f "$run/transfer_val.json" ]]; then
    python -m eval.transfer_matrix --checkpoint "$run/best.pt" --split val --device cuda \
        --mods NM HR DT HD HRDT --out "$run/transfer_val.json" | tee -a "$run/train.log"
fi

# 3. upload
if (( push )); then
    python -m train.hub push "$run"
fi
echo "== done: $run"
