#!/usr/bin/env bash
# Robin-sequence NP ablation over several seeds (VM / Linux).
#
# Runs the 6-channel model (incl. NP) and the 5-channel ablation (+experiment=
# no_np) once per seed, so the comparison can be averaged over training
# repetitions before the paired test -- Vetter et al. repeat training 10x per
# infant before their Wilcoxon signed-rank test over patients. A single seed
# gives a delta (~0.015) of the same order as the seed-to-seed variance.
#
# Requires `pip install -e .` in the repo root (the src.* imports must resolve).
#
# Usage:
#   DATA_PATH=data/robin_sequence \
#   RESULT_PATH=results \
#   SEEDS="0 1 2 3 4" \
#   bash scripts/run_np_ablation_seeds.sh
#
# Evaluate afterwards with:
#   python scripts/np_ablation_stats.py --result_path $RESULT_PATH \
#       --experiment robin_np_ablation_seeds --seeds 0 1 2 3 4
set -euo pipefail

DATA_PATH="${DATA_PATH:?set DATA_PATH to the folder containing signals/ and annotations/}"
RESULT_PATH="${RESULT_PATH:?set RESULT_PATH to the folder the result pickles go to}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXPERIMENT="${EXPERIMENT:-robin_np_ablation_seeds}"
# DATASET selects the event definition: `robin_sequence` = all apnea/hypopnea
# subtypes (the paper's task), `robin_sequence_central` = central apnea only
# (matches the CPAP cohort's event definition).
DATASET="${DATASET:-robin_sequence}"
# IDS optionally restricts the patient set, e.g. to drop patients that have too
# few positive windows under a restricted event definition. Check first with
# scripts/count_windows.py. Format: '["001","002"]'
IDS="${IDS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/src"

extra=()
[ -n "$IDS" ] && extra+=("dataset.ids=$IDS")

for seed in $SEEDS; do
  echo "=== seed ${seed}: with NP (6 channels) ==="
  python train_neonatal.py \
    dataset="$DATASET" "${extra[@]}" \
    meta.experiment="$EXPERIMENT" meta.tag="with_np_s${seed}" meta.seed="$seed" \
    meta.data_path="$DATA_PATH" meta.result_path="$RESULT_PATH"

  echo "=== seed ${seed}: without NP (5 channels) ==="
  python train_neonatal.py \
    dataset="$DATASET" +experiment=no_np "${extra[@]}" \
    meta.experiment="$EXPERIMENT" meta.tag="without_np_s${seed}" meta.seed="$seed" \
    meta.data_path="$DATA_PATH" meta.result_path="$RESULT_PATH"
done

echo
echo "Done. Result pickles in ${RESULT_PATH}/${EXPERIMENT}/"
echo "Now run: python scripts/np_ablation_stats.py --result_path ${RESULT_PATH} \\"
echo "             --experiment ${EXPERIMENT} --seeds ${SEEDS}"
