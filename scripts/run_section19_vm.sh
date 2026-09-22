#!/usr/bin/env bash
# Reproduce Section 4.5 of the thesis ON THE VM, and KEEP THE ARTEFACTS.
#
# WHY THIS EXISTS
#   #19 is currently a table of numbers with nothing behind it: the runs were
#   done on the VM with a RESULT_PATH that was never synced back, so the local
#   dataset_brainimmaturity/results/ holds only `robin_reference` (the #22
#   pipeline reference). If that VM is lost, #19 cannot be re-derived from any
#   artefact -- only re-run. This script re-runs it and puts every pickle in one
#   directory that is meant to be copied back.
#
# WHAT IT RUNS  (three dataset configs x 5 seeds x {with NP, without NP})
#   A  robin_sequence          all apnea/hypopnea subtypes   -> 0.757 expected
#   C  robin_sequence_capped   all subtypes, positives capped -> 0.684 expected
#   B  robin_sequence_central  central apnea only             -> 0.564 expected
#   All on the same 15 patients. 003/005/009/014 are dropped because they have
#   0-3 central-apnea positive windows and roc_auc_score needs both classes in
#   every leave-one-out test patient (scripts/count_windows.py).
#
# RUNTIME  30 runs. Budget several hours on GPU; check one run first (see STEP 2).
#
# USAGE
#   DATA_PATH=/path/to/robin RESULT_PATH=/path/to/results bash scripts/run_section19_vm.sh
set -euo pipefail

DATA_PATH="${DATA_PATH:?set DATA_PATH to the Robin folder containing signals/ and annotations_original/}"
RESULT_PATH="${RESULT_PATH:?set RESULT_PATH to a folder you will copy back}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# The 15 patients of #19: 001-019 minus 003, 005, 009, 014.
IDS='["001","002","004","006","007","008","010","011","012","013","015","016","017","018","019"]'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================================="
echo "data   : $DATA_PATH"
echo "results: $RESULT_PATH"
echo "seeds  : $SEEDS"
echo "=============================================================="

mkdir -p "$RESULT_PATH"

for DATASET in robin_sequence robin_sequence_capped robin_sequence_central; do
  echo
  echo "##############################################################"
  echo "# $DATASET"
  echo "##############################################################"
  DATA_PATH="$DATA_PATH" \
  RESULT_PATH="$RESULT_PATH" \
  SEEDS="$SEEDS" \
  DATASET="$DATASET" \
  IDS="$IDS" \
  EXPERIMENT="sec19_${DATASET}" \
  bash "$REPO_ROOT/scripts/run_np_ablation_seeds.sh"
done

echo
echo "=============================================================="
echo "ALL RUNS DONE. Pickles are in:"
for DATASET in robin_sequence robin_sequence_capped robin_sequence_central; do
  echo "  $RESULT_PATH/sec19_${DATASET}/"
done
echo
echo "Now compute the table (still on the VM, or locally after copying back):"
for DATASET in robin_sequence robin_sequence_capped robin_sequence_central; do
  echo "  python scripts/np_ablation_stats.py --result_path $RESULT_PATH \\"
  echo "      --experiment sec19_${DATASET} --seeds $SEEDS"
done
echo
echo "THEN COPY THE RESULTS BACK -- that is the entire point of this script:"
echo "  tar czf sec19_results.tgz -C $RESULT_PATH ."
echo "  # and scp / rsync sec19_results.tgz off the VM before shutting it down"
echo "=============================================================="
