#!/usr/bin/env bash
# The clock-drift experiment (VM / Linux): does correcting the annotation-clock
# drift change what the model can learn?
#
# Runs the CPAP lag sweep TWICE -- once with the drift correction and once
# without -- over the same seeds, so every pair of numbers differs in exactly
# one thing. Without the paired uncorrected arm this says nothing: the CPAP
# prediction AuROC has always been ~0.52-0.56, and a corrected run landing there
# too is only interpretable next to its own control.
#
# What to expect, stated up front so the result is not read as a failure:
# detection (lag<=0) already scores ~0.87 WITH the misalignment, so a <=10 s
# error inside a 30 s window is clearly survivable, and this is not expected to
# turn 0.55 into 0.80. What it buys is that the remaining number is CLEAN -- the
# label-noise objection is closed, and "detectable but not predictable ahead"
# becomes a defensible finding rather than an uncertain one.
#
# Also runs the Robin-sequence cohort as a PIPELINE REFERENCE. That cohort is
# the one the original paper used, it is not drift-corrected (the skew was
# measured on the CPAP export only), and it should reproduce ~0.80. If it does,
# the pipeline is sound and the CPAP number is a property of the CPAP data. If
# it does not, the bug is ours and the CPAP arms are not worth interpreting yet.
# This is a far stronger control than reverting the branch, which cannot even
# run on the CPAP data: `main` hard-codes the Zenodo cohort's sampling rates
# (Thorax/Abdomen 50 Hz where this cohort is 10 Hz), has no CPAP-pressure
# channel, and crashes on the generic `APNEA` event type.
#
# Requires `pip install -e .` in the repo root (the src.* imports must resolve).
#
# Usage:
#   CPAP_DATA=data/brainimmaturity \
#   ROBIN_DATA=data/robin_sequence \
#   RESULT_PATH=results \
#   SEEDS="0 1 2" \
#   bash scripts/run_clock_drift_experiment.sh
#
# Evaluate afterwards with:
#   python scripts/clock_drift_stats.py --result_path $RESULT_PATH
set -euo pipefail

CPAP_DATA="${CPAP_DATA:?set CPAP_DATA to the CPAP folder containing signals/ and annotations/}"
RESULT_PATH="${RESULT_PATH:?set RESULT_PATH to the folder the result pickles go to}"
ROBIN_DATA="${ROBIN_DATA:-}"
SEEDS="${SEEDS:-0 1 2}"
# Lags in samples at 200 Hz. 0 = detection (window overlaps the event, the
# upper bound on what is representable at all); 3000 = the 15 s prediction task
# the thesis reports; 6000/12000 = 30 s and 60 s horizons.
LAGS="${LAGS:-0 3000 6000 12000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=== CPAP: drift-corrected vs uncorrected ==="
for seed in $SEEDS; do
  for lag in $LAGS; do
    # corrected: config/dataset/neonatal.yaml already points at
    # config/clock_drift.yaml, so this arm needs no override
    echo "--- seed=$seed lag=$lag corrected"
    python src/train_neonatal.py \
      meta.experiment=cpap_drift_corrected \
      meta.tag="lag_${lag}_seed_${seed}" \
      meta.seed="$seed" \
      meta.data_path="$CPAP_DATA" \
      meta.result_path="$RESULT_PATH" \
      dataset.lag="$lag"

    # uncorrected control: null switches the drift table off, which is exactly
    # the state every previously reported CPAP number was produced in
    echo "--- seed=$seed lag=$lag UNcorrected (control)"
    python src/train_neonatal.py \
      meta.experiment=cpap_drift_uncorrected \
      meta.tag="lag_${lag}_seed_${seed}" \
      meta.seed="$seed" \
      meta.data_path="$CPAP_DATA" \
      meta.result_path="$RESULT_PATH" \
      dataset.lag="$lag" \
      dataset.clock_drift_file=null
  done
done

if [ -n "$ROBIN_DATA" ]; then
  echo "=== Robin sequence: pipeline reference (expect ~0.80 at lag=3000) ==="
  for seed in $SEEDS; do
    python src/train_neonatal.py \
      meta.experiment=robin_reference \
      meta.tag="lag_3000_seed_${seed}" \
      meta.seed="$seed" \
      meta.data_path="$ROBIN_DATA" \
      meta.result_path="$RESULT_PATH" \
      dataset=robin_sequence \
      dataset.lag=3000
  done
else
  echo "ROBIN_DATA not set -- skipping the pipeline reference run."
  echo "Set it: without that control a CPAP number near 0.5 cannot be told"
  echo "apart from a pipeline bug."
fi

echo
echo "Done. Results under $RESULT_PATH/{cpap_drift_corrected,cpap_drift_uncorrected,robin_reference}/"
echo "Summarise with: python scripts/clock_drift_stats.py --result_path $RESULT_PATH"
