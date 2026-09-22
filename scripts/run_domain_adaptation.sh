#!/usr/bin/env bash
# Cross-cohort domain adaptation, Robin sequence -> CPAP (VM / Linux).
# See Section 4.5 of the thesis for what each stage measures.
#
# Runs three stages:
#   diag  the source-internal diagnostic, 5ch and 6ch (Section 4.5 of the thesis)
#   full  the seven-variant transfer ladder, once per seed
#   coral the CORAL lambda sweep (the default lambda must not be trusted alone)
#
# Requires `pip install -e .` in the repo root (the src.* imports must resolve).
#
# Usage:
#   SOURCE_PATH=neonatal_robin_polysomnography \
#   TARGET_PATH=data/brainimmaturity \
#   RESULT_PATH=results \
#   STAGE=diag \
#   bash scripts/run_domain_adaptation.sh
#
# Run STAGE=diag FIRST and read its number before spending time on STAGE=full:
# if the source model is near chance in the shared 5-channel space, the ladder
# cannot be interpreted (Section 4.5 of the thesis).
set -euo pipefail

SOURCE_PATH="${SOURCE_PATH:?set SOURCE_PATH to the Robin cohort folder}"
RESULT_PATH="${RESULT_PATH:?set RESULT_PATH to the folder the result pickles go to}"
TARGET_PATH="${TARGET_PATH:-}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXPERIMENT="${EXPERIMENT:-domain_adaptation}"
STAGE="${STAGE:-diag}"
LAMBDAS="${LAMBDAS:-0 500 5000 50000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

run() {
  python -m src.train_domain_adaptation \
    meta.experiment="$EXPERIMENT" \
    source.data_path="$SOURCE_PATH" \
    meta.result_path="$RESULT_PATH" \
    "$@"
}

case "$STAGE" in
  diag)
    # Diagnostic only: `da.variants=[]` skips the target cohort entirely.
    for seed in $SEEDS; do
      echo "=== seed ${seed}: source-internal, 5 shared channels ==="
      run meta.tag="diag_5ch_s${seed}" meta.seed="$seed" 'da.variants=[]'

      echo "=== seed ${seed}: source-internal, 6 channels (incl. NP) ==="
      run meta.tag="diag_6ch_s${seed}" meta.seed="$seed" 'da.variants=[]' \
        network=nam_source6 \
        'dataset.signal_types=[NP,Thorax,HR,PR,SpO2,PCO2]'
    done
    ;;

  full)
    : "${TARGET_PATH:?set TARGET_PATH to the CPAP cohort folder}"
    for seed in $SEEDS; do
      echo "=== seed ${seed}: full transfer ladder ==="
      # The diagnostic was already measured in STAGE=diag; skip it here so it
      # does not repeat a 19-patient leave-one-out on every seed.
      run meta.tag="da_s${seed}" meta.seed="$seed" \
        target.data_path="$TARGET_PATH" \
        da.run_source_internal=false
    done
    ;;

  coral)
    : "${TARGET_PATH:?set TARGET_PATH to the CPAP cohort folder}"
    for seed in $SEEDS; do
      for lam in $LAMBDAS; do
        echo "=== seed ${seed}: coral lambda=${lam} ==="
        run meta.tag="coral_l${lam}_s${seed}" meta.seed="$seed" \
          target.data_path="$TARGET_PATH" \
          da.run_source_internal=false \
          'da.variants=[coral]' da.coral_lambda="$lam"
      done
    done
    ;;

  *)
    echo "Unknown STAGE '$STAGE' (expected: diag | full | coral)" >&2
    exit 1
    ;;
esac

echo
echo "Done. Result pickles in ${RESULT_PATH}/"
echo "Summarise them with: python scripts/da_summary.py --result_path ${RESULT_PATH}"
