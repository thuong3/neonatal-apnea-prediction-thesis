#!/usr/bin/env bash
# F2: nested hyperparameter selection, including dropout (VM / Linux).
#
# RUN THIS ON THE GPU VM, NOT LOCALLY. Two separate reasons, and the second is
# the one that would quietly ruin the comparison:
#
#   1. Cost. Sized for a GPU at roughly 8 s/epoch with 11 training infants:
#      ~10 h for the inner grid plus ~2 h for the final refits. On a CPU that is
#      days.
#   2. Comparability. Every number this is measured against -- the 0.5827 best
#      cell, the 0.655 train AuROC, the 0.859 Robin train AuROC -- was produced
#      on the VM. Training is not bitwise reproducible even there (an identical
#      config and seed moved pooled AuROC by 0.022, section 33.6), and a different
#      device is a larger perturbation than that. A CPU-run baseline would not
#      be the baseline.
#
# BEFORE RUNNING, prove the architecture is still the published one, so that the
# tuned arms can be compared against every existing result:
#
#     python scripts/check_dropout_identity.py
#
# WHAT TO READ IN THE OUTPUT. Not only the outer mean test AuROC. The number
# that decides how the thesis is written is `mean_train_auc`:
#
#   stays near 0.655  -> "the model cannot fit the CPAP training windows" is now
#                        measured across a capacity and schedule sweep, not
#                        assumed from one configuration. Section 22 gets stronger.
#   rises well above  -> the null holds, but its explanation changes to "it fits
#                        and none of it generalises". Section 4.3, the figure
#                        `underfitting_and_central` and section 5.1 need rewriting.
#
# Quote the OUTER mean. It estimates the tuned pipeline. The winning
# configuration's own inner-validation score is selected on the same numbers it
# would be reported with, so it is descriptive only.
#
# Requires `pip install -e .` in the repo root.
#
# Usage. The real VM paths, not placeholders -- these lines get pasted, and
# /path/to/... resolves to a FileNotFoundError eight seconds in.
#
#   DATA=data/brainimmaturity RESULTS=results \
#     bash scripts/run_nested_hp.sh
#
#   # cheap first pass, ~2 h, three configurations
#   DATA=data/brainimmaturity RESULTS=results \
#     CONFIGS='[base,wide_long,drop25]' bash scripts/run_nested_hp.sh
set -euo pipefail

DATA="${DATA:?set DATA to the dataset_brainimmaturity path}"
RESULTS="${RESULTS:?set RESULTS to the result path}"
EXPERIMENT="${EXPERIMENT:-cpap_hp}"
SEEDS="${SEEDS:-0}"
CONFIGS="${CONFIGS:-}"

cd "$(dirname "$0")/.."
python scripts/check_dropout_identity.py

for seed in $SEEDS; do
  extra=()
  if [ -n "$CONFIGS" ]; then extra+=("+hp.configs=$CONFIGS"); fi
  echo "=== nested hyperparameter selection, seed $seed ==="
  PYTHONPATH=. python src/train_nested_hp.py \
    dataset=neonatal \
    meta.experiment="$EXPERIMENT" \
    meta.tag="nested_seed${seed}" \
    meta.seed="$seed" \
    meta.data_path="$DATA" \
    meta.result_path="$RESULTS" \
    "${extra[@]}"
done

echo
echo "Per-fold selections and both AuROCs are in"
echo "  $RESULTS/$EXPERIMENT/nested_seed*_nested.json"
echo
echo "One seed bounds the effect; it does not establish it. If the outer mean"
echo "moves the headline, re-run the SELECTED configuration through the normal"
echo "pipeline at the pre-specified 10 seeds before quoting anything."
