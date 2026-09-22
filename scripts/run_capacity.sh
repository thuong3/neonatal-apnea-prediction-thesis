#!/usr/bin/env bash
# The capacity contrast: does the model fit CPAP when given the capacity, and
# does anything it learns transfer? (VM / Linux.)
#
# F2's nested run said yes and no respectively -- train 0.871, test 0.424 --
# but on 3 outer folds that SELECTION chose, so the arms were not paired and
# the fold set was not random (009 is the dead-abdomen-belt recording, 001 the
# worst fold in the original logs). See Section 4.3 of the thesis.
#
# This runs the two configurations as FIXED arms over all 15 infants, so the
# per-infant differences are paired and the Wilcoxon that PRE_SPECIFICATION.md
# 1.1 requires is actually defined:
#
#   base       20 hidden, 10 epochs   the published configuration
#   wide_long  40 hidden, 40 epochs   the capacity arm
#
# Both arms see identical windows and labels -- only the network and the
# schedule differ -- which scripts/capacity_stats.py asserts before testing.
#
# COST. base is ~30 min/seed, wide_long ~2 h/seed, so ~2.5 h per seed.
# SEEDS="0" is a 2.5 h first look; the default 0 1 2 is an overnight run and is
# the minimum for quoting anything, since one seed cannot separate a real
# difference from the 0.022 the training procedure moves on its own (33.6).
#
# RUN IT DETACHED. This outlives an SSH session only inside tmux:
#
#     tmux new -s cap
#     DATA=data/brainimmaturity RESULTS=results \
#       bash scripts/run_capacity.sh
#     # Ctrl-B then D to detach; `tmux attach -t cap` to come back
#
# Then:
#     python scripts/capacity_stats.py --seeds 0 1 2
set -euo pipefail

DATA="${DATA:?set DATA to the dataset_brainimmaturity path}"
RESULTS="${RESULTS:?set RESULTS to the result path}"
EXPERIMENT="${EXPERIMENT:-cpap_capacity}"
SEEDS="${SEEDS:-0 1 2}"

cd "$(dirname "$0")/.."

# Guard first: the tuned arms are only comparable with everything already
# reported if dropout_p=0 still leaves the published architecture untouched.
python scripts/check_dropout_identity.py

for seed in $SEEDS; do
  # The published configuration. Re-run here rather than reused from an older
  # experiment so that both arms share seeds, windows and code version -- the
  # comparison is paired or it is nothing.
  echo "=== base (20 hidden, 10 epochs), seed $seed ==="
  PYTHONPATH=. python src/train_neonatal.py dataset=neonatal \
    meta.experiment="$EXPERIMENT" meta.tag="base_s${seed}" meta.seed="$seed" \
    meta.data_path="$DATA" meta.result_path="$RESULTS"

  echo "=== wide_long (40 hidden, 40 epochs), seed $seed ==="
  PYTHONPATH=. python src/train_neonatal.py dataset=neonatal \
    meta.experiment="$EXPERIMENT" meta.tag="wide_long_s${seed}" meta.seed="$seed" \
    meta.data_path="$DATA" meta.result_path="$RESULTS" \
    network.hidden_channels='[40,40,40,40,40,40]' \
    optimizer.epochs=40
done

echo
echo "Analyse with:"
echo "  python scripts/capacity_stats.py --result_path $RESULTS \\"
echo "      --experiment $EXPERIMENT --seeds $SEEDS"
echo
echo "Read the TRAIN AuROC as well as the test one. The claim being tested is"
echo "that the capacity arm fits CPAP as well as the published config fits"
echo "Robin (0.859) while testing at or below chance."
