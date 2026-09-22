#!/usr/bin/env bash
# The B2/B3 normalisation 2x2, over several seeds (VM / Linux).
#
# Vetter et al. standardise every respiratory/PPG signal on a 30 s TIME-WINDOW
# basis, which makes two windows with identical waveform SHAPE but a tenfold
# difference in AMPLITUDE numerically identical. For obstructive events that
# costs nothing; for central apnea it destroys the leading candidate precursor
# (declining respiratory effort relative to the infant's own baseline) by
# construction. This runs all four cells so the claim can be measured:
#
#   norm_window  per-window standardisation only  -- the published baseline
#   norm_block   per-block robust normalisation   -- the proposed primary
#   norm_both    both
#   norm_none    neither                          -- the control
#
# Seeds, because a single training run's seed-to-seed variance is of the same
# order as the effect being measured (see run_np_ablation_seeds.sh). Vetter et
# al. repeat training 10x per infant before their Wilcoxon signed-rank test.
#
# BEFORE RUNNING, verify the baseline cell still reproduces the published
# pipeline bit-for-bit -- without that the cells are not comparable to any
# earlier result:
#     python scripts/check_norm_chains.py
#
# And preview whether the effect exists at all, in minutes rather than hours:
#     python scripts/norm_ablation_preview.py
#
# Requires `pip install -e .` in the repo root (the src.* imports must resolve).
#
# Usage:
#   DATA_PATH=data/brainimmaturity \
#   RESULT_PATH=results \
#   SEEDS="0 1 2 3 4" \
#   bash scripts/run_norm_ablation.sh
set -euo pipefail

DATA_PATH="${DATA_PATH:?set DATA_PATH to the folder containing signals/ and annotations/}"
RESULT_PATH="${RESULT_PATH:?set RESULT_PATH to the folder the result pickles go to}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EXPERIMENT="${EXPERIMENT:-cpap_norm_ablation}"
DATASET="${DATASET:-neonatal}"
# CELLS optionally restricts which of the four are run, e.g. CELLS="norm_block"
# to re-run only the primary. Order is deliberate: baseline first, so a broken
# run is caught against a known number before three hours are spent.
CELLS="${CELLS:-norm_window norm_block norm_both norm_none}"
IDS="${IDS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== guard: baseline cell must reproduce the published pipeline ==="
python "$REPO_ROOT/scripts/check_norm_chains.py"
echo

cd "$REPO_ROOT/src"

extra=()
[ -n "$IDS" ] && extra+=("dataset.ids=$IDS")

for seed in $SEEDS; do
  for cell in $CELLS; do
    echo "=== seed ${seed}: ${cell} ==="
    python train_neonatal.py \
      dataset="$DATASET" +experiment="$cell" "${extra[@]}" \
      meta.experiment="$EXPERIMENT" meta.tag="${cell}_s${seed}" meta.seed="$seed" \
      meta.data_path="$DATA_PATH" meta.result_path="$RESULT_PATH"
  done
done

echo
echo "Done. Result pickles in ${RESULT_PATH}/${EXPERIMENT}/"
echo
echo "Report ALL FOUR cells, per B3. The comparison that carries the"
echo "hypothesis is norm_block vs norm_window, as a distribution of paired"
echo "per-infant differences (G4) with a Wilcoxon signed-rank over the 15"
echo "infants (G3) -- never as two averages."
