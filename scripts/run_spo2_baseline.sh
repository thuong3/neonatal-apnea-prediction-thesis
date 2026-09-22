#!/usr/bin/env bash
# Reproduce the SpO2 baseline-drift analysis (Section 4.6 of the thesis).
#
# Two phenomena are separated here:
#   A  frank desaturation / intermittent hypoxemia (SpO2 < 90 % for >= 5 s)
#   B  decline of the BASELINE -- the level between the dips
# and B is tested both as a FEATURE and as a prediction TARGET of its own.
#
# Usage: scripts/run_spo2_baseline.sh [<data_path>]
set -euo pipefail

DATA="${1:-data/brainimmaturity}"
OUT="outputs/spo2_baseline"
FIG="figures/06_spo2_baseline"
mkdir -p "$OUT" "$FIG"

# 1. Feature build + full ablation. The .npz cache makes every re-run of the
#    modelling side instant; delete it to rebuild from the EDFs (~5 min).
python scripts/spo2_baseline_prediction.py "$DATA" \
    --cache "$OUT/features.npz" | tee "$OUT/ablation_results.txt"

# 2. Cohort-wide survey of how far the baseline actually moves, per segment.
python scripts/make_spo2_baseline_figure.py --survey --data_path "$DATA" \
    | tee "$OUT/baseline_survey.txt"

# 3. The two example figures cited in the write-up.
#    008/1: baseline falls ~100 -> ~92 % while IH burden rises to 40 % (A and B
#           together). 010/3: baseline sits at ~94 % with heavy IH burden for
#           2.5 h, then steps to ~99 % and the dips stop.
python scripts/make_spo2_baseline_figure.py --patient 008 --segment 1 --data_path "$DATA"
python scripts/make_spo2_baseline_figure.py --patient 010 --segment 3 --data_path "$DATA"

echo "done -> $OUT and $FIG"
