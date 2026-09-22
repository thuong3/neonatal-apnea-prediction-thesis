"""Precursor-feature gradient-boosting baseline for apnea PREDICTION.

Tests whether hand-crafted precursor features (SpO2/HR dynamics, respiratory
regularity, periodic-breathing power) computed on the pre-onset window can
predict a central apnea ahead of time, using leave-one-patient-out gradient
boosting. See Section 4.4 of the thesis.

Usage:
    python scripts/precursor_gbm.py <data_path> [lag_samples]
    # <data_path> contains signals/ and annotations/ ; lag default 3000 (15 s).
"""
import sys
import numpy as np
from scipy import signal as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.neonatal_utils import create_time_windows, load_clock_drift, read_data

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset_brainimmaturity"
LAG = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
WIN, AWAY, FS = 6000, 36000, 200
IDS = [f"{i:03d}" for i in range(1, 16)]

# CPAP cohort: annotations run ~158 ppm fast against the signals, so the
# event masks this script reads must be drift-corrected too, not just the
# ones the network trains on. See scripts/measure_clock_drift.py.
DRIFT = load_clock_drift()
ADVERSE, CUTTER = ["APNEA-CENTRAL"], ["SIGNAL-ARTIFACT"]


def stats(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return [0, 0, 0, 0, 0]
    return [x.mean(), x.std(), x.min(), x.max(), np.polyfit(np.arange(len(x)), x, 1)[0]]


def var_stats(x):
    d = np.diff(np.asarray(x, float))
    return [np.std(d) if len(d) else 0.0, np.mean(np.abs(d)) if len(d) else 0.0]


def periodic_power(env2):
    if len(env2) < 8:
        return 0.0
    f, P = sp.periodogram(env2 - env2.mean(), fs=2.0)
    return float(P[(f >= 0.01) & (f <= 0.1)].sum() / (P.sum() + 1e-9))


def window_features(seg, sl):
    feats = []
    for ch, step in [("SpO2", 100), ("HR", 200), ("PCO2", 100)]:
        x = seg[ch][sl][::step]
        feats += stats(x) + var_stats(x)
    for ch in ["Thorax", "Abdomen", "CPAP"]:
        env = np.abs(seg[ch][sl].astype(float))
        k = FS // 2
        env2 = env[: len(env) // k * k].reshape(-1, k).mean(1) if len(env) >= k else env
        feats += stats(env2) + var_stats(env2) + [periodic_power(env2)]
    return feats


def main():
    X, y, groups = [], [], []
    for pid in IDS:
        for seg in read_data(pid, DATA, clock_drift=DRIFT.get(pid)):
            anti = 1 - np.column_stack([seg[e] for e in ADVERSE]).max(1)
            cutter = 1 - np.column_stack([seg[e] for e in (ADVERSE + CUTTER)]).max(1)
            for r in create_time_windows(anti, cutter, WIN, AWAY, LAG):
                X.append(window_features(seg, r["slice"]))
                y.append(r["label"])
                groups.append(pid)
    X, y, groups = np.array(X, float), np.array(y), np.array(groups)
    print(f"{len(y)} windows, {int(y.sum())} positive, {X.shape[1]} features")

    aucs, aps, ALLY, ALLP = [], [], [], []
    for pid in IDS:
        tr, te = groups != pid, groups == pid
        if len(set(y[te].tolist())) < 2:
            continue
        npos, nneg = int(y[tr].sum()), int((y[tr] == 0).sum())
        w = np.where(y[tr] == 1, len(y[tr]) / (2 * npos), len(y[tr]) / (2 * nneg))
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0
        )
        clf.fit(X[tr], y[tr], sample_weight=w)
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        ALLY.append(y[te])
        ALLP.append(p)
        print(f"  {pid}: test AUC={aucs[-1]:.3f}")
    ALLY, ALLP = np.concatenate(ALLY), np.concatenate(ALLP)
    print(
        f"\nlag={LAG}: mean test AUC={np.mean(aucs):.3f} | mean AP={np.mean(aps):.3f} | "
        f"pooled AUC={roc_auc_score(ALLY, ALLP):.3f}"
    )


if __name__ == "__main__":
    main()
