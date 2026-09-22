"""RANDOM CROSS-VALIDATION AGAINST A TEMPORAL SPLIT, WITHIN EACH INFANT.

Section 4.7 reports that personalised models reach 0.671 under random five-fold
cross-validation within the infant and 0.551 under a temporal split, and that
the infants that looked strongest collapse the furthest. Those numbers were
documented in Section 4.4 of the thesis SS8.1 in July 2026; the code that
produced them was never committed, and no per-infant table survives. This
script re-runs the experiment so that the claim rests on something the
repository can regenerate, and draws the per-infant figure.

WHAT IS COMPARED
    Each infant is modelled on its OWN windows, nothing pooled. The two arms
    differ only in how that infant's windows are divided:

      RANDOM      stratified five-fold cross-validation over the windows in
                  arbitrary order, scored on the pooled out-of-fold
                  predictions.
      TEMPORAL    the earlier 60 % of the recording trains, the later 40 %
                  tests, in recording order.

    Windows, labels, features and classifier are identical between the arms and
    are taken from scripts/precursor_gbm.py unchanged: 30 s windows ending 15 s
    before a scored central apnea against controls at least three minutes from
    any event, SIGNAL-ARTIFACT spans cut, annotations drift-corrected, and a
    class-weighted HistGradientBoostingClassifier at the same four settings.
    The only difference between the two numbers is therefore the split, which
    is what makes the gap attributable to it.

WHY THE GAP IS LEAKAGE AND NOT VARIANCE
    Neighbouring windows in a physiological recording are not independent. Under
    a random split a window minutes away from its own neighbour can sit in
    training while that neighbour sits in test, so the model can interpolate
    between them rather than predict forward. A temporal split makes that
    impossible by construction. Nothing else changes.

REPRODUCTION
    The per-infant AUROCs are written to outputs/leakage/per_patient.csv, so a
    later run can be compared against this one line by line. Feature extraction
    reads all fifteen EDFs and is cached in outputs/leakage/features.npz;
    --replot skips it entirely.

Usage:
    python scripts/leakage_personalisation.py
    python scripts/leakage_personalisation.py --replot      # from cache
    python scripts/leakage_personalisation.py --simple      # talk version
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
matplotlib.rc_file(os.path.join(ROOT, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

from src.neonatal_utils import (  # noqa: E402
    create_time_windows,
    load_clock_drift,
    read_data,
)

# precursor_gbm.py reads its data path and lag from sys.argv AT MODULE LEVEL,
# so importing it with this script's own flags in place would fail on
# int("--out_dir"). The features must come from that file rather than a copy:
# the whole point of the comparison is that the two arms differ only in the
# split, and a re-typed feature function is one more thing that could differ.
_argv = sys.argv
sys.argv = [_argv[0]]
try:
    from scripts.precursor_gbm import window_features  # noqa: E402
finally:
    sys.argv = _argv

DATA = "data/brainimmaturity"
IDS = ["%03d" % i for i in range(1, 16)]
WIN, AWAY, LAG = 6000, 36000, 3000      # 30 s window, 3 min away, 15 s lead
ADVERSE, CUTTER = ["APNEA-CENTRAL"], ["SIGNAL-ARTIFACT"]
N_FOLDS = 5
TRAIN_FRAC = 0.60                        # the earlier 60 % trains
SEED = 0
MIN_POS = 5                              # below this a within-infant fit is noise

CACHE = os.path.join(ROOT, "outputs/leakage/features.npz")
PER_PAT = os.path.join(ROOT, "outputs/leakage/per_patient.csv")

C_RANDOM = "#eb6834"
C_TEMPORAL = "#2a78d6"
C_LINE = "#b9b8b3"
C_GUIDE = "#c9c8c3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "thesis/figures"))
    p.add_argument("--out", default="outputs/leakage/summary.txt")
    p.add_argument("--replot", action="store_true",
                   help="use the cached features instead of reading the EDFs")
    p.add_argument("--simple", action="store_true",
                   help="talk version: no title, no annotations")
    p.add_argument("--data", default=DATA)
    return p.parse_args()


def clf():
    """The classifier of scripts/precursor_gbm.py, unchanged."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=SEED)


def weights(y):
    """Balanced sample weights, as precursor_gbm.py computes them."""
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    return np.where(y == 1, len(y) / (2 * npos), len(y) / (2 * nneg))


def extract(data):
    """Per-infant windows, with a monotone time key for the temporal split."""
    drift = load_clock_drift()
    X, y, pid_of, t_of = [], [], [], []
    for pid in IDS:
        n0 = len(y)
        for si, seg in enumerate(read_data(pid, data, clock_drift=drift.get(pid))):
            anti = 1 - np.column_stack([seg[e] for e in ADVERSE]).max(1)
            cutter = 1 - np.column_stack(
                [seg[e] for e in (ADVERSE + CUTTER)]).max(1)
            for r in create_time_windows(anti, cutter, WIN, AWAY, LAG):
                X.append(window_features(seg, r["slice"]))
                y.append(r["label"])
                pid_of.append(pid)
                # Block index first, sample offset second: the blocks are
                # recorded in order, so this sorts the whole recording.
                t_of.append(si * 10 ** 9 + r["slice"].start)
        print("  %s: %d windows, %d positive"
              % (pid, len(y) - n0, int(np.sum(y[n0:]))))
    return (np.array(X, float), np.array(y, int),
            np.array(pid_of), np.array(t_of, float))


def evaluate(X, y, t):
    """(random 5-fold OOF AUROC, mean-over-folds AUROC, temporal AUROC)."""
    out = {}

    # RANDOM: stratified five-fold, scored on pooled out-of-fold predictions.
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        w = weights(y[tr])
        if w is None or len(set(y[te])) < 2:
            continue
        m = clf().fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict_proba(X[te])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], oof[te]))
    ok = ~np.isnan(oof)
    out["random"] = (roc_auc_score(y[ok], oof[ok])
                     if len(set(y[ok])) > 1 else np.nan)
    out["random_foldmean"] = float(np.mean(fold_aucs)) if fold_aucs else np.nan

    # TEMPORAL: the earlier 60 % of the recording trains, the later 40 % tests.
    order = np.argsort(t)
    cut = int(round(TRAIN_FRAC * len(order)))
    tr, te = order[:cut], order[cut:]
    w = weights(y[tr])
    if w is None or len(set(y[te])) < 2:
        out["temporal"] = np.nan
    else:
        m = clf().fit(X[tr], y[tr], sample_weight=w)
        out["temporal"] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    return out


def run(X, y, pid_of, t_of):
    rows = []
    for pid in IDS:
        m = pid_of == pid
        npos = int(y[m].sum())
        if npos < MIN_POS or len(set(y[m])) < 2:
            print("  %s: skipped, %d positive windows" % (pid, npos))
            continue
        r = evaluate(X[m], y[m], t_of[m])
        r.update(patient=pid, n=int(m.sum()), n_pos=npos)
        rows.append(r)
        print("  %s: random %.3f  temporal %.3f  (%d windows, %d positive)"
              % (pid, r["random"], r["temporal"], r["n"], npos))
    return pd.DataFrame(rows)


def make_figure(df, out_dir, simple=False):
    fig = plt.figure(figsize=(5.0, 3.6))
    ax = fig.add_subplot(111)
    ax.axhline(0.5, color=C_GUIDE, linewidth=1.0, zorder=0)

    # Uniform markers. An earlier version scaled marker area with the infant's
    # event count, so that the density-tracking of the collapse was visible in
    # the figure itself; it needed a legend to be read, and the r below says
    # the same thing in one number.
    for r in df.itertuples():
        ax.plot([0, 1], [r.random, r.temporal], "-", color=C_LINE,
                linewidth=0.9, zorder=1)
    ax.plot(np.zeros(len(df)), df["random"], "o", color=C_RANDOM,
            markersize=4, markeredgewidth=0, zorder=2)
    ax.plot(np.ones(len(df)), df["temporal"], "o", color=C_TEMPORAL,
            markersize=4, markeredgewidth=0, zorder=2)

    mr, mt = df["random"].mean(), df["temporal"].mean()
    ax.plot([0, 1], [mr, mt], "-", color="#3f3f3c", linewidth=2.0, zorder=3)
    ax.plot([0, 1], [mr, mt], "o", color="#3f3f3c", markersize=5,
            markeredgewidth=0, zorder=3)
    if not simple:
        ax.text(-0.04, mr, "mean %.3f" % mr, ha="right", va="center",
                fontsize=7.5, color="#3f3f3c")
        ax.text(1.04, mt, "mean %.3f" % mt, ha="left", va="center",
                fontsize=7.5, color="#3f3f3c")
        # The two the section names, which are the two that collapse furthest.
        for pid in ("015", "014"):
            sub = df[df["patient"] == pid]
            if len(sub):
                ax.text(1.04, float(sub["temporal"].iloc[0]), pid, ha="left",
                        va="center", fontsize=7, color="#5a5a55")
        r = float(np.corrcoef(df["n_pos"], df["temporal"] - df["random"])[0, 1])
        ax.text(0.5, 0.03,
                "the more apneas an infant has, the further it falls:  "
                "$r = %+.2f$" % r,
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color="#5a5a55")

    ax.set_xlim(-0.42, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nfive-fold", "Temporal\n60/40 split"])
    ax.set_ylabel("Within-infant AUROC")
    ax.set_ylim(0.25, 0.98)
    if not simple:
        ax.set_title("Each line is one infant", loc="left")

    os.makedirs(out_dir, exist_ok=True)
    stem = "leakage_paired" + ("_simple" if simple else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (stem, ext)),
                    bbox_inches="tight")
    plt.close(fig)
    print("figure -> %s/%s.pdf" % (out_dir, stem))


def main():
    args = parse_args()
    if args.replot and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        X, y, pid_of, t_of = z["X"], z["y"], z["pid"], z["t"]
        print("features from cache: %d windows" % len(y))
    else:
        print("extracting windows (reads every EDF)")
        X, y, pid_of, t_of = extract(args.data)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, X=X, y=y, pid=pid_of, t=t_of)
        print("cached -> %s" % CACHE)

    print("\n%d windows, %d positive, %d features"
          % (len(y), int(y.sum()), X.shape[1]))
    df = run(X, y, pid_of, t_of)
    df.to_csv(PER_PAT, index=False)

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out()
    out("WITHIN-INFANT PERSONALISED MODELS, %d infants with at least %d "
        "positive windows" % (len(df), MIN_POS))
    out("Identical windows, features and classifier; only the split differs.")
    out()
    out("  random five-fold, pooled out-of-fold   mean %.3f"
        % df["random"].mean())
    out("  random five-fold, mean over folds      mean %.3f"
        % df["random_foldmean"].mean())
    out("  temporal 60/40                         mean %.3f"
        % df["temporal"].mean())
    out("  gap                                    %.3f"
        % (df["random"].mean() - df["temporal"].mean()))
    out()
    out("  infants worse under the temporal split: %d of %d"
        % (int((df["temporal"] < df["random"]).sum()), len(df)))
    out()
    out("PER INFANT")
    out("  id    n   pos   random  temporal    change")
    for r in df.sort_values("random", ascending=False).itertuples():
        out("  %s  %4d  %4d    %.3f     %.3f    %+.3f"
            % (r.patient, r.n, r.n_pos, r.random, r.temporal,
               r.temporal - r.random))
    out()
    out("Reported in Section 4.4 of the thesis SS8.1: random 0.671, "
        "temporal 0.551,")
    out("015 0.82 -> 0.39 and 014 0.77 -> 0.48. Compare the table above.")

    make_figure(df, args.out_dir, simple=args.simple)

    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("written -> %s" % path)


if __name__ == "__main__":
    main()
