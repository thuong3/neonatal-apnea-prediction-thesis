"""PREVALENCE-APPROPRIATE METRICS FOR THE SpO2 BASELINE ENDPOINT.

`scripts/operating_points.py` does this for the 15 s apnea task. The oxygenation
endpoint of `scripts/spo2_baseline_prediction.py` reaches a leave-one-patient-out
AuROC of 0.740 at a 60 s lead, and AuROC alone says nothing about how a monitor
built on it would behave, because the endpoint occurs in 3.8 % of intervals and
AuROC is insensitive to prevalence. This script reports, for the same model,
the same features and the same folds:

  * AuPRC against the positive rate, which is what an uninformative classifier
    reaches;
  * sensitivity at a fixed false-alarm burden per hour, which is the number
    that decides whether the endpoint is usable at a bedside.

WHICH RUN THIS REPRODUCES
    The features come from the cache that `spo2_baseline_prediction.py` writes,
    and the classifier, the class weights and the leave-one-patient-out split
    are imported from that module rather than restated here, so the AuROC
    printed below is the AuROC of the sweep and not a near-miss of it.
    Defaults reproduce the headline cell: endpoint `declabs` (mean SpO2 below
    90 % over the target interval), horizon 60 s, lead 60 s, all feature blocks.

HOW ALARMS PER HOUR ARE DERIVED, AND THE ASSUMPTION IN IT
    Features are cut on a 60 s window with a 30 s stride, so a monitor scoring
    every stride emits 120 decisions per hour and

        false alarms per hour = FPR x decisions per hour.

    Two conventions are printed. At 120/h the monitor re-decides every 30 s,
    which matches the evaluation grid but means consecutive decisions cover
    overlapping target intervals. At 60/h it decides once per non-overlapping
    target interval, which is the more conservative reading. Neither is an
    observed alarm count: both extrapolate the measured false-positive rate to
    continuous operation, exactly as in `operating_points.py`.

WHY TWO INFANTS ARE MISSING
    The endpoint never occurs in two of the fifteen recordings, so those folds
    are single-class and are skipped, as they are in the sweep. Every summary
    is reported plain and count-weighted by the number of positive intervals,
    following Section 3.7 of the thesis.

Usage:
    python scripts/spo2_operating_points.py
    python scripts/spo2_operating_points.py --blocks drift --lead 120
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spo2_baseline_prediction as sbp  # noqa: E402

DEFAULT_CACHE = "outputs/spo2_baseline/features_v2.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=DEFAULT_CACHE,
                   help="feature cache written by spo2_baseline_prediction.py")
    p.add_argument("--endpoint", default="declabs")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--lead", type=int, default=60)
    p.add_argument("--blocks", default="all", choices=["all", "drift"],
                   help="'all' is the 0.740 cell, 'drift' the 0.706 one")
    p.add_argument("--alarms-per-hour", nargs="*", type=float,
                   default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--decisions-per-hour", nargs="*", type=float,
                   default=[120.0, 60.0])
    p.add_argument("--out", default="outputs/spo2_baseline/operating_points.txt")
    return p.parse_args()


def tpr_at_fpr(y, s, target_fpr):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def oof_scores(X, y, groups):
    """Leave-one-patient-out scores, one fold per infant.

    The fold rule is the one in `spo2_baseline_prediction.loo`: an infant is
    evaluated only if its own rows carry both classes and the training rows do
    too. Scores are kept per infant rather than pooled, because Section 3.7
    does not permit a pooled headline.
    """
    out = {}
    for pid in sbp.IDS:
        tr, te = groups != pid, groups == pid
        if te.sum() == 0 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        c = sbp.gbm()
        c.fit(X[tr], y[tr], sample_weight=sbp.sample_weights(y[tr]))
        out[pid] = (y[te], c.predict_proba(X[te])[:, 1])
    return out


def main():
    args = parse_args()
    if not os.path.exists(args.cache):
        raise SystemExit(f"no feature cache at {args.cache}; run "
                         f"spo2_baseline_prediction.py first")
    d = np.load(args.cache, allow_pickle=True)

    key = f"y_{args.endpoint}_{args.horizon}_L{args.lead}"
    if key not in d.files:
        raise SystemExit(f"{key} not in {args.cache}")
    y_all = d[key]
    keep = y_all >= 0                      # `decline` marks unusable as -1
    groups = d["groups"][keep]
    y = y_all[keep]

    X = (np.hstack([d["PH"], d["DR"], d["IH"], d["HI"]]) if args.blocks == "all"
         else d["DR"])[keep]

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out(f"cache:    {args.cache}")
    out(f"endpoint: {args.endpoint}  horizon {args.horizon}s  lead {args.lead}s")
    out(f"blocks:   {args.blocks}  ({X.shape[1]} features, {len(y)} intervals)")
    out(f"positive rate overall: {y.mean():.4f}")
    out()

    folds = oof_scores(X, y, groups)
    skipped = [p for p in sbp.IDS if p not in folds]
    if skipped:
        out(f"single-class folds, skipped: {', '.join(skipped)}")
        out()

    rows = []
    for pid, (yt, st) in folds.items():
        row = {"pat": pid, "n_pos": int(yt.sum()), "n": len(yt),
               "prev": float(yt.mean()),
               "auroc": roc_auc_score(yt, st),
               "auprc": average_precision_score(yt, st)}
        for dph in args.decisions_per_hour:
            for a in args.alarms_per_hour:
                row[f"sens@{a}/h@{int(dph)}"] = tpr_at_fpr(yt, st, a / dph)
        rows.append(row)

    df = pd.DataFrame(rows)
    w = df["n_pos"].to_numpy(dtype=float)

    def summ(col):
        v = df[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if not ok.any():
            return float("nan"), float("nan")
        return float(v[ok].mean()), float(np.sum(v[ok] * w[ok]) / w[ok].sum())

    out(f"{'metric':<20} {'plain mean':>11} {'count-weighted':>15}")
    out("-" * 48)
    for col in ["auroc", "auprc"]:
        a, b = summ(col)
        out(f"{col:<20} {a:>11.3f} {b:>15.3f}")
    a, b = summ("prev")
    out(f"{'positive rate':<20} {a:>11.3f} {b:>15.3f}")
    out()
    out("  An uninformative classifier reaches an AuPRC equal to the positive")
    out("  rate, so the two rows above are the comparison that matters.")
    out()

    for dph in args.decisions_per_hour:
        out(f"SENSITIVITY AT A FIXED ALARM BURDEN -- {int(dph)} decisions/h "
            f"({'every 30 s stride' if dph == 120 else 'one per target interval'})")
        for a in args.alarms_per_hour:
            m, wt = summ(f"sens@{a}/h@{int(dph)}")
            out(f"  {a:>4} false alarms/h (FPR {a / dph * 100:.2f}%):  "
                f"sensitivity {m:.3f} (count-weighted {wt:.3f})")
        out()

    out("  Extrapolated from the measured false-positive rate to continuous")
    out("  operation, not an observed alarm count.")
    out()
    out("Per patient:")
    cols = (["pat", "n_pos", "n", "prev", "auroc", "auprc"] +
            [f"sens@{a}/h@120" for a in args.alarms_per_hour])
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        out(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
