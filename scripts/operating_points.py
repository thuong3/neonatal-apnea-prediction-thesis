"""G2: PREVALENCE-APPROPRIATE METRICS AND A BEDSIDE OPERATING POINT.

Item G2 asks for three things that AuROC alone does not give:

  * AuPRC, because positives are ~8.6% here against 36% in Vetter et al., and
    AuROC is insensitive to that while precision is not;
  * TPR and precision at fixed FPR (10/20/30%), as in Vetter Table 1, with the
    explicit warning that precision at a given FPR is far lower here PURELY
    because of the prevalence change, so the two studies' tables are not
    comparable;
  * and the one that actually decides whether any of this is usable at a
    bedside: SENSITIVITY AT A FIXED FALSE-ALARM RATE PER HOUR.

HOW ALARMS PER HOUR ARE DERIVED, AND THE ASSUMPTION IN IT
    The model scores non-overlapping 30 s windows, so a continuously monitored
    hour would contain 120 of them and

        false alarms per hour = FPR x 120.

    This assumes the model runs on every consecutive window. The evaluation set
    does NOT tile the recording that way -- controls are drawn at least 3 min
    from any event -- so this is an extrapolation from the measured FPR to
    continuous operation, not a direct count. It is the right quantity for the
    clinical question and the wrong one to quote as if it had been observed.
    Stated here rather than buried, because the number is otherwise easy to
    over-read.

WHY A PLAIN MEAN OVER INFANTS IS NOT REPORTED ALONE
    Target-window counts run 22-306 across the 15 infants (section 28.2), and
    per-patient scores in this cohort are strongly anti-correlated with that
    count (section 28.4). Every summary below is therefore given both plain and
    count-weighted, and the count-weighted one is the honest headline.

Usage:
    python scripts/operating_points.py --results <path to *_results.pkl>
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

WINDOWS_PER_HOUR = 120.0  # 30 s non-overlapping windows
DEFAULT_RESULTS = ("data/"
                   "dataset_brainimmaturity/results/cpap_lagsweep/lag_3000_results.pkl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--fpr", nargs="*", type=float, default=[0.10, 0.20, 0.30])
    p.add_argument("--alarms-per-hour", nargs="*", type=float,
                   default=[0.5, 1.0, 2.0, 5.0])
    return p.parse_args()


def tpr_at_fpr(y, s, target_fpr):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def precision_at_fpr(y, s, target_fpr):
    """Precision at the threshold that yields `target_fpr`."""
    fpr, _, thr = roc_curve(y, s)
    # Take the first threshold whose FPR reaches the target rather than
    # interpolating between thresholds: interpolated scores do not correspond
    # to any achievable operating point, so the precision at one is not a
    # precision the model can actually be run at.
    idx = int(np.searchsorted(fpr, target_fpr, side="left"))
    idx = min(max(idx, 0), len(thr) - 1)
    t = thr[idx]
    pred = s >= t
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def main():
    args = parse_args()
    if not os.path.exists(args.results):
        raise SystemExit(f"no results file at {args.results}")
    res = pd.read_pickle(args.results)

    print(f"results: {args.results}")
    print(f"patients: {len(res)}\n")

    rows = []
    for pat_id in sorted(res):
        y = np.asarray(res[pat_id]["ys"]).ravel().astype(int)
        s = np.asarray(res[pat_id]["scores"]).ravel().astype(float)
        n_pos = int(y.sum())
        if n_pos < 3 or (len(y) - n_pos) < 3:
            print(f"{pat_id}: {n_pos} positives -- skipped")
            continue
        row = {
            "pat": pat_id, "n_pos": n_pos, "n": len(y),
            "prev": n_pos / len(y),
            "auroc": roc_auc_score(y, s),
            "auprc": average_precision_score(y, s),
        }
        for f in args.fpr:
            row[f"tpr@fpr{int(f * 100)}"] = tpr_at_fpr(y, s, f)
            row[f"prec@fpr{int(f * 100)}"] = precision_at_fpr(y, s, f)
        for a in args.alarms_per_hour:
            row[f"sens@{a}/h"] = tpr_at_fpr(y, s, a / WINDOWS_PER_HOUR)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable patients")

    w = df["n_pos"].to_numpy(dtype=float)

    def summ(col):
        v = df[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if not ok.any():
            return float("nan"), float("nan")
        return float(v[ok].mean()), float(np.sum(v[ok] * w[ok]) / w[ok].sum())

    print(f"{'metric':<18} {'plain mean':>11} {'count-weighted':>15}")
    print("-" * 46)
    for col in ["auroc", "auprc"]:
        a, b = summ(col)
        print(f"{col:<18} {a:>11.3f} {b:>15.3f}")
    print(f"{'prevalence':<18} {df['prev'].mean():>11.3f} "
          f"{np.sum(df['prev'] * w) / w.sum():>15.3f}")
    print()
    print("Vetter Table 1 style -- TPR and precision at fixed FPR:")
    for f in args.fpr:
        a, b = summ(f"tpr@fpr{int(f * 100)}")
        pa, pb = summ(f"prec@fpr{int(f * 100)}")
        print(f"  FPR {int(f * 100):>2}%:  TPR {a:.3f} (wtd {b:.3f})   "
              f"precision {pa:.3f} (wtd {pb:.3f})")
    print()
    print("  Precision here is not comparable with Vetter Table 1: positives are")
    print("  ~9% of windows in this cohort against 36% there, and precision at a")
    print("  fixed FPR falls with prevalence regardless of the model.")
    print()
    print("THE CLINICAL OPERATING POINT -- sensitivity at a fixed alarm burden")
    print("  (extrapolated from FPR assuming continuous 30 s windowing)")
    for a in args.alarms_per_hour:
        m, wt = summ(f"sens@{a}/h")
        print(f"  {a:>4} false alarms/h (FPR {a / WINDOWS_PER_HOUR * 100:.2f}%):  "
              f"sensitivity {m:.3f} (count-weighted {wt:.3f})")
    print()
    print("  A monitor that wakes the nurse once an hour and catches this")
    print("  fraction of apneas is the honest way to state the result. Compare")
    print("  it against the alternative of not predicting at all.")
    print()
    print("Per patient:")
    cols = ["pat", "n_pos", "prev", "auroc", "auprc"] + \
           [f"sens@{a}/h" for a in args.alarms_per_hour]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
