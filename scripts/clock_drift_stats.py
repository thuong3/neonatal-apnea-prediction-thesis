"""Summarise the clock-drift experiment: corrected vs uncorrected, paired.

Reads the `<tag>_results.pkl` files written by src/train_neonatal.py and reports
the drift-corrected and uncorrected arms side by side at each lag, averaged over
seeds, with a paired test over patients.

Why paired, and why over PATIENTS: the two arms see the same recordings, the
same splits and the same seeds, so the only thing that differs is where the
labels sit. Pairing removes the between-patient variance, which in this cohort
is far larger than the effect being looked for (per-patient AuROC ranges roughly
0.4-0.7 at a 15 s horizon). An unpaired comparison of two ~0.55 numbers would
have no power at all. Wilcoxon signed-rank rather than a t-test because 15
patients is too few to lean on normality, and it is what Vetter et al. use.

Usage:
    python scripts/clock_drift_stats.py --result_path results
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.misc_utils import read_pickle_file  # noqa: E402

ARMS = {"cpap_drift_corrected": "corrected", "cpap_drift_uncorrected": "uncorrected"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result_path", required=True)
    p.add_argument("--lags", nargs="*", type=int, default=[0, 3000, 6000, 12000])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--reference_experiment", default="robin_reference")
    p.add_argument("--out_csv", default=None)
    p.add_argument(
        "--pair", nargs=4, metavar=("CORR_EXP", "CORR_TAG", "UNCORR_EXP", "UNCORR_TAG"),
        help="compare two specific runs instead of the sweep, e.g. "
             "--pair cpap_drift_corrected lag_3000 cpap_lagsweep lag_3000. "
             "Use this when the uncorrected arm is an EXISTING run rather than "
             "one produced by run_clock_drift_experiment.sh -- a pre-existing run "
             "is a valid control only if its config matches in everything but the "
             "drift, which this checks.")
    return p.parse_args()


def compare_pair(result_path, corr_exp, corr_tag, unc_exp, unc_tag):
    """Paired corrected-vs-uncorrected comparison of two named runs."""
    a = per_patient_scores(result_path, corr_exp, corr_tag)
    b = per_patient_scores(result_path, unc_exp, unc_tag)
    if a is None or b is None:
        raise SystemExit(
            f"missing results: {corr_exp}/{corr_tag} -> {'ok' if a else 'MISSING'}, "
            f"{unc_exp}/{unc_tag} -> {'ok' if b else 'MISSING'}")

    # A pre-existing run is only a control if it differs in the drift alone.
    # Everything that changes the task or the model has to match, or the
    # comparison is measuring something else.
    ca = read_pickle_file(f"{corr_tag}_config.pkl", os.path.join(result_path, corr_exp))
    cb = read_pickle_file(f"{unc_tag}_config.pkl", os.path.join(result_path, unc_exp))
    mismatch = [k for k in ("lag", "time_window", "away", "signal_types",
                            "adverse_events", "cutter_events", "ids", "dataset_mode")
                if ca.dataset.get(k) != cb.dataset.get(k)]
    if ca.meta.seed != cb.meta.seed:
        mismatch.append("meta.seed")
    if mismatch:
        print(f"WARNING: the two runs differ in {mismatch}, not only in the drift "
              "correction -- this is NOT a clean control.\n")
    else:
        print("configs match in everything but the drift correction "
              f"(seed {ca.meta.seed}, lag {ca.dataset.lag}).\n")

    pats = sorted(set(a) & set(b))
    rows = [dict(patient=p, corrected=a[p][0], uncorrected=b[p][0],
                 delta=a[p][0] - b[p][0]) for p in pats]
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))

    d = df["delta"]
    try:
        _, pval = wilcoxon(df["corrected"], df["uncorrected"])
        ptxt = f"p = {pval:.3f}"
    except ValueError:
        ptxt = "p = n/a"
    print(f"\nmean AuROC  corrected {df['corrected'].mean():.4f}   "
          f"uncorrected {df['uncorrected'].mean():.4f}   "
          f"delta {d.mean():+.4f}")
    print(f"corrected better in {int((d > 0).sum())}/{len(d)} patients, "
          f"Wilcoxon signed-rank {ptxt}")
    print("\nNote: a null result here is the expected outcome, not a failure. "
          "Detection\nalready scored ~0.87 WITH the misalignment, so a <=10 s "
          "error inside a 30 s\nwindow is survivable. The value of the "
          "correction is that the figures now\nmatch RemLogic and the "
          "label-noise objection to the ~0.55 is closed.")
    return df


def per_patient_scores(result_path, experiment, tag):
    """{patient: (auroc, ap)} for one run, or None if it was not produced."""
    path = os.path.join(result_path, experiment)
    try:
        res = read_pickle_file(f"{tag}_results.pkl", path)
    except (FileNotFoundError, OSError):
        return None
    out = {}
    for pat_id, d in res.items():
        ys, scores = np.asarray(d["ys"]), np.asarray(d["scores"])
        # A fold with one class present cannot be scored; skip rather than let
        # roc_auc_score raise and lose the whole run.
        if len(np.unique(ys)) < 2:
            continue
        out[pat_id] = (roc_auc_score(ys, scores), average_precision_score(ys, scores))
    return out


def main():
    args = parse_args()
    if args.pair:
        df = compare_pair(args.result_path, *args.pair)
        if args.out_csv:
            df.to_csv(args.out_csv, index=False)
        return

    rows = []
    for lag in args.lags:
        for seed in args.seeds:
            for experiment, arm in ARMS.items():
                got = per_patient_scores(
                    args.result_path, experiment, f"lag_{lag}_seed_{seed}")
                if got is None:
                    continue
                for pat_id, (auc, ap) in got.items():
                    rows.append(dict(lag=lag, seed=seed, arm=arm,
                                     patient=pat_id, auroc=auc, ap=ap))
    if not rows:
        raise SystemExit(
            f"no results found under {args.result_path} for "
            f"{list(ARMS)} -- run scripts/run_clock_drift_experiment.sh first")
    df = pd.DataFrame(rows)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    pd.set_option("display.width", 200)
    print("mean test AuROC over patients (and over seeds):\n")
    piv = df.pivot_table(index="lag", columns="arm", values="auroc", aggfunc="mean")
    piv["horizon_s"] = piv.index / 200.0
    print(piv.round(4).to_string())

    print("\npaired over patients (seeds averaged first, so each patient "
          "contributes once per lag):\n")
    for lag in sorted(df["lag"].unique()):
        sub = df[df["lag"] == lag]
        wide = (sub.groupby(["patient", "arm"])["auroc"].mean()
                .unstack("arm").dropna())
        if not {"corrected", "uncorrected"} <= set(wide.columns) or len(wide) < 5:
            print(f"  lag {lag:6d} ({lag / 200:5.1f} s): "
                  "both arms not available, skipped")
            continue
        d = wide["corrected"] - wide["uncorrected"]
        try:
            stat, p = wilcoxon(wide["corrected"], wide["uncorrected"])
            ptxt = f"p={p:.3f}"
        except ValueError:      # all differences identical/zero
            ptxt = "p=n/a"
        n_better = int((d > 0).sum())
        print(f"  lag {lag:6d} ({lag / 200:5.1f} s): "
              f"corrected {wide['corrected'].mean():.3f}  "
              f"uncorrected {wide['uncorrected'].mean():.3f}  "
              f"delta {d.mean():+.3f}  "
              f"better in {n_better}/{len(d)} patients  {ptxt}")

    ref = per_patient_scores(
        args.result_path, args.reference_experiment, f"lag_3000_seed_{args.seeds[0]}")
    print("\npipeline reference (Robin-sequence cohort, the paper's own data):")
    if ref:
        aucs = np.array([v[0] for v in ref.values()])
        verdict = ("consistent with the published ~0.80 -> the pipeline is sound, "
                   "so a low CPAP number is a property of the CPAP data"
                   if aucs.mean() > 0.72 else
                   "BELOW the published ~0.80 -> suspect the pipeline before "
                   "interpreting any CPAP number")
        print(f"  mean test AuROC {aucs.mean():.3f} over {len(aucs)} patients")
        print(f"  {verdict}")
    else:
        print("  NOT RUN. Without it a CPAP AuROC near 0.5 cannot be told apart")
        print("  from a pipeline bug -- set ROBIN_DATA and re-run the script.")


if __name__ == "__main__":
    main()
