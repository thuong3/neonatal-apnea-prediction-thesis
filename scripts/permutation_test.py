"""G3: PER-INFANT PERMUTATION TESTS (Ojala & Garriga).

Item G3 asks for the original inferential machinery: per-infant permutation
tests in the style of Ojala & Garriga (2010), plus Wilcoxon signed-rank over the
infants for model comparisons. The Wilcoxon half is already used in several
scripts (clock_drift_stats.py, da_summary.py, np_ablation_stats.py); the
permutation half was not implemented anywhere. This closes that gap.

WHAT IS TESTED
    For one infant, is the model's ranking of its held-out windows better than
    it would be if the labels carried no information?

    Null: the labels are exchangeable within this infant. The label vector is
    shuffled `n_perm` times, the AuROC recomputed each time, and the p-value is

        p = (1 + #{AuROC_permuted >= AuROC_observed}) / (1 + n_perm)

    The +1 in numerator and denominator is deliberate (Phipson & Smyth 2010):
    it keeps the p-value from ever being exactly 0, which would be a claim the
    permutation count cannot support. The smallest reportable value is therefore
    1/(1 + n_perm).

    This is Ojala & Garriga's first null -- label permutation, which tests
    whether there is ANY label-feature relation. It does not test the stronger
    null in which feature dependencies are also destroyed; that variant needs
    the feature matrix, not the saved scores, and is not what G3 asks for here.

WHY PER INFANT
    Pooling every infant's windows into one permutation test would let a model
    that merely ranks high-event-rate infants above low-event-rate ones look
    significant without predicting anything within any infant. The per-infant
    test cannot be passed that way.

MULTIPLE COMPARISONS
    Fifteen tests are run, so a Benjamini-Hochberg FDR correction is reported
    beside the raw p-values. Vetter et al. report how many infants were
    individually above chance; that count is only meaningful after correction.

Usage:
    python scripts/permutation_test.py
    python scripts/permutation_test.py --results <path> --n-perm 10000
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_RESULTS = ("data/"
                   "dataset_brainimmaturity/results/cpap_lagsweep/lag_3000_results.pkl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def permutation_p(y, s, n_perm, rng):
    """Ojala & Garriga label-permutation p-value for one infant's AuROC."""
    observed = roc_auc_score(y, s)
    y_perm = y.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if roc_auc_score(y_perm, s) >= observed:
            ge += 1
    return observed, (1.0 + ge) / (1.0 + n_perm)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (monotone), same order as the input."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(ranked, 1.0)
    return out


def main():
    args = parse_args()
    if not os.path.exists(args.results):
        raise SystemExit(f"no results file at {args.results}")
    res = pd.read_pickle(args.results)
    rng = np.random.default_rng(args.seed)

    print(f"results : {args.results}")
    print(f"permutations per infant : {args.n_perm}")
    print(f"smallest reportable p   : {1.0 / (1 + args.n_perm):.5f}\n")

    rows = []
    for pat_id in sorted(res):
        y = np.asarray(res[pat_id]["ys"]).ravel().astype(int)
        s = np.asarray(res[pat_id]["scores"]).ravel().astype(float)
        n_pos = int(y.sum())
        if n_pos < 3 or (len(y) - n_pos) < 3:
            print(f"{pat_id}: {n_pos} positives -- skipped")
            continue
        auc, p = permutation_p(y, s, args.n_perm, rng)
        rows.append({"pat": pat_id, "n_pos": n_pos, "auroc": auc, "p": p})
        print(f"{pat_id}: n_pos={n_pos:4d}  AuROC={auc:.3f}  p={p:.4f}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable infants")
    df["p_bh"] = benjamini_hochberg(df["p"].to_numpy())

    sig_raw = int((df["p"] < args.alpha).sum())
    sig_bh = int((df["p_bh"] < args.alpha).sum())
    print("\n" + "=" * 66)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"above chance at alpha={args.alpha}:")
    print(f"  uncorrected            : {sig_raw}/{len(df)} infants")
    print(f"  Benjamini-Hochberg FDR : {sig_bh}/{len(df)} infants")
    print()
    print("READING")
    print("  Vetter et al. report performance significantly above chance for all")
    print("  but one of their 19 infants. The comparable count for this cohort is")
    print("  the FDR-corrected one above. Infants with very few positive windows")
    print("  cannot reach significance whatever the model does -- read the count")
    print("  together with the n_pos column, not on its own.")


if __name__ == "__main__":
    main()
