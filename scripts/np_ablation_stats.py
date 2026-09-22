"""Paired statistics for the NP ablation (model with vs. without the
nasal-pressure channel), from the `*_results.pkl` written by
src/train_neonatal.py.

Follows the evaluation of Vetter et al. (2024): if several seeds are given,
the per-patient AuROC is first averaged over the training repetitions, then a
Wilcoxon signed-rank test is computed over the n patients. With a single seed
the test is run on that seed's per-patient AuROCs directly (and the caveat is
printed).

Usage:
    # single run (tags "with_np" / "without_np")
    python scripts/np_ablation_stats.py --result_path <meta.result_path> \
        --experiment robin_np_ablation

    # seed sweep (tags "with_np_s0", "without_np_s0", ...)
    python scripts/np_ablation_stats.py --result_path <meta.result_path> \
        --experiment robin_np_ablation_seeds --seeds 0 1 2 3 4
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.misc_utils import read_pickle_file  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_path", required=True)
    p.add_argument("--experiment", default="robin_np_ablation")
    p.add_argument(
        "--seeds",
        nargs="*",
        default=[],
        help="seeds of a sweep; empty means a single run with the bare tags",
    )
    p.add_argument("--with_tag", default="with_np", help="tag of the full-channel run")
    p.add_argument("--without_tag", default="without_np", help="tag of the ablated run")
    p.add_argument("--n_boot", type=int, default=20000)
    return p.parse_args()


def load_run(exp_dir, tag):
    """Per-patient {ys, scores} of one run, or None if it is missing."""
    path = os.path.join(exp_dir, f"{tag}_results.pkl")
    if not os.path.exists(path):
        print(f"  ! missing: {path}")
        return None
    return read_pickle_file(f"{tag}_results.pkl", exp_dir)


def per_patient_auc(results, ids):
    return np.array([roc_auc_score(results[p]["ys"], results[p]["scores"]) for p in ids])


def pooled_auc(results, ids):
    ys = np.concatenate([results[p]["ys"] for p in ids])
    scores = np.concatenate([results[p]["scores"] for p in ids])
    return roc_auc_score(ys, scores)


def main():
    args = parse_args()
    exp_dir = os.path.join(args.result_path, args.experiment)
    tags = (
        [(args.with_tag, args.without_tag)]
        if not args.seeds
        else [(f"{args.with_tag}_s{s}", f"{args.without_tag}_s{s}") for s in args.seeds]
    )

    ids, n_windows = None, {}
    auc_w_runs, auc_o_runs, pooled_w, pooled_o = [], [], [], []
    for with_tag, without_tag in tags:
        run_w, run_o = load_run(exp_dir, with_tag), load_run(exp_dir, without_tag)
        if run_w is None or run_o is None:
            continue
        if ids is None:
            ids = sorted(run_w)
            n_windows = {pid: len(run_w[pid]["ys"]) for pid in ids}
        assert sorted(run_w) == ids and sorted(run_o) == ids, "patient sets differ"
        # The window table does not depend on signal_types, so both variants
        # must see identical windows and labels -- verify, otherwise the
        # comparison is not paired.
        for pid in ids:
            assert np.array_equal(run_w[pid]["ys"], run_o[pid]["ys"]), (
                f"{pid}: labels differ between {with_tag} and {without_tag}"
            )
        auc_w_runs.append(per_patient_auc(run_w, ids))
        auc_o_runs.append(per_patient_auc(run_o, ids))
        pooled_w.append(pooled_auc(run_w, ids))
        pooled_o.append(pooled_auc(run_o, ids))
        print(f"  loaded {with_tag} / {without_tag}: "
              f"pooled {pooled_w[-1]:.3f} vs {pooled_o[-1]:.3f}")

    if not auc_w_runs:
        sys.exit(f"No result pairs found in {exp_dir}")

    n_runs = len(auc_w_runs)
    # Average over training repetitions first, then test over patients.
    auc_w = np.mean(auc_w_runs, axis=0)
    auc_o = np.mean(auc_o_runs, axis=0)
    delta = auc_o - auc_w

    print(f"\nPer-patient AuROC (mean over {n_runs} run(s))")
    print(f"{'pat':>5} {'n':>5} {'withNP':>8} {'noNP':>8} {'delta':>8}")
    for pid, a, b in zip(ids, auc_w, auc_o):
        print(f"{pid:>5} {n_windows[pid]:>5} {a:>8.3f} {b:>8.3f} {b - a:>+8.3f}")

    print(
        f"\nmean per-patient AuROC:  with NP {auc_w.mean():.3f} (SD {auc_w.std(ddof=1):.3f})"
        f" | without NP {auc_o.mean():.3f} (SD {auc_o.std(ddof=1):.3f})"
        f" | delta {delta.mean():+.3f}"
    )
    print(
        f"pooled AuROC:            with NP {np.mean(pooled_w):.3f}"
        f" | without NP {np.mean(pooled_o):.3f}"
        f" | delta {np.mean(pooled_o) - np.mean(pooled_w):+.3f}"
    )
    if n_runs > 1:
        print(
            f"  seed-to-seed SD of pooled AuROC: with NP {np.std(pooled_w, ddof=1):.3f}"
            f" | without NP {np.std(pooled_o, ddof=1):.3f}"
        )

    stat, p = wilcoxon(auc_w, auc_o)
    print(f"\nWilcoxon signed-rank over n={len(ids)} patients: W={stat:.1f}, p={p:.3f}")
    print(f"  worse without NP: {(auc_o < auc_w).sum()}/{len(ids)} patients")

    rng = np.random.default_rng(0)
    boot = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean()
         for _ in range(args.n_boot)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  delta mean AuROC = {delta.mean():+.3f}, "
          f"95% bootstrap CI over patients [{lo:+.3f}, {hi:+.3f}]")

    if n_runs == 1:
        print(
            "\nNOTE: only one training run per variant. Vetter et al. average 10 "
            "repetitions per infant before the Wilcoxon test; a single seed cannot "
            "separate the ablation effect from seed variance. Use "
            "scripts/run_np_ablation_seeds.sh for a sweep."
        )


if __name__ == "__main__":
    main()
