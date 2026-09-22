"""Summarise domain-adaptation runs across seeds, with a paired test.

Reads every `*_da_results.pkl` written by src/train_domain_adaptation.py in a
result folder, averages each patient's AuROC over seeds, and compares every
ladder variant against the `target_only` control with a Wilcoxon signed-rank
test over patients -- the same procedure used for the NP ablation
(scripts/np_ablation_stats.py), so the two are directly comparable.

Averaging over seeds BEFORE the test is deliberate: the seed-to-seed SD on this
data (~0.007-0.012) is the same order as the differences at stake, so a single
seed cannot resolve them.

Usage:
    python scripts/da_summary.py --result_path /path/to/results
"""

import argparse
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

CONTROL = "target_only"


def safe_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def load_runs(result_path):
    paths = sorted(glob.glob(os.path.join(result_path, "**", "*_da_results.pkl"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"No *_da_results.pkl found under {result_path}")
    runs = []
    for path in paths:
        with open(path, "rb") as fh:
            runs.append((os.path.basename(path), pickle.load(fh)))
    return runs


def run_group(run):
    """The run's tag with its seed suffix stripped: diag_5ch_s0 -> diag_5ch.

    Runs that differ in anything but the seed (5ch vs 6ch diagnostics, CORAL
    lambdas) must not be averaged together, so the group name keeps them apart.
    """
    tag = run.get("config", {}).get("meta", {}).get("tag") or "run"
    return re.sub(r"_s\d+$", "", tag)


def collect(runs, key):
    """{name: {patient: [auc per seed]}} for 'results' or 'source_internal'."""
    per_variant = defaultdict(lambda: defaultdict(list))
    groups = {run_group(run) for _n, run in runs if run.get(key)}
    for _name, run in runs:
        blob = run.get(key)
        if not blob:
            continue
        group = run_group(run)
        # source_internal is a bare {patient: {...}}; results is nested by variant
        is_diagnostic = key == "source_internal"
        if is_diagnostic:
            blob = {group: blob}
        for variant, per_patient in blob.items():
            if is_diagnostic:
                # The name is already the group (diag_5ch / diag_6ch).
                name = variant
            else:
                # Only qualify the ladder's variant names when more than one
                # group is present, so the common case stays readable.
                name = variant if len(groups) == 1 else f"{group}/{variant}"
            for pat_id, d in per_patient.items():
                per_variant[name][pat_id].append(safe_auc(d["ys"], d["scores"]))
    return per_variant


def seed_averaged(per_variant):
    """{variant: {patient: mean auc over seeds}}"""
    return {
        variant: {p: float(np.nanmean(a)) for p, a in per_patient.items()}
        for variant, per_patient in per_variant.items()
    }


def report(title, per_variant, control=None):
    if not per_variant:
        return
    averaged = seed_averaged(per_variant)
    n_seeds = max(
        len(a) for per_patient in per_variant.values() for a in per_patient.values()
    )
    print(f"\n{title}  ({n_seeds} seed(s))")
    header = f"{'variant':<26}{'mean AUC':>10}{'SD':>8}{'n_pat':>7}"
    if control:
        header += f"{'vs ' + control:>17}{'p':>9}"
    print(header)
    print("-" * len(header))

    def control_for(name):
        """The target_only row from the SAME group as `name`."""
        if not control:
            return None
        key = f"{name.split('/')[0]}/{control}" if "/" in name else control
        return averaged.get(key)

    for variant, per_patient in averaged.items():
        control_scores = control_for(variant)
        vals = np.array(list(per_patient.values()), dtype=float)
        line = (
            f"{variant:<26}{np.nanmean(vals):>10.3f}{np.nanstd(vals):>8.3f}"
            f"{len(vals):>7}"
        )
        # `is not` rather than a name comparison: with group-qualified names the
        # control row is "da/target_only", not "target_only", and comparing it
        # against itself would print a meaningless 0.000 / n/a.
        if control_scores is not None and control_scores is not per_patient:
            # Pair strictly by patient id -- a positional zip would silently
            # mispair if any patient was dropped for being single-class.
            shared = sorted(set(per_patient) & set(control_scores))
            a = np.array([per_patient[p] for p in shared])
            b = np.array([control_scores[p] for p in shared])
            delta = float(np.nanmean(a - b))
            try:
                _, pval = wilcoxon(a, b)
                line += f"{delta:>+17.3f}{pval:>9.3f}"
            except ValueError:  # all differences zero
                line += f"{delta:>+17.3f}{'n/a':>9}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    runs = load_runs(args.result_path)
    print(f"Loaded {len(runs)} run(s) from {args.result_path}")

    # Two separate tables: the diagnostic is scored on SOURCE patients and the
    # ladder on TARGET patients, so their rows are not comparable.
    report(
        "=== DIAGNOSTIC (source cohort, leave-one-patient-out) ===",
        collect(runs, "source_internal"),
    )
    report(
        "=== LADDER (target cohort, leave-one-patient-out) ===",
        collect(runs, "results"),
        control=CONTROL,
    )


if __name__ == "__main__":
    main()
