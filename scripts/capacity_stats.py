"""The capacity contrast, under the pre-specified rule.

Answers one question: given 40 hidden units and 40 epochs instead of the
published 20 and 10, does the model fit the CPAP training windows, and does any
of it transfer to a held-out infant?

F2's nested run (Section 4.3 of the thesis) said train 0.871 and
test 0.424, which would be the strongest sentence in the results chapter -- but
it came from 3 outer folds that SELECTION chose, against 8 for the published
config. Different infants, no pairing, so no signed-rank test is defined and the
comparison is exactly the two-averages-by-eye that PRE_SPECIFICATION.md 1.1
forbids everywhere else in this study. This script analyses the fixed-arm rerun
that fixes that.

The statistics are NOT reimplemented here. `report_contrast` is imported from
`norm_ablation_stats.py`, so this contrast is judged by the same two-part rule
as the normalisation 2x2 and the closing sweeps: a Wilcoxon over the paired
per-infant differences AND a difference larger than the seed-to-seed spread of
that same contrast.

TRAIN AuROC is reported beside test, which the B3 analysis does not need to do.
It is the whole point here: the claim is that the capacity arm FITS (train near
Robin's 0.859) and does not TRANSFER (test at or below chance). A test-side null
alone would be ambiguous -- it is the pair that carries the argument.

Run:
    python scripts/capacity_stats.py --seeds 0 1 2
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from norm_ablation_stats import cw, per_patient, report_contrast  # noqa: E402
from src.misc_utils import read_pickle_file  # noqa: E402

ARMS = ("base", "wide_long")
DESC = {"base": "20 hidden, 10 epochs (published)",
        "wide_long": "40 hidden, 40 epochs (capacity arm)"}

# For context in the printout only; none of it enters the test.
REFERENCE = [
    ("published config on CPAP, section 22", 0.655, 0.555),
    ("published config on ROBIN, section 22", 0.859, 0.789),
    ("detection on CPAP, section 7", 0.880, 0.870),
    ("F2 nested, 3 selected folds, section 22.1", 0.871, 0.424),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_path", default="results")
    p.add_argument("--experiment", default="cpap_capacity")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--logs", default=None,
                   help="Hydra outputs dir (e.g. outputs/), scanned for the "
                        "per-fold train AuROC the pickles do not carry")
    return p.parse_args()


def scan_train_aucs(logs_dir, seeds):
    """{arm: {seed: [per-fold train AuROC]}} from the Hydra training logs.

    Each run's log opens with a dump of its own resolved config, so the arm and
    seed are read from the log itself rather than from the file's path -- Hydra
    names run directories by timestamp, which says nothing about which arm they
    held, and guessing from mtime would silently mislabel a rerun.
    """
    import re

    tag_re = re.compile(r"^\s*tag:\s*(\S+)\s*$", re.M)
    auc_re = re.compile(r"final train_auc=([0-9.]+)")
    out = {}
    for root, _dirs, files in os.walk(logs_dir):
        for name in files:
            if not name.endswith(".log"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            m = tag_re.search(text)
            if not m:
                continue
            tag = m.group(1).strip("'\"")
            for arm in ARMS:
                for seed in seeds:
                    if tag == f"{arm}_s{seed}":
                        vals = [float(v) for v in auc_re.findall(text)]
                        if vals:
                            out.setdefault(arm, {})[seed] = vals
    return out


def main():
    args = parse_args()
    exp_dir = os.path.join(args.result_path, args.experiment)

    runs, missing = {a: {} for a in ARMS}, []
    for arm in ARMS:
        for seed in args.seeds:
            tag = f"{arm}_s{seed}"
            if not os.path.exists(os.path.join(exp_dir, f"{tag}_results.pkl")):
                missing.append(tag)
                continue
            runs[arm][seed] = read_pickle_file(f"{tag}_results.pkl", exp_dir)
    if missing:
        print("missing runs: " + ", ".join(missing))
    if not all(runs[a] for a in ARMS):
        raise SystemExit("both arms need at least one seed")

    ids = sorted(next(iter(runs["base"].values())))

    # The assumption that would silently invalidate everything below. Capacity
    # changes the network, never the window table, so identical labels are
    # expected -- asserted rather than assumed.
    ref = next(iter(runs["base"].values()))
    for arm in ARMS:
        for seed, res in runs[arm].items():
            assert sorted(res) == ids, f"{arm}_s{seed}: patient set differs"
            for pid in ids:
                assert np.array_equal(res[pid]["ys"], ref[pid]["ys"]), (
                    f"{arm}_s{seed} patient {pid}: labels differ -- "
                    "the arms are NOT paired")
    print(f"pairing verified: both arms see identical windows and labels "
          f"({len(ids)} infants)")

    w = np.array([np.sum(ref[p]["ys"]) for p in ids], dtype=float)
    seed_auc = {a: {s: per_patient(r, ids) for s, r in runs[a].items()}
                for a in ARMS}

    print()
    print("=" * 74)
    print("[1] BOTH ARMS")
    print(f"  {'arm':<11}{'seeds':>6}{'count-weighted':>16}{'plain mean':>12}")
    for arm in ARMS:
        mean_pp = np.mean([seed_auc[arm][s] for s in sorted(seed_auc[arm])], axis=0)
        print(f"  {arm:<11}{len(seed_auc[arm]):>6}{cw(mean_pp, w):>16.4f}"
              f"{float(np.mean(mean_pp)):>12.4f}   {DESC[arm]}")

    print()
    print("=" * 74)
    print("[2] THE PAIRED CONTRAST  (pre-spec 1.1)")
    report_contrast("wide_long", "base",
                    "does the capacity arm transfer better than the published one?",
                    seed_auc, ids, w, args.alpha, f"all {len(ids)} infants")

    print()
    print("=" * 74)
    print("[3] PER-INFANT, seed-averaged")
    mb = np.mean([seed_auc["base"][s] for s in sorted(seed_auc["base"])], axis=0)
    mw = np.mean([seed_auc["wide_long"][s] for s in sorted(seed_auc["wide_long"])],
                 axis=0)
    print(f"  {'infant':<8}{'base':>9}{'wide_long':>11}{'delta':>9}{'positives':>11}")
    for pid, b, x, n in zip(ids, mb, mw, w):
        print(f"  {pid:<8}{b:>9.3f}{x:>11.3f}{x - b:>+9.3f}{int(n):>11}")

    print()
    print("=" * 74)
    print("[4] TRAIN AuROC  -- the other half of the claim")
    # `nested_leave_one_out` stores only ys and scores, so the train side lives
    # in the Hydra logs. Without it this script cannot distinguish "does not
    # fit" from "fits and does not transfer", which is the entire question.
    train = scan_train_aucs(args.logs, args.seeds) if args.logs else {}
    if not train:
        print("  not read. Pass --logs <hydra outputs dir> to extract it, or:")
        print("    grep -h 'final train_auc' <outputs>/*/*/train_neonatal.log")
    else:
        for arm in ARMS:
            vals = [v for s in sorted(train.get(arm, {}))
                    for v in train[arm][s]]
            if vals:
                print(f"  {arm:<11}mean train AuROC {np.mean(vals):.4f}  "
                      f"({len(train[arm])} seed(s), {len(vals)} folds)")
        if train.get("base") and train.get("wide_long"):
            b = np.mean([v for s in train["base"] for v in train["base"][s]])
            x = np.mean([v for s in train["wide_long"] for v in train["wide_long"][s]])
            print(f"  capacity arm fits {x - b:+.4f} better on the training data")

    print()
    print("=" * 74)
    print("[5] CONTEXT -- reference points, none of which enter the test")
    print(f"  {'':<44}{'train':>8}{'test':>8}")
    for label, tr, te in REFERENCE:
        print(f"  {label:<44}{tr:>8.3f}{te:>8.3f}")
    print()
    print("  The claim needs BOTH halves: fits like Robin (train ~0.86) AND")
    print("  transfers at or below chance. Either alone is ambiguous.")


if __name__ == "__main__":
    main()
