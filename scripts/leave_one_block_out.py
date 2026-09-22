"""D3: THE WITHIN-PATIENT CEILING -- HOW MUCH OF THE DEFICIT IS SUBJECT SHIFT?

Item D3 asks for leave-one-BLOCK-out within patient: train on three of an
infant's device blocks together with all other infants, test on the held-out
block. Contrast that with full leave-one-patient-out, where the model has never
seen the test infant at all.

    LOPO   AuROC : the model has seen 14 other infants, never this one
    LOBO   AuROC : same, plus three blocks of THIS infant

    gap = LOBO - LOPO

The gap is the part of the deficit that having data from the test infant can
fix -- i.e. the part domain adaptation or personalisation could in principle
address. If the gap is ~0, adaptation is not the answer and section E of the
to-do list can be skipped. Section 4.4 of the thesis and
Section 4.5 of the thesis already report that nothing transferred; this measures why.

    NOTE: D7 (scripts/subject_identifiability.py) found that an infant is
    identifiable from a control window at 6-7x chance, so subject-specific
    nuisance is definitely present. D3 asks the different and more useful
    question of whether that nuisance is what LIMITS the prediction.

DESIGN NOTES
    * Both arms are trained by the same function the main pipeline uses
      (`train_model_and_evaluate_performance`), with the same sampler,
      optimizer and epoch count, so the two numbers are comparable.
    * The paired comparison is per BLOCK, not per infant: for held-out block b
      of infant p, the LOPO arm is scored on exactly the same windows, so the
      difference is not confounded by which windows each arm was tested on.
    * Results are appended to the output JSON after every fold, so a run that
      is interrupted still leaves usable partial results.

COST
    15 infants x 4 blocks x 2 arms = 120 trainings. This is the most expensive
    item on the list and is meant for the GPU VM. Use --ids and --max-blocks to
    size it down, and --dry-run to print the schedule without training.

Usage:
    python scripts/leave_one_block_out.py --dry-run
    python scripts/leave_one_block_out.py --ids 001 002 003
    python scripts/leave_one_block_out.py --out outputs/d3/lobo.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.train_neonatal import (  # noqa: E402
    init_neonatal_dataset,
    set_seed,
    train_model_and_evaluate_performance,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--out", default="outputs/d3/lobo.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-blocks", type=int, default=None,
                   help="only the first N blocks per infant")
    p.add_argument("--min-test-pos", type=int, default=5,
                   help="skip a block with fewer positive windows than this")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="hydra-style overrides, e.g. optimizer.epochs=10")
    return p.parse_args()


def subset_by_block(ds, blocks, keep=True):
    """A shallow copy of a NeoNatal dataset restricted to (or excluding) blocks.

    Shallow-copying and swapping `time_window_df` keeps `_get_labels` and
    `__getitem__` intact, which matters because MultiDatasetBalancedSampler
    calls `_get_labels()` on every dataset inside the ConcatDataset. A plain
    torch Subset would not have that method and the sampler would fail.
    """
    new = copy.copy(ds)
    in_blocks = ds.time_window_df["segment_idx"].isin(blocks)
    sel = in_blocks if keep else ~in_blocks
    new.time_window_df = ds.time_window_df[sel].reset_index(drop=True)
    return new


def usable(ds, min_pos=1):
    if not len(ds.time_window_df):
        return False
    labels = ds.time_window_df["label"]
    return labels.sum() >= min_pos and (len(labels) - labels.sum()) >= 1


def auc_of(labs, scores):
    try:
        return float(roc_auc_score(labs, scores))
    except ValueError:
        return float("nan")


def main():
    args = parse_args()
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "config")):
        cfg = compose(config_name="neonatal_config", overrides=list(args.overrides))
    if args.ids:
        cfg.dataset.ids = list(args.ids)

    ids = list(cfg.dataset.ids)
    print(f"D3 leave-one-block-out.  infants={len(ids)}  "
          f"epochs={cfg.optimizer.epochs}  seed={args.seed}")
    print(f"cuda available: {torch.cuda.is_available()}\n")

    set_seed(args.seed)
    print("Reading and windowing every infant once...")
    datasets = {}
    for pat_id in ids:
        datasets[pat_id] = init_neonatal_dataset(pat_id, cfg)
    print("done.\n")

    schedule = []
    for pat_id in ids:
        blocks = sorted(datasets[pat_id].time_window_df["segment_idx"].unique().tolist())
        if args.max_blocks:
            blocks = blocks[: args.max_blocks]
        for b in blocks:
            test = subset_by_block(datasets[pat_id], [b], keep=True)
            train_own = subset_by_block(datasets[pat_id], [b], keep=False)
            n_pos = int(test.time_window_df["label"].sum())
            if n_pos < args.min_test_pos or not usable(train_own):
                print(f"  skip {pat_id} block {b}: {n_pos} positive test windows")
                continue
            schedule.append((pat_id, b, n_pos))

    print(f"{len(schedule)} folds x 2 arms = {2 * len(schedule)} trainings\n")
    if args.dry_run:
        for pat_id, b, n_pos in schedule:
            print(f"  {pat_id} block {b}: {n_pos} positive test windows")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for i, (pat_id, b, n_pos) in enumerate(schedule, 1):
        t0 = time.time()
        test = subset_by_block(datasets[pat_id], [b], keep=True)
        own_rest = subset_by_block(datasets[pat_id], [b], keep=False)
        others = [datasets[j] for j in ids if j != pat_id and usable(datasets[j])]

        # LOPO arm: the test infant contributes nothing.
        set_seed(args.seed)
        _, labs_p, scores_p, train_auc_p = train_model_and_evaluate_performance(
            cfg, ConcatDataset(others), test, log_to_wandb=False
        )
        # LOBO arm: the same, plus this infant's three other blocks.
        set_seed(args.seed)
        _, labs_b, scores_b, train_auc_b = train_model_and_evaluate_performance(
            cfg, ConcatDataset(others + [own_rest]), test, log_to_wandb=False
        )

        lopo, lobo = auc_of(labs_p, scores_p), auc_of(labs_b, scores_b)
        rows.append({
            "patient": pat_id, "block": int(b), "n_test_pos": n_pos,
            "n_test": int(len(test.time_window_df)),
            "lopo_auc": lopo, "lobo_auc": lobo, "gap": lobo - lopo,
            "lopo_train_auc": float(train_auc_p), "lobo_train_auc": float(train_auc_b),
        })
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"[{i}/{len(schedule)}] {pat_id} block {b} "
              f"({n_pos} pos): LOPO {lopo:.3f}  LOBO {lobo:.3f}  "
              f"gap {lobo - lopo:+.3f}   [{time.time() - t0:.0f}s]")

    if not rows:
        print("no folds ran")
        return

    gaps = np.array([r["gap"] for r in rows], dtype=float)
    lopo = np.array([r["lopo_auc"] for r in rows], dtype=float)
    lobo = np.array([r["lobo_auc"] for r in rows], dtype=float)
    w = np.array([r["n_test_pos"] for r in rows], dtype=float)
    ok = np.isfinite(gaps)
    print("\n" + "=" * 74)
    print(f"folds: {int(ok.sum())}")
    print(f"LOPO  mean {np.nanmean(lopo):.3f}   count-weighted "
          f"{np.sum(lopo[ok] * w[ok]) / w[ok].sum():.3f}")
    print(f"LOBO  mean {np.nanmean(lobo):.3f}   count-weighted "
          f"{np.sum(lobo[ok] * w[ok]) / w[ok].sum():.3f}")
    print(f"GAP   mean {np.nanmean(gaps):+.3f}  (SD {np.nanstd(gaps):.3f}), "
          f"positive in {int((gaps[ok] > 0).sum())}/{int(ok.sum())} folds")
    try:
        from scipy.stats import wilcoxon
        print(f"Wilcoxon signed-rank on the paired gaps: p="
              f"{wilcoxon(gaps[ok])[1]:.4f}")
    except Exception:
        pass
    print()
    print("READING")
    print("  A gap near zero means seeing three blocks of the test infant does")
    print("  not help, so the deficit is NOT subject shift and section E of the")
    print("  to-do list cannot fix it. A clearly positive gap means the opposite")
    print("  and would make personalisation (E7) worth the annotation cost.")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
