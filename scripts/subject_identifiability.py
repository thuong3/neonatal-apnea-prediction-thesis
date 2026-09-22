"""D7: HOW MUCH SUBJECT-SPECIFIC NUISANCE SURVIVES NORMALISATION?

Item D7 asks for a classifier that identifies the INFANT (and separately the
BLOCK) from control windows alone. If an infant can be recognised almost
perfectly from a window of quiet breathing, then a large part of what the
network sees is subject identity rather than physiology -- which is what makes
leave-one-patient-out hard, and which is exactly what the B2 normalisation is
supposed to reduce. It also gives a clean before/after check on B2 and B3 that
does not depend on the apnea labels at all.

DESIGN, AND WHY THE SPLIT MATTERS
    Windows from the same block are highly correlated: consecutive 30 s windows
    of the same infant on the same device share baseline, gain, belt placement
    and posture. A random train/test split over windows would therefore report
    near-perfect accuracy for trivial reasons -- it would be memorising blocks,
    not identifying infants.

    So the split is LEAVE-ONE-BLOCK-OUT: train on three of an infant's device
    blocks, test on the fourth, for every infant simultaneously. Accuracy then
    answers the question that matters -- does an infant stay recognisable when
    the device and the hours change?

    Chance is 1/15 = 6.7%.

WHAT IS COMPARED
    The same windows under the two normalisations of section 27:
        per-window standardisation (the published pipeline)
        per-block robust normalisation (B2)
    Lower identifiability is better: it means less subject-specific nuisance
    reaches the model.

Only CONTROL windows are used (label 0), per D7, so nothing here can be
contaminated by the event labels.

Usage:
    python scripts/subject_identifiability.py --ids 001 002 003
    python scripts/subject_identifiability.py            # all 15 (slow)
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import NeoNatal, load_clock_drift, read_data  # noqa: E402

DEFAULT_DATA = "data/brainimmaturity"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--cache", default="outputs/identifiability/features.npz")
    return p.parse_args()


def window_features(sig_list):
    """A few shape statistics per channel. Deliberately simple.

    The point of D7 is whether subject identity is present at all, not to build
    the best possible identifier, so the feature set is kept plain enough that a
    high score cannot be explained by feature engineering.
    """
    feats = []
    for ch in sig_list:
        ch = np.asarray(ch, dtype=float)
        if ch.size == 0 or not np.all(np.isfinite(ch)):
            feats.extend([0.0] * 7)
            continue
        q25, q50, q75 = np.percentile(ch, [25, 50, 75])
        feats.extend([
            float(ch.mean()), float(ch.std()),
            float(np.sqrt(np.mean(ch ** 2))), float(np.ptp(ch)),
            float(q25), float(q50), float(q75),
        ])
    return feats


def collect(ids, cfg, data_path, norm, drift):
    X, y_pat, y_block = [], [], []
    for pat_id in ids:
        signal_dict = read_data(
            pat_id, data_path,
            annotations_dir=cfg.get("annotations_dir", "annotations"),
            record_duration=cfg.get("record_duration", None),
            clock_drift=drift.get(pat_id),
        )
        ds = NeoNatal(
            pat_id, signal_dict, dataset_mode="list",
            signal_types=cfg.signal_types, adverse_events=cfg.adverse_events,
            cutter_events=cfg.cutter_events, time_window=cfg.time_window,
            lag=cfg.lag, away=cfg.away, **norm,
        )
        df = ds.time_window_df
        if not len(df):
            continue
        df = df[df["label"] == 0]  # D7: control windows only
        for i in range(len(df)):
            row = df.iloc[i]
            X.append(window_features(row["sig"]))
            y_pat.append(pat_id)
            y_block.append(int(row["segment_idx"]))
    return np.asarray(X, dtype=float), np.asarray(y_pat), np.asarray(y_block)


def leave_one_block_out(X, y_pat, y_block):
    """Train on three device blocks per infant, test on the fourth."""
    accs = []
    for held in sorted(set(y_block.tolist())):
        te = y_block == held
        tr = ~te
        if te.sum() < 10 or len(set(y_pat[tr].tolist())) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(X[tr], y_pat[tr])
        accs.append(accuracy_score(y_pat[te], clf.predict(X[te])))
    return np.asarray(accs)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    drift = load_clock_drift(cfg.get("clock_drift_file", None))
    chance = 1.0 / len(ids)

    print(f"D7: identifying which of {len(ids)} infants a CONTROL window came from.")
    print(f"Leave-one-block-out over the 4 device blocks. Chance = {chance:.3f}\n")

    results = {}
    for name, norm in (
        ("per-window standardisation (published)",
         dict(norm_per_window=True, norm_per_block=False)),
        ("per-block robust normalisation (B2)",
         dict(norm_per_window=False, norm_per_block=True)),
    ):
        X, y_pat, y_block = collect(ids, cfg, args.data_path, norm, drift)
        if len(X) == 0:
            print(f"{name}: no control windows")
            continue
        accs = leave_one_block_out(X, y_pat, y_block)
        results[name] = accs
        print(f"{name}:")
        print(f"    windows={len(X)}  accuracy={accs.mean():.3f} "
              f"(SD {accs.std():.3f}, folds {len(accs)})")
        print(f"    that is {accs.mean() / chance:.1f}x chance\n")

    if len(results) == 2:
        a, b = list(results.values())
        print("=" * 74)
        print(f"published : {a.mean():.3f}   B2 per-block : {b.mean():.3f}   "
              f"change {b.mean() - a.mean():+.3f}")
        print()
        print("READING")
        print("  Well above chance under both means subject identity survives")
        print("  whatever normalisation is applied -- the infants stay")
        print("  recognisable from quiet breathing alone, which is the nuisance")
        print("  variance leave-one-patient-out has to fight.")
        print("  A LOWER number under B2 means per-block normalisation removed")
        print("  some of that identity, which is the effect it was introduced")
        print("  for. A higher number means it added identity instead -- worth")
        print("  knowing before the four sweeps are read.")


if __name__ == "__main__":
    main()
