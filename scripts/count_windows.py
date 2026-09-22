"""Report how many time windows (and how many positive ones) a dataset config
produces per patient -- without training anything.

Use this before committing GPU/CPU hours to a run: leave-one-patient-out needs
both classes present in every test patient, otherwise `roc_auc_score` in
src/train_neonatal.py raises ValueError and the whole sweep dies at the end.
It is also the cheapest way to see whether a restricted event definition (e.g.
central apnea only, config/dataset/robin_sequence_central.yaml) still leaves
enough positives to learn from.

Usage:
    python scripts/count_windows.py --dataset config/dataset/robin_sequence_central.yaml \
        --data_path data/robin_sequence [--min_positive 5]
"""
import argparse
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    NeoNatal,
    load_clock_drift,
    read_data,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="path to a config/dataset/*.yaml")
    p.add_argument("--data_path", required=True, help="folder with signals/ and annotations/")
    p.add_argument(
        "--min_positive",
        type=int,
        default=5,
        help="patients below this many positive windows are flagged as unusable",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    # From the dataset config, so it matches what training would build.
    drift = load_clock_drift(cfg.get("clock_drift_file"))

    print(f"{'pat':>5} {'windows':>9} {'pos':>6} {'neg':>6} {'pos%':>7}  note")
    usable, dropped, total_pos, total_win = [], [], 0, 0
    for pat_id in cfg.ids:
        dataset = NeoNatal(
            pat_id,
            # Drift table comes from the dataset config being counted, so this
            # matches whatever a training run with that config would build.
            read_data(pat_id, args.data_path,
                      clock_drift=drift.get(pat_id)),
            dataset_mode=cfg.dataset_mode,
            signal_types=cfg.signal_types,
            adverse_events=cfg.adverse_events,
            cutter_events=cfg.cutter_events,
            time_window=cfg.time_window,
            lag=cfg.lag,
            away=cfg.away,
        )
        labels = dataset._get_labels()
        n, pos = len(labels), int(labels.sum())
        neg = n - pos
        total_win += n
        total_pos += pos

        if pos == 0 or neg == 0:
            note = "UNUSABLE (only one class -> roc_auc_score raises)"
            dropped.append(pat_id)
        elif pos < args.min_positive:
            note = f"thin ({pos} positives)"
            dropped.append(pat_id)
        else:
            note = ""
            usable.append(pat_id)
        share = 100 * pos / n if n else 0.0
        print(f"{pat_id:>5} {n:>9} {pos:>6} {neg:>6} {share:>6.1f}%  {note}")

    print(f"\ntotal: {total_win} windows, {total_pos} positive "
          f"({100 * total_pos / total_win if total_win else 0:.1f}%)")
    print(f"usable patients (>= {args.min_positive} positives): {len(usable)}/{len(cfg.ids)}")
    if dropped:
        print(f"drop these: {dropped}")
        print("\nids override for train_neonatal.py:")
        print("  dataset.ids='[" + ",".join(f'\"{p}\"' for p in usable) + "]'")


if __name__ == "__main__":
    main()
