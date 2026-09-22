"""C1 + C6: INTER-EVENT GAPS, PERIODIC BREATHING, AND WHAT SURVIVES WINDOWING.

Item C1 of the supervisor's re-analysis to-do list argues that the published
window construction may degenerate on this cohort:

    In periodic breathing -- three or more central pauses separated by no more
    than 20 s of normal breathing -- and with pauses of >=10 s by the scoring
    threshold, the surviving inter-event stretches are on the order of 10 s.
    NO CONTIGUOUS 30-SECOND WINDOW CAN BE BUILT INSIDE A PERIODIC-BREATHING RUN
    AT ALL.

If that is right, then every control window in this study comes from OUTSIDE
periodic breathing, the target/control contrast is "periodic breathing vs.
regular breathing" rather than "about to have an apnea vs. not", and the C4
stratified analysis is not optional. This script measures it rather than
assuming it, and reports:

    1. the inter-event gap distribution, per block and pooled          (C1)
    2. periodic-breathing runs and how much of the record they cover   (C1/C4)
    3. how many 30 s windows can be built INSIDE a PB run              (C1)
    4. whether any window is built by CONCATENATION across a removed
       segment -- the failure mode C1 warns about                      (C1)
    5. target-window counts per infant and per block                   (C6)

DEFINITION USED FOR PERIODIC BREATHING
    Kelly & Shannon, as cited by Vetter et al.: >=3 central pauses separated by
    <=20 s of breathing. The pause-length term of that definition (>=4 s) is not
    applied here and cannot be: this cohort's annotation layer only contains
    pauses that already passed the scorer's >=10 s threshold, so every scored
    event qualifies on duration by construction. The consequence is that these
    runs are a SUBSET of true periodic breathing -- short pauses that a scorer
    would have counted towards a PB run are invisible in the annotation. Runs
    are therefore under-counted, never over-counted.

Usage:
    python scripts/pb_and_gap_stats.py
    python scripts/pb_and_gap_stats.py --ids 001 002 --max-gap 20
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    NeoNatal,
    create_slices,
    load_clock_drift,
    read_data,
)

TARGET_FREQ = 200
DEFAULT_DATA = "data/brainimmaturity"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--max-gap", type=float, default=20.0,
                   help="max breathing gap inside a PB run, seconds (Kelly & Shannon)")
    p.add_argument("--min-events", type=int, default=3,
                   help="min pauses in a PB run (Kelly & Shannon)")
    return p.parse_args()


def event_spans(mask):
    """[(start, stop)] sample spans of the 1-runs in a 0/1 mask."""
    mask = np.asarray(mask).astype(np.int8)
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    return list(zip(starts, stops))


def pb_runs(spans, max_gap_s, min_events):
    """Maximal groups of >=min_events events whose consecutive gaps are <=max_gap_s.

    Returns [(run_start_sample, run_stop_sample, n_events)].
    """
    if len(spans) < min_events:
        return []
    max_gap = max_gap_s * TARGET_FREQ
    runs, current = [], [spans[0]]
    for prev, cur in zip(spans, spans[1:]):
        if (cur[0] - prev[1]) <= max_gap:
            current.append(cur)
        else:
            if len(current) >= min_events:
                runs.append((current[0][0], current[-1][1], len(current)))
            current = [cur]
    if len(current) >= min_events:
        runs.append((current[0][0], current[-1][1], len(current)))
    return runs


def describe(values, unit="s"):
    if len(values) == 0:
        return "n=0"
    v = np.asarray(values, dtype=float)
    return (f"n={len(v)} median={np.median(v):.1f}{unit} "
            f"IQR={np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}{unit} "
            f"min={v.min():.1f} max={v.max():.1f}")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    drift = load_clock_drift(cfg.get("clock_drift_file", None))
    window_s = cfg.time_window / TARGET_FREQ
    away_s = cfg.away / TARGET_FREQ

    print(f"window={window_s:.0f}s  lag={cfg.lag / TARGET_FREQ:.0f}s  "
          f"away={away_s:.0f}s   PB: >={args.min_events} events, "
          f"gaps <={args.max_gap:.0f}s\n")

    all_gaps, per_patient, concat_violations = [], [], 0
    pb_total_s = analysed_total_s = 0.0
    pb_windows_possible = 0

    for pat_id in ids:
        segments = read_data(
            pat_id,
            args.data_path,
            annotations_dir=cfg.get("annotations_dir", "annotations"),
            record_duration=cfg.get("record_duration", None),
            clock_drift=drift.get(pat_id),
        )
        ds = NeoNatal(
            pat_id, segments, dataset_mode="list", signal_types=cfg.signal_types,
            adverse_events=cfg.adverse_events, cutter_events=cfg.cutter_events,
            time_window=cfg.time_window, lag=cfg.lag, away=cfg.away,
        )
        df = ds.time_window_df

        pat_gaps, pat_pb_s, pat_analysed_s, pat_runs = [], 0.0, 0.0, 0
        block_rows = []
        for si, seg in enumerate(segments):
            spans = event_spans(seg["APNEA-CENTRAL"])
            seg_len_s = len(seg["APNEA-CENTRAL"]) / TARGET_FREQ
            pat_analysed_s += seg_len_s

            gaps = [(b[0] - a[1]) / TARGET_FREQ for a, b in zip(spans, spans[1:])]
            pat_gaps.extend(gaps)

            runs = pb_runs(spans, args.max_gap, args.min_events)
            run_s = sum((b - a) / TARGET_FREQ for a, b, _ in runs)
            pat_pb_s += run_s
            pat_runs += len(runs)

            # C1 bullet 3: can a contiguous 30 s window be built inside a PB run?
            # "Inside" means within the run and clear of every scored event, i.e.
            # in one of the breathing gaps between its pauses.
            for run_a, run_b, _ in runs:
                inner = [(p[1], q[0]) for p, q in zip(spans, spans[1:])
                         if p[1] >= run_a and q[0] <= run_b]
                pb_windows_possible += sum(
                    1 for s, e in inner if (e - s) >= cfg.time_window
                )

            n_block = int((df["segment_idx"] == si).sum()) if len(df) else 0
            n_pos_block = (
                int(df[(df["segment_idx"] == si) & (df["label"] == 1)].shape[0])
                if len(df) else 0
            )
            block_rows.append((si, len(spans), n_block, n_pos_block, len(runs), run_s))

            # C1 bullet 4: no window may span a removed segment. Windows are cut
            # inside the contiguous slices of the cutter mask, so this should be
            # structurally impossible -- verified rather than asserted.
            cutter = 1 - np.column_stack(
                [seg[e] for e in (list(cfg.adverse_events) + list(cfg.cutter_events))]
            ).max(axis=1)
            slices = create_slices(cutter)
            if len(df):
                for sl in df[df["segment_idx"] == si]["slice"]:
                    inside = any(
                        s.start <= sl.start and sl.stop <= s.stop for s in slices
                    )
                    if not inside:
                        concat_violations += 1

        n_pos = int(df["label"].sum()) if len(df) else 0
        n_tot = len(df)
        all_gaps.extend(pat_gaps)
        pb_total_s += pat_pb_s
        analysed_total_s += pat_analysed_s
        per_patient.append((pat_id, n_pos, n_tot, len(pat_gaps) + 1, pat_runs,
                            pat_pb_s, pat_analysed_s))

        print(f"--- patient {pat_id} ---")
        print(f"  events={sum(r[1] for r in block_rows):4d}  "
              f"windows={n_tot:5d} ({n_pos} target, {100 * n_pos / max(n_tot, 1):.1f}%)")
        print(f"  inter-event gaps: {describe(pat_gaps)}")
        if pat_gaps:
            frac = 100 * np.mean(np.asarray(pat_gaps) <= args.max_gap)
            print(f"    gaps <= {args.max_gap:.0f}s: {frac:.0f}%   "
                  f"gaps >= {window_s:.0f}s (a window could fit): "
                  f"{100 * np.mean(np.asarray(pat_gaps) >= window_s):.0f}%")
        print(f"  PB runs: {pat_runs}, covering {pat_pb_s / 60:.0f} min "
              f"({100 * pat_pb_s / max(pat_analysed_s, 1):.1f}% of analysed time)")
        for si, n_ev, n_win, n_p, n_run, run_s in block_rows:
            print(f"    block {si}: {n_ev:4d} events, {n_win:5d} windows "
                  f"({n_p:4d} target), {n_run:3d} PB runs, {run_s / 60:5.0f} min PB")
        print()

    print("=" * 78)
    g = np.asarray(all_gaps, dtype=float)
    print(f"POOLED inter-event gaps: {describe(g)}")
    print(f"  <= {args.max_gap:.0f}s (periodic-breathing spacing): "
          f"{100 * np.mean(g <= args.max_gap):.1f}%")
    print(f"  >= {window_s:.0f}s (a control window would fit between events): "
          f"{100 * np.mean(g >= window_s):.1f}%")
    print(f"  >= {away_s:.0f}s (the control rule's separation): "
          f"{100 * np.mean(g >= away_s):.1f}%")
    print()
    print(f"periodic breathing covers {pb_total_s / 3600:.1f} h of "
          f"{analysed_total_s / 3600:.1f} h analysed "
          f"({100 * pb_total_s / max(analysed_total_s, 1):.1f}%)")
    print(f"contiguous {window_s:.0f}s stretches available INSIDE PB runs: "
          f"{pb_windows_possible}")
    print()
    print(f"windows built by CONCATENATION across a removed segment: "
          f"{concat_violations}  (C1 bullet 4 -- must be 0)")
    print()
    print("C6 -- target windows per infant (the 38-fold skew):")
    counts = np.array([r[1] for r in per_patient])
    for pid, n_pos, n_tot, _, _, _, _ in per_patient:
        print(f"    {pid}: {n_pos:4d} target / {n_tot:5d} total "
              f"({100 * n_pos / max(n_tot, 1):4.1f}%)")
    print(f"    range {counts.min()}-{counts.max()} "
          f"({counts.max() / max(counts.min(), 1):.0f}-fold), median {np.median(counts):.0f}")
    print()
    print("READING")
    print("  If the pooled gap distribution sits mostly below the window length,")
    print("  C1 is confirmed: controls cannot come from inside periodic")
    print("  breathing, so the published contrast is PB vs. regular breathing")
    print("  and C4 must redefine the control set (shorter window, or controls")
    print("  drawn from PB gaps at a reduced separation).")


if __name__ == "__main__":
    main()
