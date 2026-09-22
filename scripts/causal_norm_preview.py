"""B5's DELIVERABLE, MEASURED WITHOUT TRAINING ANYTHING.

B5: "Report (i) whole-block statistics as primary ... and (ii) causal statistics
from an expanding window with a 10-minute warm-up, or a trailing 30-minute
window, matching real-time use. The difference between the two is the price of
real-time operation and deserves a sentence in the paper."

The expensive way to get that sentence is two more full sweeps. This script gets
most of it in minutes, on real data, with no GPU -- and it answers the question
that decides whether the sweeps are worth running at all:

    HOW DIFFERENT IS THE SIGNAL THE NETWORK ACTUALLY SEES?

If the causal arms hand the network almost the same numbers as the transductive
one, then the AuROC cannot differ much either, the price of real-time operation
is ~0 by construction, and that is the sentence -- no GPU required. If they
differ a lot, the sweeps are justified and this says by how much.

WHY THE SIGNAL AND NOT THE STATISTICS. The medians and scales are the mechanism,
but the network never sees them: it sees (x - m)/s. A 5% error in s matters far
more on a channel whose s is small than on one whose s is large, so comparing m
and s across channels is not interpretable. The normalised signal is, and in a
useful unit: after normalisation ONE UNIT IS ROUGHLY ONE TYPICAL BREATH
AMPLITUDE (section 27.2), so a difference of 0.05 is 5% of a breath.

THE UNIT THE ANSWER IS REPORTED IN. An absolute difference is unreadable on its
own, so everything is also expressed RELATIVE TO THE SIGNAL'S OWN AMPLITUDE:

    relative change = RMS(causal - transductive) / RMS(transductive)

i.e. "what fraction of the input the network sees actually changed". 0.01 means
the causal arm hands the network a 1%-different signal, which no network is going
to turn into a measurable AuROC difference; 0.50 means it sees something largely
different and the sweeps are clearly justified.

A REJECTED REFERENCE, recorded so it is not tried again. The first version of
this script used norm_block-vs-norm_none as a yardstick, on the grounds that
section 27.6 measured its AuROC effect at -0.0080 (not detectable). That is
invalid and the numbers it produced were meaningless: norm_none is UNNORMALISED,
so the comparison is raw units against unit scale (a Thorax RMS difference of 54,
a PR one of 509), and it is dominated by the change of unit system rather than by
any perturbation of shape. It also compares two SEPARATELY TRAINED networks, each
of which adapts to its own input scale -- so a large input difference there
genuinely does not imply a large AuROC difference. Comparing to the signal's own
amplitude avoids both problems.

Run:
    python scripts/causal_norm_preview.py --ids 001 002 003
    python scripts/causal_norm_preview.py                  # all 15, slower
"""
import argparse
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    CAUSAL_TRAILING_S,
    CAUSAL_WARMUP_S,
    NeoNatal,
    load_clock_drift,
    read_data,
)

DEFAULT_DATA = "data/brainimmaturity"
TARGET_FREQ = 200
# The floor measured in section 27.2a. Identical in all arms, so it cannot be
# what any difference below is measuring.
FLOOR = {"Abdomen": 5.228, "CPAP": 0.09263, "PR": 5.468,
         "Thorax": 3.273, "Thorax_sum": 0.3637}
# Below this fraction of the signal's own amplitude, a difference in the input
# cannot plausibly produce a detectable difference in AuROC at 15 infants.
# Deliberately generous: 5% of the input changing is already a lot to ask a
# network to ignore, so a verdict of "negligible" here is a conservative one.
NEGLIGIBLE_REL = 0.05

ARMS = {
    "whole_block": dict(norm_per_window=False, norm_per_block=True,
                        norm_stats_mode="whole_block"),
    "expanding": dict(norm_per_window=False, norm_per_block=True,
                      norm_stats_mode="expanding"),
    "trailing": dict(norm_per_window=False, norm_per_block=True,
                     norm_stats_mode="trailing"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None)
    return p.parse_args()


def build(pat_id, signal_dict, cfg, **norm):
    return NeoNatal(
        pat_id, signal_dict, dataset_mode="list",
        signal_types=cfg.signal_types, adverse_events=cfg.adverse_events,
        cutter_events=cfg.cutter_events, time_window=cfg.time_window,
        lag=cfg.lag, away=cfg.away, norm_scale_floor=FLOOR, **norm,
    )


def per_window_diff(a_sigs, b_sigs, channel_index):
    """Per window: (absolute RMS difference, difference relative to amplitude).

    `a` is the transductive arm and is the denominator -- it is the primary, so
    "how much did the causal arm move the input" is measured against it.
    """
    absolute, relative = [], []
    for a, b in zip(a_sigs, b_sigs):
        xa = np.asarray(a[channel_index], dtype=float)
        xb = np.asarray(b[channel_index], dtype=float)
        n = min(len(xa), len(xb))
        xa, xb = xa[:n], xb[:n]
        d = float(np.sqrt(np.mean((xa - xb) ** 2)))
        amp = float(np.sqrt(np.mean(xa ** 2)))
        absolute.append(d)
        relative.append(d / amp if amp > 0 else np.nan)
    return np.asarray(absolute), np.asarray(relative)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    drift = load_clock_drift(cfg.get("clock_drift_file", None))
    # The channels per-block normalisation actually touches. HR/SpO2/PCO2 are
    # range-normalised over fixed clinical ranges and are identical in all arms.
    channels = [(i, c) for i, c in enumerate(cfg.signal_types)
                if c in ("CPAP", "Thorax", "PR")]
    arms = ("expanding", "trailing")

    print("B5 PREVIEW -- the price of real-time normalisation, before training")
    print()
    print(f"arms: whole_block (transductive, PRIMARY) | expanding "
          f"({CAUSAL_WARMUP_S / 60:.0f}-min warm-up) | trailing "
          f"({CAUSAL_TRAILING_S / 60:.0f} min)")
    print("measure: RMS(causal - transductive) per window, absolute and as a")
    print("         fraction of the transductive signal's own amplitude")
    print()

    absol = {a: {c: [] for _, c in channels} for a in arms}
    rel = {a: {c: [] for _, c in channels} for a in arms}
    # Drift vs position in the block: the expanding arm should converge on the
    # transductive one as history accumulates, the trailing arm should not.
    by_hour = {a: {} for a in arms}
    warm_frac, n_windows = [], 0
    t0 = time.time()

    for pat_id in ids:
        signal_dict = read_data(
            pat_id, args.data_path,
            annotations_dir=cfg.get("annotations_dir", "annotations"),
            record_duration=cfg.get("record_duration", None),
            clock_drift=drift.get(pat_id),
        )
        sigs, table = {}, None
        for arm, kw in ARMS.items():
            ds = build(pat_id, signal_dict, cfg, **kw)
            if not len(ds.time_window_df):
                sigs = {}
                break
            sigs[arm] = list(ds.time_window_df["sig"])
            if table is None:
                table = ds.time_window_df
        if not sigs:
            print(f"{pat_id}: no windows")
            continue

        # Windows inside the warm-up use statistics that postdate them. Counted,
        # not hand-waved -- it is the one concession the causal arms make.
        starts = table["slice"].map(lambda s: s.start).to_numpy()
        warm = float(np.mean(starts < CAUSAL_WARMUP_S * TARGET_FREQ))
        warm_frac.append(warm)
        n_windows += len(table)
        hours = starts / (TARGET_FREQ * 3600.0)

        line = [f"{pat_id}: {len(table):5d} win, warm-up {warm:5.1%}"]
        for arm in arms:
            for idx, name in channels:
                a, r = per_window_diff(sigs["whole_block"], sigs[arm], idx)
                absol[arm][name].append(a)
                rel[arm][name].append(r)
                if name == "Thorax":
                    for h, v in zip(np.floor(hours).astype(int), r):
                        by_hour[arm].setdefault(int(h), []).append(v)
            th = rel[arm]["Thorax"][-1]
            line.append(f"{arm}: Thorax {np.nanmedian(th):6.2%}")
        print(" | ".join(line), flush=True)

    if not warm_frac:
        print("no usable data")
        return

    print()
    print(f"collected {n_windows} windows over {len(warm_frac)} infants "
          f"in {(time.time() - t0) / 60:.1f} min")
    print(f"windows inside the warm-up: {np.mean(warm_frac):.1%} -- these use "
          "statistics that postdate them (the documented concession)")

    print()
    print("=" * 78)
    print("[1] HOW FAR THE INPUT MOVES vs the transductive arm")
    print("    absolute is in normalised units (1.0 ~ one typical breath);")
    print("    relative is the fraction of the signal's own amplitude.")
    print()
    print(f"    {'arm':<11} {'channel':<9} {'abs med':>9} {'abs p95':>9} "
          f"{'REL med':>9} {'rel p95':>9}")
    summary = {}
    for arm in arms:
        for _, name in channels:
            a = np.concatenate(absol[arm][name])
            r = np.concatenate(rel[arm][name])
            r = r[np.isfinite(r)]
            summary[(arm, name)] = float(np.median(r))
            print(f"    {arm:<11} {name:<9} {np.median(a):>9.4f} "
                  f"{np.percentile(a, 95):>9.4f} {np.median(r):>8.2%} "
                  f"{np.percentile(r, 95):>8.2%}")

    print()
    print("=" * 78)
    print("[2] DOES THE PRICE FALL AS HISTORY ACCUMULATES?  (Thorax, relative)")
    print("    The expanding arm sees more of the block as the night goes on and")
    print("    should converge on the transductive statistics. The trailing arm")
    print("    keeps a 30-min memory and should NOT converge -- if it does, the")
    print("    two are not really different transforms and one of them is wrong.")
    print()
    print(f"    {'hour into block':<19}" + "".join(f"{a:>13}" for a in arms))
    for h in sorted(set().union(*[set(by_hour[a]) for a in arms])):
        cells = []
        for a in arms:
            v = [x for x in by_hour[a].get(h, []) if np.isfinite(x)]
            cells.append(f"{np.median(v):>12.2%}" if v else f"{'-':>12}")
        print(f"    {h}-{h + 1} h{'':<12}" + "".join(cells))

    print()
    print("=" * 78)
    print("VERDICT")
    worst_arm, worst_ch = max(summary, key=summary.get)
    worst = summary[(worst_arm, worst_ch)]
    print(f"  largest median relative change: {worst:.2%} "
          f"({worst_arm}, {worst_ch})")
    print(f"  threshold for 'negligible': {NEGLIGIBLE_REL:.0%}")
    if worst < NEGLIGIBLE_REL:
        print("  -> The causal arms hand the network essentially the same input.")
        print("     The price of real-time operation is below what this study")
        print("     could detect; report it as measured, and the sweeps would")
        print("     confirm a null expensively.")
    else:
        print("  -> The causal arms change the input materially. Run both causal")
        print("     cells at the 10 pre-specified seeds and compare against")
        print("     norm_block, which is the primary.")
    print()
    print("  NOTE: this bounds the INPUT difference, not the AuROC difference.")
    print("  A small input change cannot produce a large output change, but the")
    print("  converse is not guaranteed -- read it as a screening test, exactly")
    print("  as section 27.4's preview is read.")


if __name__ == "__main__":
    main()
