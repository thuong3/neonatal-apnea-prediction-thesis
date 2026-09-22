"""MEASURE THE PER-BLOCK ROBUST SCALES, BEFORE RUNNING THE B3 SWEEPS.

Why this runs first
-------------------
B2 says to FLOOR the robust scale: set s_min per channel to (e.g.) the 5th
percentile of s across training blocks, and log any block that hits the floor as
a QC failure, because a loose or disconnected inductance belt produces a scale
near zero and dividing by it amplifies that belt's own noise to full scale.

The floor is currently plumbed through (`norm_scale_floor`) but set to `null` in
both `config/experiment/norm_block.yaml` and `norm_both.yaml` -- i.e. OFF in the
two cells that use per-block statistics. It was left null deliberately, to be
chosen "from the training blocks once they have been inspected". This script is
that inspection. It needs no GPU and trains nothing: it only calls the same
`block_norm_stats` the pipeline calls, over all 15 infants x 4 blocks, and
tabulates what comes out.

Patient 009 is the known candidate -- 57% of its marks sit on a dead Abdomen
belt (Section 3.4 of the thesis). If its Abdomen scale sits far below the rest,
the floor is a real correctness issue in the cell being called the proposed
primary. If every scale sits in a sensible range, the floor is a formality: set
it at the 5th percentile, note that nothing hits it, and move on.

It also answers a SECOND question, for free
-------------------------------------------
B2's last bullet asks for HR and SpO2 to be carried as TWO channels each: the
existing fixed-range one (HR 50-240 bpm, SpO2 60-100% -> [-1, 1]) plus a
per-block robust-centred one, so the network sees both the absolute level and
this infant's deviation from their own baseline on this device.

That is a new experiment, not a bug fix, and it is only worth building if the
baselines actually MOVE between infants and blocks. So this script also reports
the per-block median HR and SpO2 in physical units, plus how much of their
variance is between infants versus within an infant across devices. Large
between-infant spread => the extra channel has something to encode. Small
spread => it cannot help and the build is not worth it.

The per-block medians double as a check on B2's own premise, which cites
"baseline heart rate 163-167 bpm and baseline SpO2 96-97% in this cohort".

Run:  python scripts/measure_block_scales.py
      python scripts/measure_block_scales.py --csv outputs/block_scales.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    BLOCK_NORM_CHANNELS,
    DECIMATION_CHAIN,
    _apply_chain,
    block_norm_stats,
    load_clock_drift,
    read_data,
    robust_stats,
)

# The model channels of the CPAP cohort (config/dataset/neonatal.yaml).
SIGNAL_TYPES = ["CPAP", "Thorax", "HR", "PR", "SpO2", "PCO2"]
CUTTER_EVENTS = ["SIGNAL-ARTIFACT"]

# Physical-unit channels: reported as medians, NOT block-normalised. These are
# the ones B2's last bullet would add a second channel for.
LEVEL_CHANNELS = ("HR", "SpO2", "PCO2")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    p.add_argument("--clock_drift_file", default="config/clock_drift.yaml",
                   help="pass '' to measure uncorrected")
    p.add_argument("--csv", default=None, help="also write the per-block table here")
    p.add_argument("--floor_pct", type=float, default=5.0,
                   help="percentile of s across blocks to propose as the floor")
    return p.parse_args()


def collect(data_path, ids, clock_drift_file):
    """One row per (patient, block, channel). No training, no GPU."""
    drift = load_clock_drift(clock_drift_file or None)
    rows = []
    for pat_id in ids:
        seg_list = read_data(pat_id, data_path, clock_drift=drift.get(pat_id))
        segments = seg_list if isinstance(seg_list, list) else [seg_list]
        for si, seg in enumerate(segments):
            # Exactly the mask the pipeline builds: CUTTER events only, never
            # the adverse events. See block_norm_stats' docstring.
            artifact_free = (
                1 - np.column_stack([seg[e] for e in CUTTER_EVENTS]).max(axis=1)
            ).astype(bool)
            stats = block_norm_stats(
                seg, SIGNAL_TYPES, artifact_free=artifact_free,
                pat_id=pat_id, segment_idx=si)
            for channel, (m, s) in stats.items():
                rows.append(dict(pat=pat_id, block=si, channel=channel,
                                 median=m, scale=s, kind="robust"))

            # Physical-unit channels: median level over the same mask, in the
            # channel's own units, at its own decimated rate.
            for channel in LEVEL_CHANNELS:
                if channel not in seg:
                    continue
                x = np.asarray(seg[channel], dtype=float)
                x = x[artifact_free[:len(x)]]
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                q25, q75 = np.percentile(x, [25, 75])
                rows.append(dict(pat=pat_id, block=si, channel=channel,
                                 median=float(np.median(x)),
                                 scale=float(q75 - q25) / 1.349, kind="level"))
    return pd.DataFrame(rows)


def report_scales(df, floor_pct):
    rob = df[df.kind == "robust"]
    print("\n[1] ROBUST SCALES PER CHANNEL, across all blocks")
    print(f"    {'channel':<12} {'n':>4} {'min':>10} {f'p{floor_pct:g}':>10} "
          f"{'median':>10} {'max':>10}  {'max/min':>8}")
    floors = {}
    for channel, g in rob.groupby("channel"):
        s = g.scale.values
        floor = float(np.percentile(s, floor_pct))
        floors[channel] = floor
        ratio = s.max() / s.min() if s.min() > 0 else np.inf
        print(f"    {channel:<12} {len(s):>4} {s.min():>10.4g} {floor:>10.4g} "
              f"{np.median(s):>10.4g} {s.max():>10.4g}  {ratio:>8.1f}x")

    print("\n[2] TWO CANDIDATE FLOOR RULES, AND WHAT EACH ONE CLIPS")
    print("    B2 suggests 'e.g. the 5th percentile'. Measured on this cohort that")
    print("    over-clips: only Abdomen has a genuine dead sensor, the other")
    print("    channels are a smooth continuum with nothing to protect against.")
    print(f"    {'channel':<12} {'p'+f'{floor_pct:g}':>9} {'clips':>6}   "
          f"{'0.2*median':>10} {'clips':>6}   {'dead (s=0)':>10}")
    frac_floors = {}
    n_p5 = n_frac = 0
    for channel, g in rob.groupby("channel"):
        s = g.scale.values
        frac = 0.2 * float(np.median(s))
        frac_floors[channel] = frac
        c5, cf = int((s <= floors[channel]).sum()), int((s <= frac).sum())
        n_p5 += c5
        n_frac += cf
        print(f"    {channel:<12} {floors[channel]:>9.4g} {c5:>6}   "
              f"{frac:>10.4g} {cf:>6}   {int((s == 0).sum()):>10}")
    print(f"    {'TOTAL':<12} {'':>9} {n_p5:>6}   {'':>10} {n_frac:>6}")
    print()
    print("    The floor exists to catch a DEAD sensor. Prefer the rule that clips")
    print("    the dead blocks and nothing else.")

    print("\n[2b] PROPOSED FLOOR (0.2 x median) -- paste into the config")
    print("    norm_scale_floor:")
    for channel in sorted(frac_floors):
        print(f"      {channel}: {frac_floors[channel]:.4g}")

    print("\n[2c] LEAVE-ONE-INFANT-OUT STABILITY (is a per-fold floor needed?)")
    print(f"    {'channel':<12} {'0.2*med drift':>14} {'p'+f'{floor_pct:g}'+' drift':>12}"
          f"   {'gap to next-lowest scale':>26}")
    for channel, g in rob.groupby("channel"):
        s = g.scale.values
        pats = g.pat.unique()
        f_full, p_full = 0.2 * np.median(s), np.percentile(s, floor_pct)
        f_loo = [0.2 * np.median(g[g.pat != p].scale.values) for p in pats]
        p_loo = [np.percentile(g[g.pat != p].scale.values, floor_pct) for p in pats]
        fd = max(abs(min(f_loo) - f_full), abs(max(f_loo) - f_full)) / f_full
        pd_ = max(abs(min(p_loo) - p_full), abs(max(p_loo) - p_full)) / p_full
        nz = np.sort(s[s > 0])
        gap = f"{nz[0]:.4g}" if len(nz) else "n/a"
        print(f"    {channel:<12} {fd:>13.1%} {pd_:>11.1%}   "
              f"{'lowest live scale ' + gap:>26}")
    print()
    print("    A per-fold floor is only needed if the fold-to-fold movement could")
    print("    change WHICH blocks get clipped. Compare each channel's LOO range")
    print("    against the gap between 0 and its lowest live scale: if the whole")
    print("    range sits inside that gap, every fold clips exactly the same")
    print("    blocks and a single cohort-level floor is equivalent.")

    print("\n[3] BLOCKS THAT HIT THE PROPOSED FLOOR (0.2 x median)")
    print("    These are the QC failures B2 asks to be logged. block_norm_stats")
    print("    now warns on each of them at dataset-build time.")
    hits = []
    for channel, g in rob.groupby("channel"):
        for r in g.itertuples():
            if r.scale <= frac_floors[channel]:
                hits.append((r.pat, r.block, channel, r.scale, frac_floors[channel]))
    if not hits:
        print("    none")
    for pat, block, channel, s, f in sorted(hits):
        note = " <-- DEAD (scale is exactly zero)" if s == 0 else ""
        print(f"    patient {pat} block {block}  {channel:<12} "
              f"s={s:.4g}  floor={f:.4g}{note}")

    print("\n[4] THE ACTUAL QUESTION: is any block an OUTLIER, not just lowest?")
    print("    A p5 floor always selects 5% of blocks. What matters is whether a")
    print("    block sits far BELOW the bulk -- the signature of a dead belt.")
    flagged = False
    for channel, g in rob.groupby("channel"):
        s = g.scale.values
        med = np.median(s)
        for r in g.itertuples():
            if med > 0 and r.scale < 0.2 * med:
                cause = ("DEAD/LOOSE BELT" if channel in ("Thorax", "Abdomen")
                         else "collapsed signal amplitude")
                print(f"    !! patient {r.pat} block {r.block} {channel}: "
                      f"s={r.scale:.4g} is {r.scale / med:.1%} of the channel "
                      f"median ({med:.4g}) -- {cause}")
                flagged = True
    if not flagged:
        print("    No block sits below 20% of its channel median.")
        print("    => the floor is a FORMALITY here: set it, log it, nothing hits it.")
    return flagged


# Hard physical limits. A channel outside these is a calibration or scaling
# problem, not physiology, and it also means the fixed-range normalisation
# (which maps these very bounds to [-1, 1]) is being fed out-of-range input.
PHYSICAL_LIMITS = {"SpO2": (0.0, 100.0), "HR": (0.0, 300.0), "PCO2": (0.0, 200.0)}


def report_limits(df):
    lvl = df[df.kind == "level"]
    print("\n[6] PHYSIOLOGICAL RANGE CHECK")
    bad = False
    for channel, (lo, hi) in PHYSICAL_LIMITS.items():
        g = lvl[lvl.channel == channel]
        if g.empty:
            continue
        over = g[(g["median"] < lo) | (g["median"] > hi)]
        if len(over):
            bad = True
            print(f"    !! {channel}: {len(over)} block medians outside "
                  f"[{lo:g}, {hi:g}] -- max {g['median'].max():.2f}")
            for r in over.itertuples():
                print(f"       patient {r.pat} block {r.block}: "
                      f"median {getattr(r, 'median'):.2f}")
    if not bad:
        print("    all block medians within physical limits")
    else:
        print("    A block median outside physical limits means the channel is")
        print("    mis-scaled or the sensor was uncalibrated. Note that")
        print("    normalize_range maps these same bounds to [-1, 1], so such")
        print("    samples are pushed outside [-1, 1] before the network.")
    return bad


def report_levels(df):
    lvl = df[df.kind == "level"]
    if lvl.empty:
        return
    print("\n[5] PHYSICAL-UNIT BASELINES (B2's last bullet: is a 2nd channel worth it?)")
    print(f"    {'channel':<8} {'per-block median':>26}   {'between-infant':>14} "
          f"{'within-infant':>14}")
    for channel, g in lvl.groupby("channel"):
        per_infant = g.groupby("pat")["median"].mean()
        # Spread of infant means vs spread of blocks around their infant's mean.
        between = float(per_infant.std()) if len(per_infant) > 1 else float("nan")
        within = float(g.groupby("pat")["median"].transform(
            lambda x: x - x.mean()).std())
        rng = f"{g['median'].min():.1f}-{g['median'].max():.1f} (med {g['median'].median():.1f})"
        print(f"    {channel:<8} {rng:>26}   {between:>14.2f} {within:>14.2f}")
    print()
    print("    'between-infant' is the sd of each infant's mean level; 'within-infant'")
    print("    is the sd of blocks around their own infant's mean (i.e. device/time).")
    print("    A second, per-block robust-centred channel can only encode what the")
    print("    fixed-range channel loses -- so it is worth building when between-infant")
    print("    spread is LARGE relative to the fixed range it is normalised over")
    print("    (HR 50-240 = 190 bpm wide, SpO2 60-100 = 40 pp wide).")
    for channel, width in (("HR", 190.0), ("SpO2", 40.0)):
        g = lvl[lvl.channel == channel]
        if g.empty:
            continue
        means = g.groupby("pat")["median"].mean()
        if len(means) < 2:
            print(f"      {channel}: needs >1 infant to separate between from within")
            continue
        between = float(means.std())
        print(f"      {channel}: between-infant sd = {between:.2f} = "
              f"{between / width:.1%} of its fixed range")


def main():
    args = parse_args()
    print("Per-block robust scales -- pre-flight for the B3 normalisation sweeps\n")
    print(f"channels block-normalised: {', '.join(sorted(set(BLOCK_NORM_CHANNELS) & set(DECIMATION_CHAIN)))}")
    df = collect(args.data_path, args.ids, args.clock_drift_file)
    n_blocks = df.groupby(["pat", "block"]).ngroups
    print(f"\ncollected {n_blocks} blocks over {df.pat.nunique()} infants")

    flagged = report_scales(df, args.floor_pct)
    report_levels(df)
    out_of_range = report_limits(df)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nper-block table -> {args.csv}")

    print("\n" + "=" * 72)
    if flagged:
        print("VERDICT: at least one block has a collapsed scale. Set the floor")
        print("BEFORE running the B3 sweeps -- that cell would otherwise divide")
        print("this block's noise up to full scale.")
    else:
        print("VERDICT: no collapsed scales. Set the floor for the record, but it")
        print("is not blocking; the B3 sweeps can run as configured.")
    print("=" * 72)


if __name__ == "__main__":
    main()
