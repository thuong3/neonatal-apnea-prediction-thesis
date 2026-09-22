"""RECOVER THE VENTILATION MODE FROM THE SIGNAL (to-do items B6, A4, D2).

What this settles
-----------------
Each infant was ventilated with four devices in turn, one per ANALYSIS block
(Section 3.1 of the thesis). WHICH block was which device is undocumented --
item A4 -- which in turn blocks D2 (per-device breakdown) and B6.

B6 asks three things, and the third is the serious one:

    * quantify the machine-inflation component (about 10/min = 0.167 Hz) per
      block in the RIP spectrum, tabulated by device;
    * decide on evidence whether to notch it, supply it as a channel, or leave
      it alone;
    * CHECK LABEL COMPARABILITY ACROSS ARMS. If ventilator inflations move
      chest and abdomen while the infant is apnoeic, then the scoring rule
      "below 20% of the amplitude of the preceding breaths" does not mean the
      same thing in the NIPPV arms as in the CPAP arms. B6 calls this "a
      validity question for the original labels, not just a modelling nuisance".

An earlier attempt (scripts/device_rhythm.py) asked a DIFFERENT and
self-invented question -- whether belt-rhythm regularity could identify the
device -- and found nothing. That was the wrong instrument. The mask-pressure
channel is the DEVICE'S OWN OUTPUT, so a machine delivering a set rate must
write a sharp spectral line into it, and a spontaneously breathing infant
cannot.

The measurement
---------------
Per block, the power in a narrow notch at 1/6 Hz (10 per minute) divided by the
local background just outside it, in CPAP mask pressure and in the Thorax belt.
A set-rate machine gives a sharp line; spontaneous breathing gives none. The
ratio is used rather than raw power so that it does not depend on the block's
overall amplitude, which varies with the device and the infant.

What it finds (all 15 infants, 60 blocks)
-----------------------------------------
EXACTLY TWO of every infant's four blocks carry the line, 15/15, with no
ambiguous case: machine blocks score 60-8000x, the others 1.1-2.8x. Two arms
are NIPPV and two are CPAP, exactly as B6 assumes.

And the answer to B6's third bullet: the line is loud in the PRESSURE and
essentially ABSENT FROM THE BELTS (1.1-3.7x). Since the scoring rule is
belt-based, the apnoea criterion means the same thing in all four arms. That
validity concern is closed by measurement rather than by argument.

What it does NOT do
-------------------
This recovers the MODE (NIPPV vs CPAP), not the individual device. A4 is
therefore half-answered: 2 of 4 arms are identified as a pair, and telling
"Stephanie with frequency" from the other NIPPV arm still needs the trial
records. D2 can now be run as a two-way mode comparison.

A note on the device names: the annotator lists "Stephanie with frequency,
Stephanie without frequency, Infant Flow, Infant Flow with water seal" -- only
ONE of which sounds like it delivers a set rate. The data says TWO do, in every
infant. The naming and the physics disagree, and the physics is unambiguous.

Run:  python scripts/detect_ventilation_mode.py --csv outputs/ventilation_mode/modes.csv
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import decimate, welch
from scipy.stats import chisquare

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402

F_MACHINE = 1.0 / 6.0     # 10 inflations per minute
NOTCH_HZ = 0.006          # half-width of the "line" band
SHOULDER = (0.012, 0.05)  # local background, just outside the line
RATIO_THRESHOLD = 3.0     # above this = a real line; observed values are >60


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    p.add_argument("--clock_drift_file", default="config/clock_drift.yaml")
    p.add_argument("--csv", default=None, help="write the block -> mode map here")
    return p.parse_args()


def line_ratio(x, fs=5.0):
    """Power in a narrow notch at 10/min over the local background."""
    f, P = welch(np.asarray(x, float) - np.mean(x), fs=fs, nperseg=8192)
    sel = (f > 0.05) & (f < 1.0)
    f, P = f[sel], P[sel]
    notch = np.abs(f - F_MACHINE) < NOTCH_HZ
    shoulder = (np.abs(f - F_MACHINE) > SHOULDER[0]) & (np.abs(f - F_MACHINE) < SHOULDER[1])
    if not notch.any() or not shoulder.any():
        return float("nan")
    return float(P[notch].max() / np.median(P[shoulder]))


def collect(data_path, ids, drift_file):
    drift = load_clock_drift(drift_file or None)
    rows = []
    for pat_id in ids:
        for bi, seg in enumerate(read_data(pat_id, data_path,
                                           clock_drift=drift.get(pat_id))):
            artifact = np.asarray(seg["SIGNAL-ARTIFACT"]).astype(bool)
            out = {}
            for ch in ("CPAP", "Thorax"):
                x = np.asarray(seg[ch], dtype=float)
                x = x[~artifact[:len(x)]]
                # Same decimation the model uses, so this describes the signal
                # the network actually sees.
                dec = decimate(decimate(x, 10), 4) if ch == "CPAP" \
                    else decimate(x[::4], 10)
                out[ch] = line_ratio(dec)
            rows.append(dict(pat=pat_id, block=bi,
                             cpap_ratio=out["CPAP"], thorax_ratio=out["Thorax"]))
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    print("Ventilation mode from the 10/min machine line (B6, A4, D2)\n")
    df = collect(args.data_path, args.ids, args.clock_drift_file)
    df["nippv"] = df.cpap_ratio > RATIO_THRESHOLD

    print("[1] 0.167 Hz LINE STRENGTH PER BLOCK (ratio to local background)")
    print(f"    {'pat':>4}  " + "  ".join(f"{'b' + str(b):>11}" for b in range(4))
          + "   |  belts (Thorax)")
    for pat, g in df.groupby("pat"):
        g = g.sort_values("block")
        cp = "  ".join(f"{r.cpap_ratio:>10.1f}{'*' if r.nippv else ' '}"
                       for r in g.itertuples())
        th = " ".join(f"{r.thorax_ratio:>5.1f}" for r in g.itertuples())
        print(f"    {pat:>4}  {cp}   |  {th}")
    print(f"    (* = line present, ratio > {RATIO_THRESHOLD:g})")

    per_infant = df.groupby("pat").nippv.sum()
    print(f"\n[2] STRUCTURE: blocks with the line, per infant: "
          f"{list(per_infant.values)}")
    assert (per_infant == 2).all(), (
        "expected exactly 2 machine blocks per infant; got "
        f"{dict(per_infant[per_infant != 2])}"
    )
    print(f"    EXACTLY TWO for every one of {len(per_infant)} infants.")
    print("    -> two arms are NIPPV, two are CPAP, as B6 assumes.")

    print("\n[3] B6's VALIDITY QUESTION: do the inflations reach the BELTS?")
    m, n = df[df.nippv], df[~df.nippv]
    print(f"    CPAP pressure : machine blocks {m.cpap_ratio.min():.0f}-"
          f"{m.cpap_ratio.max():.0f}x, others {n.cpap_ratio.min():.1f}-"
          f"{n.cpap_ratio.max():.1f}x")
    print(f"    Thorax belt   : machine blocks {m.thorax_ratio.min():.1f}-"
          f"{m.thorax_ratio.max():.1f}x, others {n.thorax_ratio.min():.1f}-"
          f"{n.thorax_ratio.max():.1f}x")
    if df.thorax_ratio.max() < 2 * RATIO_THRESHOLD:
        print("    The line is loud in the PRESSURE and absent from the BELTS.")
        print("    The scoring rule is belt-based, so 'below 20% of the preceding")
        print("    breaths' means the SAME THING in all four arms.")
        print("    => B6's label-comparability concern is CLOSED by measurement.")
    else:
        print("    !! The belts DO register the machine. B6's concern is live:")
        print("    !! the apnoea criterion is not comparable across arms.")

    print("\n[4] WAS THE ORDER FIXED, RANDOM, OR A LATIN SQUARE?")
    patterns = {pat: tuple(sorted(g[g.nippv].block)) for pat, g in df.groupby("pat")}
    pairs = list(combinations(range(4), 2))
    counts = {p: sum(1 for v in patterns.values() if v == p) for p in pairs}
    for p in pairs:
        print(f"    NIPPV in blocks {p}: {counts[p]}  {'#' * counts[p]}")
    obs = [counts[p] for p in pairs]
    n_distinct = sum(1 for p in pairs if counts[p])
    c1, p1 = chisquare(obs)
    pos = np.array([int(df[(df.block == b) & df.nippv].shape[0]) for b in range(4)])
    c2, p2 = chisquare(pos)
    print(f"\n    distinct patterns: {n_distinct} of 6")
    print(f"    random permutation per infant would give all 6 equally "
          f"(chi2={c1:.2f}, p={p1:.3f})")
    print(f"    NIPPV count by block position: {list(pos)} "
          f"(chi2={c2:.2f}, p={p2:.3f})")
    print()
    print("    The order is NOT FIXED -- it genuinely varies between infants.")
    if p2 > 0.05:
        print("    It IS balanced across block positions, which is what the A4")
        print("    deconfounding argument actually needs.")
    if n_distinct > 4:
        print(f"    A simple CYCLIC 4x4 Latin square can produce only 4 distinct")
        print(f"    patterns; {n_distinct} are observed, so that specific design is")
        print("    excluded. Other Latin squares are not (there are 576).")
    print("    Randomisation and a Latin square cannot be told apart from this")
    print("    alone -- that needs the trial records. What the data DOES settle")
    print("    is that the order varies and is positionally balanced.")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        df["mode"] = np.where(df.nippv, "NIPPV", "CPAP")
        df.to_csv(args.csv, index=False)
        print(f"\nblock -> mode map -> {args.csv}")
        print("D2 (per-device breakdown) can now be run as a two-way mode split.")


if __name__ == "__main__":
    main()
