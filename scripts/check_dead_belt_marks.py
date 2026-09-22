"""ARE THE MARKS SCORED ON A DEAD BELT REAL EVENTS? (Section 3.4 of the thesis)

The problem
-----------
The annotator's rule was "Thorax and Abdomen simultaneously flat for >=10 s". A
belt that has stopped recording is also flat. So for any mark scored while one
belt was dead, the rule cannot distinguish a real central apnea from a sensor
failure -- the evidence for the mark is exactly the thing that broke.

6.3a originally drew the pessimistic conclusion from that: patient 009 has 57% of
its marks on a dead Abdomen belt, so 009 was called "doubtful on its own terms"
and flagged for possible exclusion.

Why that reasoning was wrong
----------------------------
It treats an UNFALSIFIABLE mark as a FALSE one. Those are not the same. The rule
cannot confirm the mark, but a signal that has nothing to do with the belts can:
a real central apnea is followed by a desaturation, and a flat sensor is not.

So this script ignores the belts as evidence and asks SpO2 instead. For every
mark with a clean window it measures the drop from a pre-event baseline
(-60..-30 s) to the nadir after onset (0..+45 s), then splits the marks by
whether the Abdomen belt was dead AT THAT MARK -- 6.3a's own criterion,
peak-to-peak below 2% of that recording's median 10 s belt amplitude.

If the dead-belt marks were sensor artefacts they would desaturate at background
rate. They do the opposite: 17 of 18 of 009's reach a >=3% desaturation, against
70% cohort-wide. The marks are real, and 6.3a's exclusion advice is withdrawn.

Reading the output
------------------
The load-bearing number is the desaturation RATE of the dead-belt marks on their
own. The live-vs-dead contrast within 009 rests on only 8 live marks and is
underpowered -- do not lean on it.

Run:  python scripts/check_dead_belt_marks.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    TARGET_FREQ,
    load_clock_drift,
    read_data,
)

BASE_FROM, BASE_TO = 60, 30   # s before onset: baseline window
NADIR_TO = 45                 # s after onset: search for the nadir
AMP_WIN_S = 10                # s: window for the belt-amplitude reference
DEAD_FRAC = 0.02              # 6.3a's criterion: ptp below 2% of the median
DESAT_LEVELS = (3.0, 4.0)     # % drop counted as a desaturation


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    p.add_argument("--clock_drift_file", default="config/clock_drift.yaml")
    p.add_argument("--focus", default="009", help="the patient 6.3a singled out")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def belt_reference(x, win):
    """Median peak-to-peak over consecutive `win`-sample windows."""
    n = len(x) // win
    if n == 0:
        return np.nan
    return float(np.median([np.ptp(x[i * win:(i + 1) * win]) for i in range(n)]))


def collect(data_path, ids, clock_drift_file):
    drift = load_clock_drift(clock_drift_file or None)
    win = AMP_WIN_S * TARGET_FREQ
    pre, post = BASE_FROM * TARGET_FREQ, NADIR_TO * TARGET_FREQ
    rows = []
    for pat_id in ids:
        for si, seg in enumerate(read_data(pat_id, data_path,
                                           clock_drift=drift.get(pat_id))):
            apnea = np.asarray(seg["APNEA-CENTRAL"]).astype(int)
            artifact = np.asarray(seg["SIGNAL-ARTIFACT"])
            spo2 = np.asarray(seg["SpO2"], dtype=float)
            belts = {c: np.asarray(seg[c], dtype=float)
                     for c in ("Thorax", "Abdomen")}
            ref = {c: belt_reference(x, win) for c, x in belts.items()}

            starts = np.flatnonzero(np.diff(apnea) == 1) + 1
            ends = np.flatnonzero(np.diff(apnea) == -1) + 1
            for s0 in starts:
                later = ends[ends > s0]
                if not len(later):
                    continue
                e0 = later[0]
                if s0 - pre < 0 or s0 + post > len(apnea):
                    continue
                # A mark whose window overlaps a scored artefact tells us
                # nothing either way -- the SpO2 there is not trustworthy.
                if artifact[s0 - pre:s0 + post].any():
                    continue
                base = np.median(spo2[s0 - pre:s0 - BASE_TO * TARGET_FREQ])
                row = dict(pat=pat_id, block=si,
                           drop=float(base - spo2[s0:s0 + post].min()))
                for c in ("Thorax", "Abdomen"):
                    row[c + "_dead"] = (
                        bool(np.ptp(belts[c][s0:e0]) < DEAD_FRAC * ref[c])
                        if ref[c] and ref[c] > 0 else True
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def line(label, g):
    if not len(g):
        print(f"    {label:<32} {'--':>5}")
        return
    rates = "  ".join(f"{(g['drop'] >= lv).mean():>9.0%}" for lv in DESAT_LEVELS)
    print(f"    {label:<32} {len(g):>5} {g['drop'].median():>12.2f}%{rates}")


def main():
    args = parse_args()
    print("Are dead-belt marks real events? -- Section 3.4 of the thesis\n")
    df = collect(args.data_path, args.ids, args.clock_drift_file)
    if df.empty:
        raise SystemExit("no usable marks")

    header = "  ".join(f">={lv:g}% desat" for lv in DESAT_LEVELS)
    print(f"{len(df)} apneas with a clean {BASE_FROM} s pre / {NADIR_TO} s post window\n")
    print(f"[1] SpO2 DROP AFTER ONSET (baseline -{BASE_FROM}..-{BASE_TO} s, "
          f"nadir 0..+{NADIR_TO} s)")
    print(f"    {'group':<32} {'n':>5} {'median drop':>13}  {header}")
    line("whole cohort", df)
    line(f"cohort excluding {args.focus}", df[df.pat != args.focus])
    line(f"patient {args.focus}, all marks", df[df.pat == args.focus])

    focus = df[df.pat == args.focus]
    print(f"\n[2] THE QUESTION: {args.focus}'s marks split by belt state at the mark")
    print(f"    {'group':<32} {'n':>5} {'median drop':>13}  {header}")
    line("Abdomen live", focus[~focus.Abdomen_dead])
    line("Abdomen DEAD", focus[focus.Abdomen_dead])

    print("\n[3] SAME SPLIT COHORT-WIDE, as a control")
    print(f"    {'group':<32} {'n':>5} {'median drop':>13}  {header}")
    line("Abdomen live", df[~df.Abdomen_dead])
    line("Abdomen DEAD", df[df.Abdomen_dead])

    print("\n[4] MARKS ON A DEAD ABDOMEN BELT, BY PATIENT")
    t = df.groupby("pat").Abdomen_dead.agg(["sum", "count"])
    t = t[t["sum"] > 0]
    for pat, r in t.iterrows():
        print(f"    {pat}: {int(r['sum'])}/{int(r['count'])} "
              f"({r['sum'] / r['count']:.1%})")

    dead = df[df.Abdomen_dead]
    rate = (dead["drop"] >= DESAT_LEVELS[0]).mean() if len(dead) else 0.0
    cohort = (df["drop"] >= DESAT_LEVELS[0]).mean()
    print("\n[5] VERDICT")
    print(f"    dead-belt marks reaching >={DESAT_LEVELS[0]:g}% desaturation: {rate:.0%}")
    print(f"    cohort baseline rate:                            {cohort:.0%}")
    if rate >= cohort:
        print("    Dead-belt marks desaturate at least as reliably as the cohort.")
        print("    They are REAL events -- an unfalsifiable mark is not a false one.")
        print("    6.3a's exclusion advice is withdrawn; no data is excluded.")
    else:
        print("    !! Dead-belt marks desaturate LESS than the cohort. Re-open 6.3a.")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nper-mark table -> {args.csv}")


if __name__ == "__main__":
    main()
