"""WHAT A SCORED DESATURATION ACTUALLY IS, IN EACH COHORT.

The two cohorts were scored twelve years apart under different rules, and the
rules disagree about what a desaturation is:

  * Robin sequence, per the 2020 AASM criteria used by Lim et al. and quoted in
    the supporting information of Vetter et al. (2024): "hypoxia" is an
    SpO2 decrease of at least 3 % within a 5-second duration.
  * Brainimmaturity, per Sievers (2008): "a desaturation event was defined as a
    fall in SpO2 to <= 80 %".

Section 4.2 compares how often a scored central apnea is followed by a scored
desaturation, and gets 81.4 % in Robin against 12.1 % here. If the two
annotation sets mean different things by "desaturation", part of that gap is
definitional. This script settles it from the recordings rather than from the
published criteria, by measuring what the annotated events look like in the
saturation trace itself.

FOR EACH SCORED DESATURATION IT MEASURES
  * the baseline, as the median saturation over the 30 s before the onset;
  * the nadir, as the minimum over the event and the 10 s following it;
  * the drop, baseline minus nadir.

WHAT WOULD CONFIRM THE PUBLISHED CRITERIA
    Under Sievers' rule nearly every event should reach 80 % or below. Under
    the AASM rule most events should fall by around 3 % or more while ending
    well above 80 %. The two are mutually exclusive for the great majority of
    events, so the recordings can decide between them.

Usage:
    python scripts/desat_criterion_check.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import read_data  # noqa: E402

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"
FS = 200                     # rate of the arrays read_data returns
BASELINE_S = 30              # window before onset used for the baseline
TAIL_S = 10                  # how far past the event end the nadir is sought

COHORTS = {
    "Brainimmaturity (Sievers 2008)": dict(
        path="data/"
             "dataset_brainimmaturity",
        sub="annotations", ids=["%03d" % i for i in range(1, 16)],
        record_duration=None),
    "Robin sequence (AASM 2020)": dict(
        path="data/robin_sequence/"
             "data/neonatal_robin_polysomnography",
        sub="annotations_original", ids=["%03d" % i for i in range(1, 20)],
        record_duration=10),
}


def annotations(path, sub, pid):
    f = os.path.join(path, sub, "annotations%s.txt" % pid)
    if not os.path.exists(f):
        return None
    a = pd.read_csv(f, sep="\t")
    a = a[a["type"] != "type"].copy()
    t0 = datetime.strptime(a["timestamp"].iloc[0], TS_FMT)
    a["t"] = [(datetime.strptime(s, TS_FMT) - t0).total_seconds()
              for s in a["timestamp"]]
    a["duration"] = pd.to_numeric(a["duration"], errors="coerce").fillna(0.0)
    return a


def measure(seg_spo2, t, dur, seg_start_s):
    """(baseline, nadir, drop) for one event, or None if out of range.

    At least half of the baseline window and half of the event window must
    carry a physiological value, the same coverage rule that
    `scripts/coupling_like_for_like.py` applies. A nadir taken from a handful
    of surviving samples either side of a dropout is not a measurement of the
    event, and the two cohorts differ in how much dropout they carry.
    """
    i0 = int(round((t - seg_start_s) * FS))
    i1 = int(round((t - seg_start_s + dur + TAIL_S) * FS))
    b0 = i0 - BASELINE_S * FS
    if b0 < 0 or i1 > len(seg_spo2) or i1 <= i0:
        return None
    base = seg_spo2[b0:i0]
    ev = seg_spo2[i0:i1]
    n_base, n_ev = len(base), len(ev)
    base = base[(base > 40) & (base <= 100)]      # drop dropouts and overshoot
    ev = ev[(ev > 40) & (ev <= 100)]
    if len(base) < 0.5 * n_base or len(ev) < 0.5 * n_ev:
        return None
    b = float(np.median(base))
    n = float(np.min(ev))
    return b, n, b - n


def main():
    for name, cfg in COHORTS.items():
        rows = []
        for pid in cfg["ids"]:
            a = annotations(cfg["path"], cfg["sub"], pid)
            if a is None:
                continue
            try:
                segments = read_data(pid, cfg["path"],
                                     annotations_dir=cfg["sub"],
                                     record_duration=cfg["record_duration"])
            except Exception as exc:                      # noqa: BLE001
                print(f"  {pid}: {exc}")
                continue
            # read_data returns one segment per ANALYSIS span, in order; the
            # annotation clock shares its origin with the first timestamp.
            starts = a.loc[a["type"] == "ANALYSIS-START", "t"].values
            des = a[a["type"] == "DESAT"]
            for seg, s0 in zip(segments, starts):
                if "SpO2" not in seg:
                    continue
                x = np.asarray(seg["SpO2"], dtype=float)
                span = len(x) / FS
                for _, e in des.iterrows():
                    if not (s0 <= e["t"] < s0 + span):
                        continue
                    m = measure(x, e["t"], float(e["duration"]), s0)
                    if m is not None:
                        rows.append(m)
        if not rows:
            print(f"\n{name}: no measurable events\n")
            continue
        r = np.asarray(rows)
        base, nadir, drop = r[:, 0], r[:, 1], r[:, 2]
        print(f"\n{name}  --  {len(r)} scored desaturations measured")
        print(f"  baseline SpO2 before onset   median {np.median(base):5.1f} %"
              f"   IQR {np.percentile(base, 25):.1f}-{np.percentile(base, 75):.1f}")
        print(f"  nadir during/after the event median {np.median(nadir):5.1f} %"
              f"   IQR {np.percentile(nadir, 25):.1f}-{np.percentile(nadir, 75):.1f}")
        print(f"  drop from baseline           median {np.median(drop):5.1f} pp"
              f"  IQR {np.percentile(drop, 25):.1f}-{np.percentile(drop, 75):.1f}")
        print(f"  reaching 80 % or below       {np.mean(nadir <= 80) * 100:5.1f} %"
              f"   <- Sievers' criterion")
        print(f"  falling by at least 3 points {np.mean(drop >= 3) * 100:5.1f} %"
              f"   <- the AASM criterion")


if __name__ == "__main__":
    main()
