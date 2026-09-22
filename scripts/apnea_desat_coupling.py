"""HOW OFTEN IS AN APNEA FOLLOWED BY A DESATURATION? -- both cohorts, by subtype.

Poets (Sleep Med 2010) describes apnea and desaturation as near-inseparable in
untreated preterm infants: the interval from apnea onset to desaturation onset
has a median of 0.8 s. In the CPAP cohort only 12 % of scored central apneas are
followed by a scored DESAT. Two readings compete:

  (i)  CPAP works. It raises lung volume, which stabilises oxygenation, so an
       apnea on a well-recruited lung need not desaturate (Poets section 4).
       Under this reading the 12 % is the treatment, and it explains a great deal
       about why prediction is hard in this cohort.
  (ii) The DESAT scoring in the CPAP export is simply more conservative, and the
       number is an artefact of how it was annotated.

The Robin-sequence cohort separates them. It is UNTREATED, scored by the same
group, and it carries all three apnea subtypes. If reading (i) is right, Robin
should show a much higher coupling than the CPAP cohort. If (ii) is right, the
two cohorts should look similar and the CPAP number says nothing about CPAP.

Robin additionally allows the comparison Poets' section 2 makes: obstructive
events are expected to couple to desaturation MORE strongly than central ones,
because a central apnea can terminate before the saturation has time to fall.

Annotation-only: no EDF is read, so this runs in seconds.

Usage:
    python scripts/apnea_desat_coupling.py
    python scripts/apnea_desat_coupling.py --link_s 10 20 30
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"
APNEA_TYPES = ["APNEA-CENTRAL", "APNEA-OBSTRUCTIVE", "APNEA-MIXED"]

COHORTS = {
    "CPAP (treated, ours)": dict(
        path="data/brainimmaturity",
        sub="annotations", ids=["%03d" % i for i in range(1, 16)]),
    "Robin sequence (untreated)": dict(
        path="data/robin_sequence/"
             "data/neonatal_robin_polysomnography",
        sub="annotations_original", ids=["%03d" % i for i in range(1, 20)]),
}


def load(path, sub, pid):
    f = os.path.join(path, sub, "annotations%s.txt" % pid)
    if not os.path.exists(f):
        return None
    a = pd.read_csv(f, sep="\t")
    a = a[a["type"] != "type"]
    t0 = datetime.strptime(a["timestamp"].iloc[0], TS_FMT)
    a = a.assign(t=[(datetime.strptime(s, TS_FMT) - t0).total_seconds()
                    for s in a["timestamp"]])
    return a


def coupling(a, apnea_type, link_s):
    """(n apneas, fraction followed by a DESAT onset within link_s)."""
    ap = a.loc[a["type"] == apnea_type, "t"].values
    ds = a.loc[a["type"] == "DESAT", "t"].values
    if len(ap) == 0:
        return 0, np.nan
    if len(ds) == 0:
        return len(ap), 0.0
    hit = [bool(((ds >= t) & (ds <= t + link_s)).any()) for t in ap]
    return len(ap), float(np.mean(hit))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--link_s", nargs="*", type=float, default=[20.0])
    args = ap_.parse_args()

    for link_s in args.link_s:
        print("=" * 78)
        print("DESAT beginning within %.0f s of an apnea onset" % link_s)
        print("=" * 78)
        for name, cfg in COHORTS.items():
            rows = []
            for pid in cfg["ids"]:
                a = load(cfg["path"], cfg["sub"], pid)
                if a is None:
                    continue
                r = {"patient": pid, "n_desat": int((a["type"] == "DESAT").sum())}
                for t in APNEA_TYPES:
                    n, frac = coupling(a, t, link_s)
                    r["n_" + t.split("-")[1][:4].lower()] = n
                    r["pct_" + t.split("-")[1][:4].lower()] = (
                        np.nan if np.isnan(frac) else round(100 * frac, 1))
                rows.append(r)
            df = pd.DataFrame(rows)
            print("\n%s  (n=%d recordings)" % (name, len(df)))
            print(df.to_string(index=False))
            # cohort-level: pooled over all events, not the mean of percentages
            print("  pooled over all events:")
            for t in APNEA_TYPES:
                k = t.split("-")[1][:4].lower()
                tot = df["n_" + k].sum()
                if tot == 0:
                    print("    %-20s none scored" % t)
                    continue
                wsum = (df["n_" + k] * df["pct_" + k].fillna(0)).sum() / tot
                print("    %-20s %5d events, %5.1f %% followed by a DESAT"
                      % (t, tot, wsum))
            print()


if __name__ == "__main__":
    main()
