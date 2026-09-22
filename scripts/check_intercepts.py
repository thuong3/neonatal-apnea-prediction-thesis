"""CHECK the four non-zero `intercept_s` values in config/clock_drift.yaml.

The rule those four were set by is stated in the config header and in
Section 3.3 of the thesis: apply the recording's slope, re-measure the lag
against the annotator's own rule (Thorax and Abdomen both flat for >= 10 s),
pool it over every apnea in the recording, and set the intercept only where that
pooled residual is at least a second AND flat across the recording. A residual
that still tilts means the SLOPE is wrong and should be refitted; a residual
that is large but level is a genuine constant offset, which is what an intercept
is for.

That quantity is not in any of the CSVs `measure_clock_drift.py` writes.
`belts_icpt` there is the intercept of the free Theil-Sen fit, estimated jointly
with the slope -- a different number, which is why 003 (1.41) and 005 (2.09)
look as though they should have qualified and did not. This script measures the
quantity the rule is actually about.

READ-ONLY: it writes no file and does not touch config/clock_drift.yaml.

    python scripts/check_intercepts.py [--data_path ...] [--ids 003 005 009]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_clock_drift as mcd   # noqa: E402

# The rule, as stated in the config header. Both clauses must hold.
MIN_ABS_RESID_S = 1.0     # below this, fitting a constant fits the estimator's noise
MAX_THIRD_SPREAD_S = 1.0  # "flat across the recording", measured between thirds


def pooled_residual(bins):
    """Event-weighted median lag, and the spread of that lag between thirds.

    `bins` are the `belts` rows for one recording, measured with the slope
    already applied, so each `lag_s` is what the slope left behind.
    """
    good = bins[bins["peak_z"] >= mcd.MIN_PEAK_Z]
    if len(good) < mcd.MIN_BINS_FOR_FIT:
        return None

    lag = good["lag_s"].to_numpy(float)
    n = good["n_events"].to_numpy(float)
    order = np.argsort(lag)
    cum = np.cumsum(n[order])
    pooled = float(lag[order][np.searchsorted(cum, cum[-1] / 2.0)])

    t = good["t_centre"].to_numpy(float)
    edges = np.quantile(t, [0.0, 1 / 3, 2 / 3, 1.0])
    thirds = [lag[(t >= lo) & (t <= hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    medians = [float(np.median(v)) for v in thirds if len(v)]
    spread = max(medians) - min(medians) if len(medians) > 1 else float("nan")
    return dict(pooled_s=pooled, third_spread_s=spread, n_bins=len(good),
                n_events=int(good["n_events"].sum()), thirds=medians)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", default="data/"
                                          "Preprocessing BA/dataset_brainimmaturity")
    p.add_argument("--yaml", default=os.path.join(mcd.ROOT, "config/clock_drift.yaml"))
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    args = p.parse_args()

    params = mcd.load_params(args.yaml)
    print(f"slope from {args.yaml}, intercept forced to 0 so the residual it was "
          f"measured from is visible.\n")

    rows = []
    for pat_id in args.ids:
        slope_ppm, applied_icpt = params.get(pat_id, (0.0, 0.0))
        # Slope only. With the intercept applied the residual is what the
        # intercept already removed, which answers nothing.
        bins = pd.DataFrame(mcd.measure_patient(pat_id, args.data_path,
                                                (slope_ppm, 0.0)))
        belts = bins[bins["anchor"] == "belts"] if len(bins) else bins
        got = pooled_residual(belts) if len(belts) else None
        if got is None:
            rows.append(dict(patient=pat_id, applied=applied_icpt, pooled=np.nan,
                             spread=np.nan, n_bins=0, n_events=0, verdict="no belts fit"))
            continue

        qualifies = (abs(got["pooled_s"]) >= MIN_ABS_RESID_S
                     and got["third_spread_s"] <= MAX_THIRD_SPREAD_S)
        rows.append(dict(patient=pat_id, applied=applied_icpt,
                         pooled=got["pooled_s"], spread=got["third_spread_s"],
                         n_bins=got["n_bins"], n_events=got["n_events"],
                         verdict="intercept" if qualifies else "leave at 0"))

    out = pd.DataFrame(rows)
    out["agrees"] = [
        ("-" if r.verdict == "no belts fit" else
         "yes" if (r.verdict == "intercept") == (abs(r.applied) > 0) else "NO")
        for r in out.itertuples(index=False)]

    pd.set_option("display.width", 200)
    print("\npooled residual against the annotator's rule, slope applied, "
          "intercept not:")
    print(out.round(2).to_string(index=False))

    print(f"\nrule: |pooled| >= {MIN_ABS_RESID_S} s AND spread between thirds "
          f"<= {MAX_THIRD_SPREAD_S} s")
    disagree = out[out["agrees"] == "NO"]
    if len(disagree):
        print("\nRECORDINGS WHERE THE CONFIG AND THE RULE DISAGREE:")
        print(disagree.round(2).to_string(index=False))
        print("\nEither the rule as documented is not the rule that was applied, "
              "or the config needs updating. Do not change the config from this "
              "script's output alone -- read the per-third medians first.")
    else:
        print("\nEvery recording's config intercept agrees with the documented "
              "rule. The four non-zero values are reproduced, and the eleven "
              "zeros are justified.")


if __name__ == "__main__":
    main()
