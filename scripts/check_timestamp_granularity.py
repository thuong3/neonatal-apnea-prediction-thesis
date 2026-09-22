"""MEASURE THE TIMESTAMP GRANULARITY OF THE SCORING (to-do list item A3).

The question
------------
A3 asks whether events were marked to the second or on fixed epochs. It matters
because it puts a floor under the usable prediction horizon: if marks had been
snapped to a 30 s grid, a 15 s offset would sit inside the quantisation error
and the prediction task would be measuring nothing.

Until this script existed, A3 was answered INDIRECTLY, by pointing at the
clock-drift residual in Section 3.3 of the thesis (mean absolute 0.29 s) and
arguing that a sub-second residual implies sub-second marks. That argument runs
the wrong way round: the residual is a property of a fit that was made TO these
marks, against an anchor derived from the signal. It is evidence about the drift
correction, not about the annotator's software. A3 asks for a property of the
scoring FILE, so this script measures the file and nothing else.

The test
--------
If a scorer works on a grid of spacing g, every onset lands near a multiple of
g. Map each onset onto the circle,

    theta = 2*pi * (t mod g) / g

and take the RESULTANT LENGTH R of those angles (the Rayleigh statistic):

    R = |mean(exp(i*theta))|     0 = perfectly uniform, 1 = all on one phase

R is the right statistic here for one specific reason: it is INVARIANT TO THE
CHOICE OF TIME ORIGIN. Shifting the origin rotates every angle by the same
amount, which moves the mean vector but not its length. So the test does not
need to know when the recorder started, where the scorer's grid was anchored, or
whether the ANALYSIS-START mark is itself on the grid -- a grid at ANY phase is
detected. A plain "distance to the nearest multiple of g" test lacks that
property and can be fooled by a grid offset by g/4.

Under the null (no grid) R is about sqrt(pi/(4n)); the 0.1% critical value is
sqrt(-ln(0.001)/n). Both are printed, so the margin is visible rather than
hidden behind a verdict string.

Why durations get a DIFFERENT test
----------------------------------
The fold test is only a grid test when the variable is spread broadly compared
to g. Onsets satisfy that overwhelmingly: they scatter over hours against grids
of at most 60 s. Durations do not -- apnea durations run 3.8-69.6 s with a
median of 11.5 s, so at g = 1 s the fold is sampling the shape of the duration
distribution itself, and the analytic Rayleigh null (which assumes a broad
underlying density) reads that curvature as a grid. Applying it here reports
APNEA-CENTRAL as "quantised" at 0.5 s and 1.0 s while only 1.12% of those
durations are actually whole seconds, against 1% by chance. That is the test
failing, not a finding.

So durations are tested directly instead, by the fraction landing exactly on a
whole second, against a binomial null. This is immune to the shape of the
distribution and is what "rounded" means in plain terms anyway.

That test yields a POSITIVE CONTROL, which is the reason durations are reported
at all. SIGNAL-ARTIFACT spans really are rounded -- 19.5% end on a whole second
-- while APNEA-CENTRAL (1.12%) and DESAT (0.82%) sit at chance. The scorer did
round, when dragging an artefact region out by hand over minutes, and the method
detects it. That makes the null result on onsets a genuine negative rather than
a test too blunt to see anything, which is the objection a reader would
otherwise raise against a table of non-findings.

Artefact-span rounding does not touch A3's conclusion: those spans are used as a
mask, and the prediction horizon is measured from event ONSETS.

Run:  python scripts/check_timestamp_granularity.py
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Grids a polysomnography scorer might plausibly have worked on. 30 s is the
# classic AASM sleep-staging epoch and the one A3 is really asking about; the
# short spacings would betray a scorer clicking on a decimated display.
ONSET_GRIDS = (0.005, 0.01, 0.02, 1 / 30, 0.1, 0.5, 1.0, 2.0,
               5.0, 10.0, 15.0, 20.0, 30.0, 60.0)
# Durations whose rounding would be a scoring decision. SIGNAL-ARTIFACT is
# deliberately not here: it is the positive control, not a finding.
PHYSIOLOGICAL_TYPES = ("APNEA-CENTRAL", "APNEA", "DESAT")

# Marks carrying a scored onset. ANALYSIS-START/STOP are excluded from this
# view but kept in the pooled one: they are placed by the same operator in the
# same software, so they probe the same grid.
EVENT_TYPES = ("APNEA-CENTRAL", "APNEA", "DESAT", "SIGNAL-ARTIFACT")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--annotations_dir", default="annotations")
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    return p.parse_args()


def load_annotations(data_path, annotations_dir, ids):
    """One row per mark, with `rel` = seconds after that recording's first mark.

    The origin is arbitrary on purpose -- see the docstring: the resultant
    length does not depend on it. A per-recording origin only keeps the numbers
    small enough that float64 holds full sub-millisecond precision, which a raw
    2005 POSIX timestamp (~1.1e9 s) would not.
    """
    rows = []
    for pat_id in ids:
        path = os.path.join(data_path, annotations_dir, f"annotations{pat_id}.txt")
        if not os.path.exists(path):
            path = os.path.join(data_path, f"annotations{pat_id}.txt")
        if not os.path.exists(path):
            print(f"    (no annotation file for {pat_id}, skipped)")
            continue
        anno = pd.read_csv(path, sep="\t")
        t = np.array([
            datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
            for s in anno["timestamp"]
        ])
        for typ, dur, rel in zip(anno["type"], anno["duration"], t - t[0]):
            rows.append((pat_id, typ, float(dur), float(rel)))
    return pd.DataFrame(rows, columns=["pid", "type", "dur", "rel"])


def resultant_length(values, grid):
    """Rayleigh R of `values` folded onto `grid`. 0 = uniform, 1 = one phase."""
    theta = 2.0 * np.pi * (np.asarray(values, dtype=float) % grid) / grid
    return float(abs(np.mean(np.exp(1j * theta))))


def grid_table(values, grids, label):
    """Print R per candidate grid; return the grids that look quantised."""
    n = len(values)
    expected = np.sqrt(np.pi / (4.0 * n))       # E[R] under the uniform null
    critical = np.sqrt(-np.log(0.001) / n)      # 0.1% Rayleigh critical value
    print(f"  {label} (n = {n})")
    print(f"    null: E[R] = {expected:.4f}, 0.1% critical value R = {critical:.4f}")
    print(f"    {'grid (s)':>10}  {'R':>8}   verdict")
    hits = []
    for g in grids:
        r = resultant_length(values, g)
        if r > critical:
            hits.append(g)
        print(f"    {g:>10.5f}  {r:>8.4f}   {'QUANTISED' if r > critical else 'uniform'}")
    return hits


def main():
    args = parse_args()
    print("A3 -- timestamp granularity of the scoring\n")
    df = load_annotations(args.data_path, args.annotations_dir, args.ids)
    if df.empty:
        raise SystemExit(f"no annotations found under {args.data_path}")

    print(f"{len(df)} marks over {df.pid.nunique()} recordings")
    for typ, count in df.type.value_counts().items():
        print(f"    {typ:<18} {count}")
    print()

    print("[1] ONSET grid")
    scored = df[df.type.isin(EVENT_TYPES)]
    onset_hits = grid_table(df.rel.values, ONSET_GRIDS, "all marks")
    print()
    onset_hits += grid_table(scored.rel.values, ONSET_GRIDS, "scored events only")
    print()

    print("[2] ONSET fractional-second part (scored events)")
    frac = scored.rel.values % 1.0
    print(f"    mean {frac.mean():.4f} (uniform -> 0.5000), "
          f"sd {frac.std():.4f} (uniform -> 0.2887)")
    print("    deciles " + " ".join(
        f"{q:.3f}" for q in np.percentile(frac, np.arange(0, 101, 10))))
    print()

    print("[3] DURATION rounding, per event type (positive control)")
    print(f"    {'type':<18} {'n':>6}  {'whole second':>12}  {'p (binomial)':>13}")
    dur_hits = []
    for typ, group in df[df.dur > 0].groupby("type"):
        cents = np.round(group.dur.values * 100).astype(int)
        n, k = len(cents), int(np.sum(cents % 100 == 0))
        # One-sided: is a whole-second duration more common than the 1-in-100
        # the two-decimal format gives by chance?
        p = float(binomtest(k, n, 0.01, alternative="greater").pvalue)
        if typ in PHYSIOLOGICAL_TYPES and p < 0.001:
            dur_hits.append((typ, k / n))
        print(f"    {typ:<18} {n:>6}  {k / n * 100:>11.2f}%  {p:>13.2e}")
    print("    chance = 1.00% (the two-decimal text format)")
    print()
    print("    SIGNAL-ARTIFACT spans are dragged out by hand over minutes and the")
    print("    scorer rounded them; the physiological events sit at chance. The")
    print("    method therefore does detect rounding where rounding exists, which")
    print("    is what makes the onset result above a real negative.")
    apnea = df[df.type.isin(("APNEA-CENTRAL", "APNEA"))]
    print(f"    apnea durations: {apnea.dur.nunique()} distinct values among "
          f"{len(apnea)} events, {apnea.dur.min():.2f}-{apnea.dur.max():.2f} s "
          f"(median {apnea.dur.median():.2f})")
    print()

    print("[4] VERDICT")
    assert not onset_hits, (
        f"onsets are quantised at {onset_hits} s -- A3's conclusion in "
        "Section 3.3 of the thesis is WRONG and the horizon argument in C2 "
        "must be redone against that grid"
    )
    assert not dur_hits, (
        f"physiological-event durations are rounded: {dur_hits} -- the events "
        "themselves were scored on a grid, not just the artefact spans"
    )
    print("    Onsets show no grid at any tested spacing, down to 5 ms, and the")
    print("    fractional-second part is uniform. Events were marked at")
    print("    continuous sub-second resolution -- NOT on fixed epochs, and not")
    print("    even rounded to whole seconds. Durations of the physiological")
    print("    events are ungridded too; only hand-dragged SIGNAL-ARTIFACT spans")
    print("    are rounded, which is what shows the test is not simply blunt.")
    print("    => scoring granularity puts NO floor under the prediction horizon;")
    print("       the binding constraint is the clock-drift residual (6.2).")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
