"""MEASURE the annotation-vs-signal clock drift of the CPAP export, per
recording, as a linear time-warp instead of a single constant offset.

Why a drift and not an offset
-----------------------------
An earlier script (`measure_annotation_offset.py`, since deleted) asked "by how
many seconds is the annotation shifted?" and answered with one number per
recording (median +6 s, spread +3.5 .. +9 s). The premise is wrong: the shift is not constant within a recording. Splitting a night into
quarters and re-estimating the shift in each gives, as the cohort median,

    quarter 1: -0.5 s   quarter 2: +3.0 s   quarter 3: +6.5 s   quarter 4: +10.0 s

monotone in 13/13 recordings that carry enough DESAT marks. The annotation is
essentially CORRECT at the start of the night and ~10 s early by the end. Fitting
one constant to that ramp is what produced the "+3.5 .. +9 s per recording"
spread -- it is an average over wherever that recording's events happened to
fall, not a property of the recording.

The magnitude, ~157 ppm (~13.5 s/day), is the classic signature of a sampling
clock running against the recorder's real-time clock: annotations are stamped
from the RTC, sample indices are counted on the sampling crystal. We do not need
the cause to be right for the correction to be right -- the fit is empirical --
but it is why a LINEAR model, and not a step per ANALYSIS block, is fitted.

The model
---------
    drift(t) = intercept + slope * t   [seconds, t = seconds after the EDF start]

An event stamped at t is found in the signal drift(t) seconds LATER than the
annotation says.

The SLOPE is the dominant term and the well-determined one. The INTERCEPT stays
0 for most recordings: with the slope applied the residual against the scoring
rule is already under a second, and fitting a constant to that would be fitting
the estimator's own noise. Four recordings (009, 010, 011, 015) carry a real
constant of 1.5-1.7 s, measured as the pooled residual against the `belts`
anchor and flat across the night; for 015 that is +1.63 s over 457 events with a
0.15 s spread between thirds. See config/clock_drift.yaml.

(An earlier version of this file argued the intercept must be 0 on physical
grounds, because the recorder's real-time clock and its sample counter start
together. That argument held only while every anchor was a PROXY for the
scoring -- the proxies bracketed zero with equal-and-opposite bias, so zero was
their midpoint. The `belts` anchor is not a proxy but condition (i) of the
scoring rule itself, so a residual against it is a real misplacement rather
than anchor bias.)

Anchors
-------
Each anchor is an event type scored against a signal feature whose position is
unambiguous, as a function of an applied lag:

  belts       APNEA vs THORAX AND ABDOMEN BOTH FLAT for >= 10 s. THE
              AUTHORITATIVE ONE, and preferred outright wherever it fits.
              Sievers scored an event where (i) the belt amplitude fell below
              20 % of the preceding breaths AND (ii) no breathing movement was
              visible on the esophageal pressure channel, for >= 10 s. This
              anchor implements condition (i), the half present in the primary
              export -- so it is a condition of the scoring rule itself, not a
              proxy for it. Aligning marks to it therefore aligns them to the
              scorer, which is what "matching RemLogic" means. Corroborated by
              the marked durations, whose median is 11-12 s in every recording
              -- right at the 10 s threshold.
  eso         APNEA vs loss of effort on the ESOPHAGEAL PRESSURE catheter, i.e.
              condition (ii) of that same rule. The best physiological measure
              of effort, but only present in the dataset_v2 export, which is why
              (i) and not (ii) is the anchor that runs by default. The two
              conditions agree on where the events sit (Section 3.2 of the thesis
              section 6.2).
  desat       DESAT vs the FALLING EDGE of SpO2. Sharp, and independent of how
              apneas were scored.
  apnea_cpap  APNEA vs the loss of CPAP-PRESSURE MODULATION relative to the
              local baseline. Mask pressure is this cohort's only airflow proxy;
              airflow drops in central AND obstructive apnea, so this does not
              depend on the subtype. It is nevertheless the least reliable of
              the three -- under a leak the mask pressure keeps oscillating --
              which is what the residual gate below is for.
  apnea_spo2  APNEA vs the desaturation that FOLLOWS it. The apnea-to-desat
              delay is a physiological constant of a few tens of seconds, so
              this anchor's intercept is meaningless -- but a constant delay
              does not tilt the line, so its SLOPE is valid. This is what gives
              the recordings with few or no DESAT marks (001, 013, 015) a
              usable measurement.

An earlier version of this script rejected the belts on the grounds that this
cohort's events are generic `APNEA` relabelled to `APNEA-CENTRAL`
(Section 3.2 of the thesis), so many would be obstructive, with effort continuing
through them. That reasoning was wrong: under the scoring rule above an event
was only ever marked BECAUSE both belts were flat, whatever the true
physiological subtype. SIGNAL-ARTIFACT was tried and rejected on its own merits: measured on patient 013, the signal statistics inside a scored
artifact are indistinguishable from the 60 s before it (std ratios 0.1-0.3 both
inside and before), so it carries no usable edge.

Events are POOLED INTO TIME BINS before the lag is estimated. A single event
gives a very noisy argmax -- the per-event estimate has R^2 ~ 0.1 against time,
which is what made the drift look like scatter rather than a ramp. Pooling ~10+
events per bin first, then fitting a line across bins, is what makes it obvious.
The fit is Theil-Sen (median of pairwise slopes), so a bin that lands on a bad
optimum cannot drag the slope.

Output
------
  clock_drift_bins.csv           per patient/anchor/bin: centre time, fitted lag
  clock_drift_per_patient.csv    every anchor's fit, accepted or not
  clock_drift_coefficients.csv   the chosen slope per patient
  config/clock_drift.yaml        ready for src/neonatal_utils.py
  clock_drift.pdf/.png           lag-vs-time, anchor agreement, consequence

Usage:
    python scripts/measure_clock_drift.py
    python scripts/measure_clock_drift.py --from_csv     # re-fit / re-plot only
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyedflib  # noqa: E402
from scipy.stats import theilslopes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHORS = ("belts", "eso", "desat", "apnea_cpap", "apnea_spo2")
ANCHOR_COLOR = {"belts": "#b8352a", "eso": "#7b3fa0", "desat": "#2a78d6",
                "apnea_cpap": "#eb6834", "apnea_spo2": "#0f9b8e"}
C_LINE = "#c9c8c3"
C_TEXT = "#52514e"

# Lag search per anchor. The drift reaches ~+10 s at the end of a night, so
# +-25 s is generous -- except for apnea_spo2, where the peak also carries the
# apnea-to-desaturation delay and therefore sits tens of seconds late.
LAG_RANGE = {"belts": (-25.0, 25.0),
             "eso": (-25.0, 25.0),
             "desat": (-25.0, 25.0),
             "apnea_cpap": (-25.0, 25.0),
             "apnea_spo2": (-15.0, 75.0)}
LAG_STEP = 0.25   # below the SpO2 sample period; refine_argmax interpolates within it

MIN_EVENTS_PER_BIN = 10   # below this a pooled optimum is not stable
MAX_BINS = 12
MIN_BINS_FOR_FIT = 3      # Theil-Sen needs >= 3 points to be worth the name
# A bin whose score curve has no real optimum (flat, or a ridge at the search
# edge) is dropped, not fitted: the peak must stand this far above the curve's
# own median, in units of the curve's spread.
MIN_PEAK_Z = 2.0

# Acceptance gate for a whole fit. Without it, an anchor that simply failed on a
# recording still yields a slope, and that slope goes into the config and
# corrupts the labels it was meant to fix. The first run of this script produced
# -677 ppm (005) and -1266 ppm (007) from apnea_cpap for exactly that reason.
MAX_RESID_S = 2.5         # residual RMS about the fitted line
# Range a recorder clock can plausibly drift over. The LOWER bound is deliberately
# below zero: a well-behaved crystal drifts by ~nothing, and an earlier 20 ppm
# floor rejected patient 009's correct answer (belts: 2.4 ppm, confirmed
# independently by the post-correction residual) purely for being too GOOD,
# which then forced it onto a wrong 151 ppm proxy fit and left it 6.8 s out.
PLAUSIBLE_PPM = (-50.0, 400.0)

ANALYSIS_FS = 5.0         # Hz, working rate for the CPAP-modulation score
ENV_WIN_S = 3.0           # envelope window ~2-3 neonatal breaths
BASELINE_BLOCK_S = 60.0   # local baseline: per-minute medians, smoothed
BASELINE_BLOCKS = 5

DESAT_SMOOTH_S = 2.0
MIN_SPAN_S = {"belts": 10.0, "eso": 5.0, "desat": 3.0, "apnea_cpap": 5.0, "apnea_spo2": 5.0}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "figures/01_data_quality"))
    p.add_argument("--ids", nargs="*", default=[f"{i:03d}" for i in range(1, 16)])
    p.add_argument("--from_csv", action="store_true")
    p.add_argument("--yaml_out", default=os.path.join(ROOT, "config/clock_drift.yaml"))
    p.add_argument(
        "--verify", action="store_true",
        help="re-measure with the coefficients in config/clock_drift.yaml already "
             "applied to the annotations. The whole point of the correction is "
             "that this leaves no drift, so the fitted slopes must collapse to "
             "~0 ppm. Also the only check that catches a SIGN error, which would "
             "otherwise silently double the drift instead of removing it. "
             "Writes to clock_drift_verify.* and does not touch the config.")
    return p.parse_args()


def load_params(path):
    """{patient: (slope_ppm, intercept_s)} from a config written previously."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {k: (float(v.get("slope_ppm", 0.0)), float(v.get("intercept_s", 0.0)))
            for k, v in cfg["patients"].items()}


def corrected_seconds(t, params):
    """Annotation time -> array time, per the yaml header's convention.

    Applies the intercept as well as the slope. Slope-only would silently
    under-correct the four recordings that carry a measured constant
    (009, 010, 011, 015) and make the verification look worse than reality.
    """
    slope_ppm, intercept_s = params
    return t * (1.0 + slope_ppm * 1e-6) + intercept_s


# ---------------------------------------------------------------------------
# signal loading / scores
# ---------------------------------------------------------------------------
def read_channels(pat_id, data_path, labels):
    """{label: (signal, fs)} plus the EDF start time, reading only what we need."""
    edf = pyedflib.EdfReader(os.path.join(data_path, "signals", f"signals{pat_id}.edf"))
    try:
        available = edf.getSignalLabels()
        out = {}
        for lab in labels:
            if lab in available:
                i = available.index(lab)
                out[lab] = (edf.readSignal(i), float(edf.getSampleFrequency(i)))
        return pd.Timestamp(edf.getStartdatetime()), out
    finally:
        edf._close()


def _resample_mean(x, fs, target_fs):
    """Block-average down to `target_fs`. Anti-aliasing is not needed: every use
    below feeds a rolling envelope/median, which is a low-pass anyway."""
    k = max(1, int(round(fs / target_fs)))
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).mean(axis=1), fs / k


def desat_score(x, fs):
    """Rate of SpO2 DECLINE, rectified. Maximal exactly on a falling edge."""
    xs = pd.Series(x).rolling(max(int(DESAT_SMOOTH_S * fs), 1),
                              center=True, min_periods=1).mean().values
    s = np.clip(-np.gradient(xs) * fs, 0.0, None)
    return np.nan_to_num(s), fs


def cpap_modulation_loss(x, fs):
    """Fractional REDUCTION of respiratory modulation vs the local baseline.

    1.0 = completely quiet relative to the surrounding minutes, 0.0 = normal.
    Judged against a LOCAL baseline (per-minute medians, smoothed over 5 min)
    rather than a global one, because mask pressure, belt gain and breathing
    amplitude all wander over a night; a global baseline would turn a quiet hour
    into one long pseudo-event.
    """
    y, fs2 = _resample_mean(np.asarray(x, dtype=float), fs, ANALYSIS_FS)
    env = pd.Series(y).rolling(max(int(ENV_WIN_S * fs2), 3),
                               center=True, min_periods=1).std().values
    blk = max(1, int(BASELINE_BLOCK_S * fs2))
    n = (len(env) // blk) * blk
    if n < blk * BASELINE_BLOCKS:
        return np.zeros_like(env), fs2
    med = np.nanmedian(env[:n].reshape(-1, blk), axis=1)
    med = pd.Series(med).rolling(BASELINE_BLOCKS, center=True, min_periods=1).median().values
    base = np.interp(np.arange(len(env)), (np.arange(len(med)) + 0.5) * blk, med)
    with np.errstate(invalid="ignore", divide="ignore"):
        red = 1.0 - env / base
    return np.nan_to_num(np.clip(red, 0.0, 1.0)), fs2


def belts_both_flat(thorax, abdomen, fs):
    """The ANNOTATOR'S OWN RULE, reproduced: Thorax AND Abdomen simultaneously
    flat, judged against each belt's own local baseline.

    This is the authoritative anchor for this cohort, because it is not a proxy
    for how the events were scored -- it IS how they were scored. The scorer
    marked an event where both belts went relatively flat at the same time for
    at least 10 s (hence MIN_SPAN_S["belts"] = 10 and the 11-12 s median marked
    duration in every recording). Aligning the marks to that rule therefore
    aligns them to the scorer, which is exactly what "matching RemLogic" means.

    The elementwise MAX of the two normalised envelopes is the "both flat"
    condition: the LOUDER belt must be quiet. Taking the mean or the min would
    let one dead belt satisfy the rule on its own.

    Note this supersedes the earlier reasoning that the belts were unusable
    because the cohort's generic `APNEA` marks include obstructive events, where
    effort continues. Under the scoring rule above that cannot arise: an event
    was only ever marked BECAUSE both belts were flat, whatever the true
    physiological subtype.
    """
    a, fs2 = _resample_mean(np.asarray(thorax, dtype=float), fs, ANALYSIS_FS)
    b, _ = _resample_mean(np.asarray(abdomen, dtype=float), fs, ANALYSIS_FS)
    n = min(len(a), len(b))
    norm = []
    for y in (a[:n], b[:n]):
        w = max(int(ENV_WIN_S * fs2), 3)
        ser = pd.Series(y)
        env = (ser.rolling(w, center=True).max()
               - ser.rolling(w, center=True).min()).values
        blk = max(1, int(BASELINE_BLOCK_S * fs2))
        m = (len(env) // blk) * blk
        if m < blk * BASELINE_BLOCKS:
            return np.zeros(n), fs2
        med = np.nanmedian(env[:m].reshape(-1, blk), axis=1)
        med = pd.Series(med).rolling(BASELINE_BLOCKS, center=True,
                                     min_periods=1).median().values
        base = np.interp(np.arange(len(env)), (np.arange(len(med)) + 0.5) * blk, med)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm.append(np.nan_to_num(env / base, nan=1.0, posinf=1.0))
    louder = np.maximum(norm[0], norm[1])
    return np.clip(1.0 - louder, 0.0, 1.0), fs2


def eso_effort_loss(x, fs):
    """Fractional loss of RESPIRATORY EFFORT vs the local baseline.

    Esophageal pressure is the gold standard for effort and is the channel these
    events were scored on, which makes it the most authoritative anchor
    available -- more so than the belts (which keep moving under CPAP and
    through obstructive events) or mask pressure (which a leak keeps
    oscillating). It is only in the `dataset_v2` export, hence the fallback to
    the other anchors for recordings where it is absent.

    Peak-to-peak rather than the std used for mask pressure: ESO carries large
    slow swings on which a rolling std is dominated by the baseline wander.
    """
    y, fs2 = _resample_mean(np.asarray(x, dtype=float), fs, ANALYSIS_FS)
    w = max(int(ENV_WIN_S * fs2), 3)
    ser = pd.Series(y)
    env = (ser.rolling(w, center=True).max() - ser.rolling(w, center=True).min()).values
    blk = max(1, int(BASELINE_BLOCK_S * fs2))
    n = (len(env) // blk) * blk
    if n < blk * BASELINE_BLOCKS:
        return np.zeros_like(env), fs2
    med = np.nanmedian(env[:n].reshape(-1, blk), axis=1)
    med = pd.Series(med).rolling(BASELINE_BLOCKS, center=True, min_periods=1).median().values
    base = np.interp(np.arange(len(env)), (np.arange(len(med)) + 0.5) * blk, med)
    with np.errstate(invalid="ignore", divide="ignore"):
        loss = 1.0 - env / base
    return np.nan_to_num(np.clip(loss, 0.0, 1.0)), fs2


# ---------------------------------------------------------------------------
# lag estimation
# ---------------------------------------------------------------------------
def score_curve(score, fs, events, lags, min_span_s):
    """Mean score inside the annotated spans, as a function of applied lag."""
    n = len(score)
    out = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        vals = []
        for t, dur in events:
            i0 = int((t + lag) * fs)
            i1 = i0 + int(max(dur, min_span_s) * fs)
            if i0 >= 0 and i1 <= n and i1 > i0:
                vals.append(score[i0:i1].mean())
        if vals:
            out[j] = float(np.mean(vals))
    return out


def refine_argmax(lags, curve):
    """Argmax with a parabolic fit through the neighbouring samples, so the
    estimate is not quantised to LAG_STEP. Returns (lag, z) where z is how far
    the peak stands above the curve's own median in units of its spread --
    the quality gate that rejects a flat or edge-ridden curve."""
    if np.all(np.isnan(curve)):
        return np.nan, 0.0
    i = int(np.nanargmax(curve))
    med = np.nanmedian(curve)
    spread = np.nanstd(curve)
    z = (curve[i] - med) / spread if spread > 0 else 0.0
    if i == 0 or i == len(curve) - 1 or not np.all(np.isfinite(curve[i - 1:i + 2])):
        return float(lags[i]), float(z)
    y0, y1, y2 = curve[i - 1], curve[i], curve[i + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(lags[i] + delta * (lags[1] - lags[0])), float(z)


def bin_events(events, n_bins):
    """Split time-sorted events into `n_bins` groups of equal COUNT.

    Equal count, not equal duration: events cluster (a bad hour produces
    hundreds), and equal-duration bins would leave some with two events and
    others with three hundred.
    """
    ev = sorted(events, key=lambda e: e[0])
    return [list(g) for g in np.array_split(np.array(ev, dtype=object), n_bins)]


def measure_anchor(pat_id, anchor, score, fs, events):
    lo, hi = LAG_RANGE[anchor]
    lags = np.arange(lo, hi + LAG_STEP / 2, LAG_STEP)
    n_bins = int(np.clip(len(events) // MIN_EVENTS_PER_BIN, 0, MAX_BINS))
    if n_bins < MIN_BINS_FOR_FIT:
        return []
    rows = []
    for b, grp in enumerate(bin_events(events, n_bins)):
        if len(grp) == 0:
            continue
        curve = score_curve(score, fs, grp, lags, MIN_SPAN_S[anchor])
        lag, z = refine_argmax(lags, curve)
        rows.append(dict(patient=pat_id, anchor=anchor, bin=b, n_events=len(grp),
                         t_centre=float(np.median([e[0] for e in grp])),
                         lag_s=lag, peak_z=z))
    return rows


def fit_drift(bins):
    """Theil-Sen fit of lag = intercept + slope * t over the accepted bins."""
    good = bins[bins["peak_z"] >= MIN_PEAK_Z]
    if len(good) < MIN_BINS_FOR_FIT:
        return None
    t = good["t_centre"].to_numpy(float)
    y = good["lag_s"].to_numpy(float)
    if t.max() - t.min() < 3600.0:   # a ramp cannot be measured over < 1 h
        return None
    slope, intercept, _, _ = theilslopes(y, t)
    resid = y - (intercept + slope * t)
    return dict(n_bins=len(good), n_events=int(good["n_events"].sum()),
                slope_ppm=slope * 1e6, intercept_s=intercept,
                resid_rms_s=float(np.sqrt(np.mean(resid ** 2))),
                total_drift_s=float(slope * (t.max() - t.min())))


def accepted(fit):
    """Is this fit trustworthy enough to correct a recording with?"""
    return (fit is not None
            and fit["resid_rms_s"] <= MAX_RESID_S
            and PLAUSIBLE_PPM[0] <= fit["slope_ppm"] <= PLAUSIBLE_PPM[1])


# ---------------------------------------------------------------------------
def measure_patient(pat_id, data_path, params=None):
    anno = pd.read_csv(
        os.path.join(data_path, "annotations", f"annotations{pat_id}.txt"), sep="\t")
    t0, chans = read_channels(
        pat_id, data_path,
        ["SpO2 Radical", "SpO2", "CPAP Pressure", "Thorax", "Abdomen",
         "?sophagusdruck", "Oesophagusdruck"])
    anno["s"] = (pd.to_datetime(anno["timestamp"]) - t0).dt.total_seconds()
    if params is not None:
        anno["s"] = corrected_seconds(anno["s"], params)

    desats = [(r.s, float(r.duration))
              for r in anno[anno["type"] == "DESAT"].itertuples(index=False)]
    apneas = [(r.s, float(r.duration))
              for r in anno[anno["type"] == "APNEA-CENTRAL"].itertuples(index=False)]

    rows = []
    spo2 = chans.get("SpO2 Radical") or chans.get("SpO2")
    if spo2 is not None:
        s, fs = desat_score(*spo2)
        rows += measure_anchor(pat_id, "desat", s, fs, desats)
        rows += measure_anchor(pat_id, "apnea_spo2", s, fs, apneas)

    cpap = chans.get("CPAP Pressure")
    if cpap is not None:
        s, fs = cpap_modulation_loss(*cpap)
        rows += measure_anchor(pat_id, "apnea_cpap", s, fs, apneas)

    thorax, abdomen = chans.get("Thorax"), chans.get("Abdomen")
    if thorax is not None and abdomen is not None and len(apneas) >= 2 * MIN_EVENTS_PER_BIN:
        s, fs = belts_both_flat(thorax[0], abdomen[0], thorax[1])
        rows += measure_anchor(pat_id, "belts", s, fs, apneas)

    # Only present in the dataset_v2 export -- point --data_path there to use it.
    eso = chans.get("?sophagusdruck") or chans.get("Oesophagusdruck")
    if eso is not None:
        s, fs = eso_effort_loss(*eso)
        rows += measure_anchor(pat_id, "eso", s, fs, apneas)

    n = {a: sum(1 for r in rows if r["anchor"] == a) for a in ANCHORS}
    print(f"  {pat_id}: {len(desats):4d} desat, {len(apneas):4d} apnea "
          f"-> bins {n}")
    return rows


def build_per_patient(bins_df, ids):
    """Every anchor's fit for every patient, with the accept/reject decision."""
    out = []
    for pat in ids:
        g = bins_df[bins_df["patient"] == pat] if len(bins_df) else bins_df
        rec = {"patient": pat}
        for anchor in ANCHORS:
            f = fit_drift(g[g["anchor"] == anchor]) if len(g) else None
            if f:
                rec[f"{anchor}_ppm"] = f["slope_ppm"]
                rec[f"{anchor}_icpt"] = f["intercept_s"]
                rec[f"{anchor}_resid"] = f["resid_rms_s"]
                rec[f"{anchor}_nbin"] = f["n_bins"]
                rec[f"{anchor}_ok"] = accepted(f)
            else:
                rec[f"{anchor}_ok"] = False
        out.append(rec)
    return pd.DataFrame(out)


def choose_coefficients(per_pat):
    """Per patient, the slope that goes into the config.

    Among the anchors that pass the gate, take the one with the SMALLEST
    residual: the residual is a direct measure of how well a straight line
    described that anchor's bins, which is exactly the property being relied on.
    A recording where no anchor passes falls back to the cohort median slope --
    better than leaving it uncorrected, but flagged as `cohort_median` so it can
    be excluded from any claim that rests on exact per-recording timing.
    """
    accepted_ppm = []
    for a in ANCHORS:
        col, ok = f"{a}_ppm", f"{a}_ok"
        if col in per_pat:
            accepted_ppm += list(per_pat.loc[per_pat[ok] == True, col].dropna())  # noqa: E712
    med_ppm = float(np.median(accepted_ppm)) if accepted_ppm else 157.0

    rows = []
    for r in per_pat.to_dict("records"):
        cands = [(r[f"{a}_resid"], a, r[f"{a}_ppm"])
                 for a in ANCHORS
                 if r.get(f"{a}_ok") and np.isfinite(r.get(f"{a}_ppm", np.nan))]
        # `belts` wins outright when it passed, regardless of residual: it
        # reproduces the annotator's own scoring rule (Thorax and Abdomen both
        # flat for >=10 s), so it defines the target every other anchor is only
        # a proxy for. Among the proxies, smallest residual wins.
        primary = [c for c in cands if c[1] == "belts"]
        if primary:
            resid, src, ppm = primary[0]
        elif cands:
            resid, src, ppm = min(cands, key=lambda c: c[0])
        else:
            resid, src, ppm = np.nan, "cohort_median", med_ppm
        rows.append(dict(patient=r["patient"], source=src, slope_ppm=float(ppm),
                         resid_rms_s=resid, n_anchors_ok=len(cands)))
    return pd.DataFrame(rows), med_ppm


def write_yaml(coef, path, med_ppm):
    # `intercept_s` is NOT regenerated. It is measured separately -- the pooled
    # residual against the annotator's rule after the slope is applied, see
    # scripts/check_intercepts.py -- and hand-entered here. Four recordings
    # carry a real one (009 +1.71, 010 +1.52, 011 -1.49, 015 +1.63). Writing
    # 0.0 unconditionally, as this function used to, silently reverted those
    # four to slope-only on any rerun. Existing values are read back and
    # carried forward instead.
    keep_icpt = {}
    if os.path.exists(path):
        try:
            keep_icpt = {pat: icpt for pat, (_, icpt) in load_params(path).items()}
        except Exception as exc:            # a malformed config must not silently
            print(f"WARNING: could not read intercepts from {path} ({exc}); "  # zero them
                  "they would be lost -- aborting the write.")
            raise

    lines = [
        "# Annotation-clock drift of the CPAP export, measured per recording by",
        "# scripts/measure_clock_drift.py. Consumed by src/neonatal_utils.py.",
        "#",
        "# An annotation timestamp t (seconds after the EDF start) maps to a",
        "# sample index as",
        "#",
        "#     index = (t*(1 + slope_ppm*1e-6) + intercept_s) * TARGET_FREQ",
        "#",
        "# i.e. an event stamped at t is FOUND IN THE SIGNAL that much LATER than",
        "# the annotation says: ~0 s at the start of a recording, ~+10 s by the end",
        "# of a 24 h night.",
        "#",
        "# The slope is the measured, validated quantity. `intercept_s` is NOT",
        "# regenerated by the script: it is measured separately as the pooled",
        "# residual against the annotator's own rule (Thorax and Abdomen both",
        "# flat >= 10 s) after the slope has been applied, and set only where",
        "# that residual is >= 1 s AND flat across thirds of the recording.",
        "# Eleven recordings are left at 0, where the residual is already under",
        "# a second and fitting a constant would fit the estimator's own noise.",
        "# Any value already in this file is carried through a rerun unchanged;",
        "# scripts/check_intercepts.py re-measures them without writing.",
        "#",
        "# An earlier version of this header argued the intercept MUST be 0,",
        "# because the proxy anchors bracketed zero (DESAT ~-1.9 s, CPAP",
        "# modulation ~+1.4 s) with equal-and-opposite bias. That held while the",
        "# anchors were only proxies for the scoring. The belts anchor is not a",
        "# proxy -- it reproduces the scorer's decision -- so a residual against",
        "# it is a real misplacement. See Section 3.3 of the thesis.",
        "#",
        f"# Cohort median slope: {med_ppm:.1f} ppm (~{med_ppm * 86400e-6:.1f} s per 24 h).",
        "# `source` is the anchor the fit came from; 'cohort_median' means no",
        "# anchor passed the quality gate for that recording.",
        "",
        "patients:",
    ]
    for r in coef.itertuples(index=False):
        icpt = keep_icpt.get(r.patient, 0.0)
        lines.append(f'  "{r.patient}": {{slope_ppm: {r.slope_ppm:.2f}, '
                     f"intercept_s: {icpt}, source: {r.source}}}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def make_figure(bins_df, per_pat, coef, out_dir, med_ppm):
    # `top` has to clear the five-line standfirst below the suptitle, which
    # reaches down to ~0.83 of the figure; at 0.835 it printed straight through
    # panel (a)'s title.
    fig = plt.figure(figsize=(7.2, 7.9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.95, 0.85],
                          hspace=0.62, wspace=0.30,
                          top=0.790, bottom=0.070, left=0.11, right=0.98)

    # (a) every accepted bin of every recording, against time of night
    ax = fig.add_subplot(gs[0, :])
    for anchor in ("desat", "apnea_cpap"):
        g = bins_df[(bins_df["anchor"] == anchor) & (bins_df["peak_z"] >= MIN_PEAK_Z)]
        ax.scatter(g["t_centre"] / 3600.0, g["lag_s"], s=13,
                   color=ANCHOR_COLOR[anchor], alpha=0.75, linewidths=0,
                   label=anchor, zorder=3)
    tt = np.linspace(0, 24, 50)
    ax.plot(tt, med_ppm * 1e-6 * tt * 3600.0, color=C_TEXT, lw=1.7, ls="--", zorder=4,
            label=f"fitted drift, {med_ppm:.0f} ppm")
    ax.axhline(0, color=C_LINE, lw=1.0, zorder=1)
    ax.axhline(6.0, color="#b8352a", lw=1.0, ls=":", zorder=2)
    ax.text(0.3, 6.6, "the +6 s constant previously applied everywhere",
            fontsize=7, color="#b8352a", va="bottom")
    ax.set_xlim(0, 24)
    ax.set_ylim(-12, 18)
    ax.set_xlabel("Hours into the recording")
    ax.set_ylabel("Annotation is early by [s]")
    ax.set_title("(a) The shift is a ramp, not a constant", loc="left")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
    # apnea_spo2 is deliberately not drawn here: its lag carries the
    # apnea-to-desaturation delay on top of the drift, so it sits tens of
    # seconds high and would compress this panel's y-axis to nothing. Its
    # slope still appears in (b), which is all it is used for.

    # (b) do the independent anchors agree on the SLOPE?
    ax = fig.add_subplot(gs[1, 0])
    for k, anchor in enumerate(ANCHORS):
        col, ok = f"{anchor}_ppm", f"{anchor}_ok"
        if col not in per_pat:
            continue
        sub = per_pat[per_pat[ok] == True]  # noqa: E712
        ax.scatter(sub[col], np.full(len(sub), k) + np.random.default_rng(0)
                   .uniform(-0.13, 0.13, len(sub)),
                   s=24, color=ANCHOR_COLOR[anchor], linewidths=0, zorder=3)
    ax.axvline(med_ppm, color=C_TEXT, lw=1.2, ls="--", zorder=2)
    ax.set_yticks(range(len(ANCHORS)))
    ax.set_yticklabels(ANCHORS, fontsize=7)
    ax.set_ylim(-0.6, len(ANCHORS) - 0.4)
    ax.set_xlabel("Fitted drift [ppm]")
    ax.set_title("(b) Three anchors, one answer", loc="left")

    # (c) the intercepts straddle zero -> the offset at t=0 is anchor bias
    ax = fig.add_subplot(gs[1, 1])
    for k, anchor in enumerate(("desat", "apnea_cpap")):
        col, ok = f"{anchor}_icpt", f"{anchor}_ok"
        if col not in per_pat:
            continue
        sub = per_pat[per_pat[ok] == True]  # noqa: E712
        ax.scatter(sub[col], np.full(len(sub), k), s=24,
                   color=ANCHOR_COLOR[anchor], linewidths=0, zorder=3)
        if len(sub):
            ax.plot([sub[col].median()], [k], marker="|", ms=16, mew=2,
                    color=C_TEXT, zorder=4)
    ax.axvline(0, color=C_LINE, lw=1.4, zorder=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["desat", "apnea_cpap"], fontsize=7)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Fitted intercept [s]")
    ax.set_title("(c) Straddling 0 = anchor bias,\n     so no intercept is applied",
                 loc="left")

    # (d) what the drift does to the training labels
    ax = fig.add_subplot(gs[2, :])
    hours = np.linspace(0, 24, 100)
    for _, r in coef.iterrows():
        ax.plot(hours, 15.0 + r["slope_ppm"] * 1e-6 * hours * 3600.0,
                color=C_LINE, lw=0.8, zorder=1)
    ax.plot(hours, 15.0 + med_ppm * 1e-6 * hours * 3600.0,
            color=ANCHOR_COLOR["desat"], lw=2.0, zorder=3, label="cohort median")
    ax.axhline(15.0, color=C_TEXT, lw=1.2, ls="--", zorder=2,
               label="the 15 s horizon the config asks for")
    ax.set_xlim(0, 24)
    ax.set_xlabel("Hours into the recording")
    ax.set_ylabel("True prediction\nhorizon [s]")
    ax.set_title("(d) Consequence: positive windows are not where the config says",
                 loc="left")
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6)

    fig.suptitle(
        "The CPAP annotation clock drifts against the signal clock:\n"
        f"~{med_ppm:.0f} ppm (~{med_ppm * 86400e-6:.0f} s per 24 h), "
        "not a fixed +6 s offset",
        x=0.02, y=0.985, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(
        0.02, 0.900,
        "Three independent anchors: DESAT marks against the falling edge of SpO2; APNEA marks against the loss of CPAP-pressure\n"
        "modulation; APNEA marks against the desaturation that follows them. Events are pooled into bins of >=10 before the lag is\n"
        "estimated, then a Theil-Sen line is fitted across bins and gated on its residual. Panel (d) shows why this matters for\n"
        "training: with `lag: 3000` the positive windows sit 15 s before the ANNOTATED onset, so the real horizon slides towards 25 s\n"
        "by the end of a night -- a systematic, time-dependent label error that no fixed offset can remove.",
        ha="left", va="top", fontsize=7.3, color=C_TEXT)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"clock_drift.{ext}"), format=ext)
    plt.close(fig)


def run_verify(args):
    """Re-measure with the correction already applied; the drift must vanish."""
    slopes = load_params(args.yaml_out)
    print(f"VERIFY: re-measuring with {args.yaml_out} applied. "
          "Fitted slopes must collapse to ~0 ppm.")
    rows = []
    for pat_id in args.ids:
        rows += measure_patient(pat_id, args.data_path, slopes.get(pat_id, (0.0, 0.0)))
    bins_df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    bins_df.to_csv(os.path.join(args.out_dir, "clock_drift_verify_bins.csv"),
                   index=False)
    per_pat = build_per_patient(bins_df, args.ids)
    per_pat.round(2).to_csv(
        os.path.join(args.out_dir, "clock_drift_verify_per_patient.csv"), index=False)

    pd.set_option("display.width", 240)
    show = ["patient"] + [f"{a}_{s}" for a in ANCHORS for s in ("ppm", "resid")]
    print("\nresidual drift after correction (ppm; was ~+158 before):")
    print(per_pat.reindex(columns=show).round(1).to_string(index=False))

    # What is left of the drift. Only fits that would have PASSED THE GATE count:
    # a fit the gate rejects was never used to correct anything, so including it
    # here would measure the anchors' failure modes rather than the correction.
    # Reported in seconds over 24 h as well, since a recording-length ramp is
    # what actually matters.
    def summarise(name, vals):
        v = np.asarray(vals, dtype=float)
        if not len(v):
            return
        print(f"  {name:12s} n={len(v):3d}  median {np.median(v):+6.1f} ppm "
              f"({np.median(v) * 86400e-6:+5.2f} s/24 h)   "
              f"worst |{np.max(np.abs(v)):.0f}| ppm "
              f"({np.max(np.abs(v)) * 86400e-6:.1f} s/24 h)")

    print("\nresidual drift, over fits that pass the quality gate "
          f"(resid <= {MAX_RESID_S} s):")
    pooled = []
    for a in ANCHORS:
        if f"{a}_ppm" not in per_pat:
            continue
        keep = per_pat[per_pat[f"{a}_resid"] <= MAX_RESID_S][f"{a}_ppm"].dropna()
        summarise(a, keep)
        pooled += list(keep)
    summarise("ALL", pooled)
    print("\nbefore correction the same quantity was +158 ppm (+13.7 s per 24 h).")
    return per_pat


def main():
    args = parse_args()
    if args.verify:
        run_verify(args)
        return

    bins_csv = os.path.join(args.out_dir, "clock_drift_bins.csv")

    if args.from_csv:
        bins_df = pd.read_csv(bins_csv, dtype={"patient": str})
    else:
        print(f"measuring clock drift on {len(args.ids)} recordings:")
        rows = []
        for pat_id in args.ids:
            rows += measure_patient(pat_id, args.data_path)
        bins_df = pd.DataFrame(rows)
        os.makedirs(args.out_dir, exist_ok=True)
        bins_df.to_csv(bins_csv, index=False)

    per_pat = build_per_patient(bins_df, args.ids)
    coef, med_ppm = choose_coefficients(per_pat)
    per_pat.round(2).to_csv(
        os.path.join(args.out_dir, "clock_drift_per_patient.csv"), index=False)
    coef.round(3).to_csv(
        os.path.join(args.out_dir, "clock_drift_coefficients.csv"), index=False)
    write_yaml(coef, args.yaml_out, med_ppm)
    make_figure(bins_df, per_pat, coef, args.out_dir, med_ppm)

    pd.set_option("display.width", 240)
    show = ["patient"] + [f"{a}_{s}" for a in ANCHORS for s in ("ppm", "resid", "ok")]
    print("\nevery anchor's fit (ppm = parts per million of elapsed time):")
    print(per_pat.reindex(columns=show).round(1).to_string(index=False))
    print("\nchosen per recording -> config/clock_drift.yaml:")
    print(coef.round(2).to_string(index=False))

    # Cross-anchor agreement is the evidence that these numbers are real and not
    # an artefact of one score definition, so it is printed, not just plotted.
    for a, b in (("desat", "apnea_cpap"), ("desat", "apnea_spo2"),
                 ("apnea_cpap", "apnea_spo2")):
        if f"{a}_ppm" not in per_pat or f"{b}_ppm" not in per_pat:
            continue
        m = per_pat[(per_pat[f"{a}_ok"] == True) & (per_pat[f"{b}_ok"] == True)]  # noqa: E712
        if len(m) >= 3:
            d = (m[f"{a}_ppm"] - m[f"{b}_ppm"]).abs()
            print(f"  {a} vs {b}: n={len(m)}, median |difference| = {d.median():.1f} ppm")

    n_fallback = int((coef["source"] == "cohort_median").sum())
    print(f"\ncohort median slope {med_ppm:.1f} ppm (~{med_ppm * 86400e-6:.1f} s per 24 h); "
          f"{len(coef) - n_fallback}/{len(coef)} recordings measured individually, "
          f"{n_fallback} on the cohort median")
    print(f"written to {args.out_dir}: clock_drift.pdf/.png + 3 CSVs, "
          f"and {args.yaml_out}")


if __name__ == "__main__":
    main()
