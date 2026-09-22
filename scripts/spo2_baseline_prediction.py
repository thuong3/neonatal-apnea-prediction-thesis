"""SpO2 BASELINE DRIFT as a predictor -- the two-phenomena analysis.

Motivation (supervisor's feedback, backed by the AOP literature): oxygenation in
these infants deteriorates in **two physiologically distinct ways**, and the
model built so far can only see one of them.

  A -- FRANK DESATURATION ("intermittent hypoxemia", IH). Short, deep dips.
       Dormishian et al. (J Pediatr 2023): IH = SpO2 < 90 % for >= 5 s;
       SEVERE IH = SpO2 < 80 % for >= 5 s; consecutive dips less than 5 s apart
       count as one episode. This is what the `DESAT` annotation marks and what
       makes DETECTION work at AuROC 0.87 (Section 4.2 of the thesis).

  B -- BASELINE DECLINE. The *average* saturation drifts down over tens of
       minutes -- e.g. from 97 % to 90 % -- with few or no frank desaturations.
       Clinically this is at least as important, and the current pipeline is
       BLIND TO IT BY CONSTRUCTION:
         * `standardize()` (src/neonatal_utils.py) z-scores every 30 s window
           on its own, so absolute SpO2 level is removed before the network
           ever sees it -- 95 % and 88 % look identical;
         * a 30 s window is two orders of magnitude shorter than the drift.

Why B might be a genuine PRECURSOR rather than only a consequence (Poets, Sleep
Med 2010): apneas reduce functional residual capacity, and a reduced lung volume
*raises loop gain*, destabilising respiratory control (the same mechanism that
produces periodic breathing). Falling baseline saturation is a proxy for falling
lung volume, so the causal arrow plausibly runs both ways. That is exactly why
the event-history control below is not optional.

WHAT THIS SCRIPT DOES
  1. builds a causal long-term SpO2 baseline B_T(t) (T = 5/15/30/60 min) that
     EXCLUDES the events themselves, so it measures the level *between* dips;
  2. derives the supervisor's predictor  M - B_T  (window mean minus baseline)
     plus baseline level, baseline slope, and the IH-burden block for A;
  3. evaluates three endpoints -- annotated apnea, signal-derived IH, and
     baseline decline -- as a FEATURE-GROUP ABLATION against the existing
     physiological and event-history blocks.

THE CONTROL THAT DECIDES THE RESULT
  A depressed baseline is partly a *consequence* of recent apneas, so
  "low baseline predicts apnea" could just be the clustering signal of
  Section 4.6 of the thesis re-expressed through SpO2. The honest test is
  whether `drift` adds anything ON TOP OF `history`; both are reported, and so
  is the combination. Note that `drift` has a deployment advantage `history`
  does not: it needs no annotations at test time.

Evaluation is leakage-safe, matching scripts/clustering_prediction.py:
cross-patient leave-one-out, and within-patient temporal split with a buffer gap.

Usage:
    python scripts/spo2_baseline_prediction.py [<data_path>] [--no-physio]
                                               [--cache <file.npz>]
"""
import argparse
import os
import sys
import warnings

import numpy as np
from scipy import signal as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402

FS = 200                      # rate of the arrays read_data returns
HZ = 1                        # everything in this script is analysed at 1 Hz
WIN_S = 60                    # feature window
STRIDE_S = 30
HORIZONS_S = [60, 120, 180]   # predict "within the next H seconds"
# LEAD: a gap between the end of the feature window and the start of the target
# interval. Without it (lead = 0) the target begins at the very next sample, and
# a "prediction" of a signal-derived SpO2 event is partly DETECTION: a
# desaturation takes ~10-20 s to develop, so a window ending at t already
# contains the start of a fall that crosses 90 % at t+10. That is precisely the
# flaw #15 identifies in the published prediction literature, so it has to be
# controlled here rather than reproduced. Target interval = [e+lead, e+lead+H).
LEADS_S = [0, 60, 120]
BASE_T_MIN = [5, 15, 30, 60]  # long-term baseline timescales
IH_LOOKBACK_MIN = [5, 30]     # IH-burden lookbacks
DECLINE_DROP = 3.0            # "baseline decline" endpoint: >= 3 % below B_30
# ABSOLUTE variant of the decline endpoint (#23.7c). The relative target is
# defined against B30, which is ALSO an input feature: a high B30 raises the
# threshold and mechanically makes a "decline" more likely, so its 0.677 is an
# upper bound rather than an honest number. This target is a fixed clinical
# level instead -- mean SpO2 below 90 % over the target interval -- so no
# feature can move the goalposts. 90 % is Dormishian et al."s IH threshold,
# i.e. the same level the field already treats as hypoxemia.
DECLINE_ABS = 90.0
# An apnea counts as CLINICALLY SIGNIFICANT if a scored DESAT begins within
# this many seconds of its onset. Poets (Sleep Med 2010) puts the true
# apnea->desaturation interval at a median of 0.8 s, but the SCORED onsets of
# the two events are placed independently by the annotator, so the link needs
# a tolerance rather than a coincidence test. 20 s is generous enough to
# absorb that and still far shorter than the 60 s prediction horizon.
DESAT_LINK_S = 20
IDS = [f"{i:03d}" for i in range(1, 16)]

# SpO2 validity. Dropouts are coded as small values (0.3 .. 3 %), not as 0, so a
# plausibility floor is the right mask. Values ABOVE 100 % also occur (up to
# ~106 %, and in one segment for a third of all samples): that is a per-recording
# gain/offset error in the EDF's physical scaling, not a dropout. They are kept
# rather than discarded, because (a) discarding them would throw away a third of
# some recordings, and (b) the headline predictor is a DIFFERENCE within one
# recording (M - B), in which a per-recording gain error largely cancels. The
# absolute-level features are the ones to distrust; `--report-gain` prints the
# per-segment overshoot so the effect is visible rather than hidden.
SPO2_MIN_VALID = 50.0
IH_THRESHOLD = 90.0           # Dormishian et al. 2023: IH  = SpO2 < 90 %
IH_SEVERE = 80.0              # Dormishian et al. 2023: severe IH = SpO2 < 80 %
IH_MIN_DUR_S = 5              # ... for at least 5 s
IH_MERGE_GAP_S = 5            # dips less than 5 s apart are one episode

DRIFT = load_clock_drift()

# COHORT SWITCHES. Defaults are the CPAP cohort, so every result reported in
# #23-#25 is reproduced unchanged; main() overrides them for Robin. They are
# module-level rather than threaded through every signature because
# spo2_context() is called from three scripts.
APNEA_TYPE = "APNEA-CENTRAL"      # the prediction target
ARTIFACT_TYPE = "SIGNAL-ARTIFACT"  # CPAP; Robin marks ACTIVITY-MOVE instead
ANNOTATIONS_DIR = "annotations"    # Robin ships annotations_original/
RECORD_DURATION = None             # Robin needs 10 (see Section 4.1 of the thesis)
USE_DRIFT = True                   # clock-drift table is CPAP-specific
WITH_PR = False                    # append the PR block (see physio_features)
_BP = sp.butter(2, [5 / (FS / 2), 40 / (FS / 2)], "band")


# --------------------------------------------------------------------------
# causal rolling statistics
# --------------------------------------------------------------------------
def causal_mean(x, valid, w, min_count):
    """Mean of x over the w samples STRICTLY BEFORE each index, valid ones only.

    Strictly-before matters: element i must not see sample i, so that a feature
    computed at a window end cannot borrow from the window it is predicting.
    NaN wherever fewer than `min_count` valid samples are available.
    """
    v = valid.astype(np.float64)
    xs = np.where(valid, x, 0.0).astype(np.float64)
    csx = np.concatenate([[0.0], np.cumsum(xs)])
    csv = np.concatenate([[0.0], np.cumsum(v)])
    n = len(x)
    hi = np.arange(n)                       # exclusive end = i  (strictly past)
    lo = np.maximum(0, hi - w)
    cnt = csv[hi] - csv[lo]
    tot = csx[hi] - csx[lo]
    out = np.full(n, np.nan)
    ok = cnt >= min_count
    out[ok] = tot[ok] / cnt[ok]
    return out


BLOCK_S = 10          # baseline is evaluated on a 10 s grid, then held at 1 Hz
BASE_PCTL = 75        # "the level between the dips" -- see causal_percentile


def causal_percentile(x, valid, w_s, pctl, min_frac=0.2):
    """Causal rolling percentile of SpO2 -- the BASELINE definition.

    A first attempt defined the baseline as the mean of samples above the IH
    threshold (90 %). That is self-defeating: in exactly the segments where
    phenomenon B is strongest -- the baseline itself sinks below 90 % -- no
    sample qualifies and the baseline becomes UNDEFINED (visible as a gap in
    the first version of figures/06_spo2_baseline, patient 008 segment 1).

    An upper percentile solves it without any absolute cutoff. Desaturations
    are excursions into the LOWER tail, so the 75th percentile of the recent
    past tracks the level between the dips at any absolute level, and stays
    defined however low the infant sits. It is also robust to the residual
    dropout samples that survive the plausibility mask.

    Evaluated on a 10 s grid and held at 1 Hz: a 30-60 min baseline has no
    meaningful content at 1 Hz, and the grid makes the rolling percentile cheap.
    """
    n = len(x)
    nb = n // (BLOCK_S * HZ)
    if nb < 2:
        return np.full(n, np.nan)
    k = BLOCK_S * HZ
    xb = np.where(valid, x, np.nan)[: nb * k].reshape(nb, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        block = np.nanmedian(xb, axis=1)          # one robust value per 10 s
    wb = max(2, w_s // BLOCK_S)
    out_b = np.full(nb, np.nan)
    # strictly-past window [i-wb, i): the value at block i never sees block i
    for i in range(1, nb):
        seg = block[max(0, i - wb):i]
        good = seg[np.isfinite(seg)]
        if len(good) >= max(3, int(min_frac * min(wb, i))):
            out_b[i] = np.percentile(good, pctl)
    # A long SIGNAL-ARTIFACT stretch would otherwise blank the baseline exactly
    # where the recording is worst. The baseline is a slow quantity, so carry
    # the last value forward across a BOUNDED gap (15 min) and only then give
    # up. Beyond that the held value would be stale enough to invent a trend.
    max_hold = (15 * 60) // BLOCK_S
    last, age = np.nan, 0
    for i in range(nb):
        if np.isfinite(out_b[i]):
            last, age = out_b[i], 0
        elif np.isfinite(last) and age < max_hold:
            age += 1
            out_b[i] = last
        else:
            last = np.nan
    out = np.repeat(out_b, k)
    if len(out) < n:
        out = np.concatenate([out, np.full(n - len(out), out[-1] if len(out) else np.nan)])
    return out[:n]


def episodes(below, min_dur, merge_gap):
    """(start, stop) index pairs of runs in `below`, short gaps merged first."""
    b = below.astype(np.int8)
    if not b.any():
        return []
    d = np.diff(np.concatenate([[0], b, [0]]))
    starts = np.where(d == 1)[0]
    stops = np.where(d == -1)[0]
    merged = [[starts[0], stops[0]]]
    for s, e in zip(starts[1:], stops[1:]):
        if s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_dur]


# --------------------------------------------------------------------------
# per-segment SpO2 derivations (computed once, reused by every window)
# --------------------------------------------------------------------------
def spo2_context(seg):
    """Everything derived from SpO2 for one segment, at 1 Hz."""
    step = FS // HZ
    s = np.asarray(seg["SpO2"][::step], float)
    arti = np.asarray(seg[ARTIFACT_TYPE][::step], bool)
    valid = (s >= SPO2_MIN_VALID) & np.isfinite(s)

    # The baseline must measure the level BETWEEN the dips, otherwise a night
    # full of desaturations drags "baseline" down and phenomena A and B stop
    # being separable -- the whole point of the analysis.
    #
    # It is SIGNAL-DERIVED (an upper percentile of the recent past), not
    # annotation-derived, for two reasons: it keeps the feature deployable
    # without a scorer, and it keeps the DESAT annotation available as an
    # independent control instead of being baked into the predictor.
    base_ok = valid & ~arti

    ctx = {"spo2": s, "valid": valid, "arti": arti, "n": len(s)}
    for T in BASE_T_MIN:
        ctx[f"B{T}"] = causal_percentile(s, base_ok, T * 60, BASE_PCTL)
    # baseline SLOPE: how far the baseline has moved over the recent past.
    # Expressed as % per 10 min so the number is readable in the write-up.
    for T, back_s in [(30, 600), (60, 1800)]:
        b = ctx[f"B{T}"]
        sh = np.full_like(b, np.nan)
        k = back_s * HZ
        sh[k:] = b[k:] - b[:-k]
        ctx[f"dB{T}"] = sh * (600.0 / back_s)

    ep = episodes(valid & (s < IH_THRESHOLD), IH_MIN_DUR_S * HZ, IH_MERGE_GAP_S * HZ)
    sev = episodes(valid & (s < IH_SEVERE), IH_MIN_DUR_S * HZ, IH_MERGE_GAP_S * HZ)
    ctx["ih_onsets"] = np.array([a for a, _ in ep], dtype=int)
    ctx["ih_nadir"] = np.array([s[a:b].min() for a, b in ep]) if ep else np.zeros(0)
    ctx["sev_onsets"] = np.array([a for a, _ in sev], dtype=int)
    for thr in (80.0, 85.0, 90.0):
        ctx[f"below{int(thr)}"] = (valid & (s < thr)).astype(np.float64)
    return ctx


def nanmean_slice(x, valid, lo, hi):
    m = valid[lo:hi]
    return float(x[lo:hi][m].mean()) if m.any() else np.nan


# --------------------------------------------------------------------------
# feature blocks
# --------------------------------------------------------------------------
def drift_features(ctx, e):
    """Phenomenon B: where is the infant now relative to its OWN recent normal."""
    s, valid = ctx["spo2"], ctx["valid"]
    w0 = e - WIN_S * HZ
    f = []
    M = nanmean_slice(s, valid, w0, e)                    # window mean
    M30 = nanmean_slice(s, valid, e - 30 * HZ, e)         # Dormishian basal scale
    f += [M, M30]
    for T in BASE_T_MIN:
        B = ctx[f"B{T}"][e]
        f += [B, M - B]                                   # <- the supervisor's predictor
    f += [ctx["dB30"][e], ctx["dB60"][e]]
    seg = s[w0:e][valid[w0:e]]
    if len(seg) >= 5:
        f += [float(seg.std()),
              float(np.polyfit(np.arange(len(seg)), seg, 1)[0] * 60.0),  # %/min
              float(seg.min() - ctx["B30"][e])]
    else:
        f += [np.nan, np.nan, np.nan]
    return f


DRIFT_NAMES = (["M", "M30"]
               + [n for T in BASE_T_MIN for n in (f"B{T}", f"M-B{T}")]
               + ["dB30", "dB60", "sd_win", "slope_win", "min-B30"])


def ih_features(ctx, e):
    """Phenomenon A: recent burden of frank desaturation, signal-derived."""
    f = []
    for L in IH_LOOKBACK_MIN:
        lo = max(0, e - L * 60 * HZ)
        den = max(1, e - lo)
        for thr in (80, 85, 90):
            f.append(float(ctx[f"below{thr}"][lo:e].sum()) / den)
        on = ctx["ih_onsets"]
        sel = (on >= lo) & (on < e)
        f.append(int(sel.sum()))
        sv = ctx["sev_onsets"]
        f.append(int(((sv >= lo) & (sv < e)).sum()))
        f.append(float(ctx["ih_nadir"][sel].mean()) if sel.any() else np.nan)
    return f


IH_NAMES = [f"{n}_{L}min" for L in IH_LOOKBACK_MIN
            for n in ("frac<80", "frac<85", "frac<90", "n_ih", "n_severe", "mean_nadir")]


def onsets(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int)]))
    return np.where(d == 1)[0]


def history_features(ap_on, ds_on, ap_mask, ds_mask, e):
    """Event-history / clustering block (same definition as clustering_prediction.py,
    re-expressed at 1 Hz). This is the CONTROL the drift block must beat."""
    f = []
    for L in (5, 10):
        lo = max(0, e - L * 60 * HZ)
        f.append(int(((ap_on >= lo) & (ap_on < e)).sum()))
        f.append(float(ap_mask[lo:e].mean()))
        f.append(int(((ds_on >= lo) & (ds_on < e)).sum()))
        f.append(float(ds_mask[lo:e].mean()))
    pa, pd_ = ap_on[ap_on < e], ds_on[ds_on < e]
    f.append(float(e - pa[-1]) if len(pa) else float(e))
    f.append(float(e - pd_[-1]) if len(pd_) else float(e))
    return f


HISTORY_NAMES = [f"{n}_{L}min" for L in (5, 10)
                 for n in ("n_apnea", "frac_apnea", "n_desat", "frac_desat")] + \
                ["s_since_apnea", "s_since_desat"]


def stats(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return [0.0] * 5
    return [x.mean(), x.std(), x.min(), x.max(),
            float(np.polyfit(np.arange(len(x)), x, 1)[0])]


def var_stats(x):
    d = np.diff(np.asarray(x, float))
    return [float(np.std(d)) if len(d) else 0.0,
            float(np.mean(np.abs(d))) if len(d) else 0.0]


def periodic_power(env):
    if len(env) < 8:
        return 0.0
    f, P = sp.periodogram(env - env.mean(), fs=2.0)
    return float(P[(f >= 0.01) & (f <= 0.1)].sum() / (P.sum() + 1e-9))


def physio_features(seg, e_1hz, chans):
    """The established precursor block, for reference (Section 4.4 of the thesis)."""
    sl = slice((e_1hz - WIN_S * HZ) * (FS // HZ), e_1hz * (FS // HZ))
    f = []
    for ch, st in [("HR", 200), ("PCO2", 100)]:
        if ch not in chans:
            f += [np.nan] * 7
            continue
        x = seg[ch][sl][::st]
        f += stats(x) + var_stats(x)
    for ch in ("Thorax", "Abdomen", "CPAP"):
        if ch not in chans:
            f += [np.nan] * 8
            continue
        env = np.abs(seg[ch][sl].astype(float))
        k = FS // 2
        e2 = env[: len(env) // k * k].reshape(-1, k).mean(1) if len(env) >= k else env
        f += stats(e2) + var_stats(e2) + [periodic_power(e2)]
    # THE PR CHANNEL. `signal_types` in config/dataset/neonatal.yaml lists six
    # channels for this cohort and this block historically used five of them --
    # PR was simply never added. APPENDED at the end, behind a switch that is
    # OFF by default, so every cached feature file and every number in #23-#26
    # reproduces bit-for-bit; `--with-pr` writes a WIDER cache that is not
    # interchangeable with the old one.
    #
    # DESPITE THE NAME IT IS NOT A RATE. Measured on patient 001: its dominant
    # frequency is 2.54 Hz (= 152/min, i.e. the pulse itself) against HR's
    # smooth 0.10 Hz trend, it ranges 16.6-1027.8, and it moves ~110 units
    # between consecutive seconds. It is the oximeter's PLETHYSMOGRAM WAVEFORM.
    # Subsampling it to 1 Hz the way HR is treated would alias the pulse into
    # noise -- the first version of this block did exactly that and produced a
    # "mean pulse rate" of 511. It is therefore treated like the effort belts:
    # per-0.5 s peak-to-trough AMPLITUDE, which is a perfusion index, and which
    # is the physiologically interesting quantity here because peripheral
    # vasoconstriction accompanies apnea and desaturation.
    if WITH_PR:
        if "PR" not in chans:
            f += [np.nan] * 8
        else:
            x = seg["PR"][sl].astype(float)
            k = FS // 2
            xb = (x[: len(x) // k * k].reshape(-1, k) if len(x) >= k
                  else x.reshape(1, -1))
            amp = xb.max(1) - xb.min(1)
            f += stats(amp) + var_stats(amp) + [periodic_power(amp)]
    return f


PHYSIO_NAMES = [f"{ch}_{n}" for ch in ("HR", "PCO2")
                for n in ("mean", "sd", "min", "max", "slope", "dsd", "dabs")] + \
               [f"{ch}_{n}" for ch in ("Thorax", "Abdomen", "CPAP")
                for n in ("mean", "sd", "min", "max", "slope", "dsd", "dabs", "pb")]
PR_NAMES = [f"PRamp_{n}" for n in
            ("mean", "sd", "min", "max", "slope", "dsd", "dabs", "pb")]


def physio_names():
    """Column names of the PH block, tracking the WITH_PR switch."""
    return PHYSIO_NAMES + (PR_NAMES if WITH_PR else [])


# --------------------------------------------------------------------------
# evaluation (identical protocol to clustering_prediction.py)
# --------------------------------------------------------------------------
def gbm():
    # Early stopping is on (it is ~10x faster than a fixed 300 iterations and
    # regularises at the same time). Its validation split is drawn from the
    # TRAINING rows only -- other patients under LOO, earlier windows under the
    # temporal split -- so it never sees the evaluated data.
    # 120 iterations at lr 0.10 rather than 300 at 0.05: on this data that is
    # ~3x faster (4.6 s vs 12.9 s per LOO fold on 4 cores) and the difference is
    # within fold-to-fold noise. Every block in the ablation uses the identical
    # setting, so the COMPARISON -- which is what the section reports -- is
    # unaffected either way.
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.10, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, n_iter_no_change=10, validation_fraction=0.1,
        random_state=0)


def sample_weights(y):
    p, n = int(y.sum()), int((y == 0).sum())
    if p == 0 or n == 0:
        return np.ones(len(y))
    return np.where(y == 1, len(y) / (2 * p), len(y) / (2 * n))


def loo(X, y, groups):
    a = []
    for pid in IDS:
        tr, te = groups != pid, groups == pid
        if te.sum() == 0 or len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        c = gbm()
        c.fit(X[tr], y[tr], sample_weight=sample_weights(y[tr]))
        a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
    return (float(np.nanmean(a)) if a else np.nan), len(a)


def within_temporal(X, y, groups, times):
    a = []
    for pid in IDS:
        idx = groups == pid
        if idx.sum() == 0:
            continue
        o = np.argsort(times[idx])
        Xp, yp = X[idx][o], y[idx][o]
        n = len(yp)
        tr, te = slice(0, int(0.60 * n)), slice(int(0.70 * n), n)  # 10 % buffer gap
        ytr, yte = yp[tr], yp[te]
        if ytr.sum() >= 5 and yte.sum() >= 5 and (yte == 0).sum() >= 5:
            c = gbm()
            c.fit(Xp[tr], ytr, sample_weight=sample_weights(ytr))
            a.append(roc_auc_score(yte, c.predict_proba(Xp[te])[:, 1]))
    return (float(np.nanmean(a)) if a else np.nan), len(a)


# --------------------------------------------------------------------------
def build(data_path, with_physio):
    DR, IH, HI, PH = [], [], [], []
    groups, times = [], []
    Y = {}
    for H in HORIZONS_S:
        for L in LEADS_S:
            for ep in ("apnea", "ih", "decline", "declabs",
                       "apon", "apdesat", "apnodesat"):
                Y[f"{ep}_{H}_L{L}"] = []
    gain_report = []

    for pid in IDS:
        for si, seg in enumerate(read_data(
                pid, data_path,
                annotations_dir=ANNOTATIONS_DIR,
                record_duration=RECORD_DURATION,
                clock_drift=DRIFT.get(pid) if USE_DRIFT else None)):
            chans = set(seg.keys())
            ctx = spo2_context(seg)
            n = ctx["n"]
            step = FS // HZ
            ap_mask = np.asarray(seg[APNEA_TYPE][::step], bool)
            ds_mask = np.asarray(seg["DESAT"][::step], bool)
            ap_on, ds_on = onsets(ap_mask), onsets(ds_mask)
            # Which apneas are followed by a desaturation. This splits the apnea
            # target into the clinically significant events and the rest, WITHOUT
            # conditioning the evaluation on the future: every window is still
            # scored, only the label changes. (Restricting the test set to
            # apnea-bearing windows would select on a future outcome and could
            # not be reproduced by any deployed monitor.)
            if len(ap_on):
                ap_sig = np.array([
                    bool(((ds_on >= t) & (ds_on <= t + DESAT_LINK_S * HZ)).any())
                    for t in ap_on])
            else:
                ap_sig = np.zeros(0, dtype=bool)
            arti, s, valid = ctx["arti"], ctx["spo2"], ctx["valid"]
            gain_report.append((pid, si, float((s > 100).mean())))

            # All label sets must live on the SAME windows to be comparable, so
            # the bound uses the largest lead+horizon rather than each in turn.
            maxH = (max(HORIZONS_S) + max(LEADS_S)) * HZ
            e = WIN_S * HZ
            while e + maxH <= n:
                w0 = e - WIN_S * HZ
                # skip windows that are mostly artifact or mostly dropout
                if arti[w0:e].mean() <= 0.5 and valid[w0:e].mean() >= 0.5:
                    DR.append(drift_features(ctx, e))
                    IH.append(ih_features(ctx, e))
                    HI.append(history_features(ap_on, ds_on, ap_mask, ds_mask, e))
                    PH.append(physio_features(seg, e, chans) if with_physio
                              else [0.0])
                    groups.append(pid)
                    times.append(si * 1e9 + e)
                    B30 = ctx["B30"][e]
                    on = ctx["ih_onsets"]
                    for H in HORIZONS_S:
                        h = H * HZ
                        for L in LEADS_S:
                            a = e + L * HZ          # target interval [a, a+h)
                            b = a + h
                            Y[f"apnea_{H}_L{L}"].append(int(ap_mask[a:b].any()))
                            # onset-based variants, so A / B / C are comparable
                            in_win = (ap_on >= a) & (ap_on < b)
                            Y[f"apon_{H}_L{L}"].append(int(in_win.any()))
                            Y[f"apdesat_{H}_L{L}"].append(
                                int((in_win & ap_sig).any()))
                            Y[f"apnodesat_{H}_L{L}"].append(
                                int((in_win & ~ap_sig).any()))
                            Y[f"ih_{H}_L{L}"].append(
                                int(((on >= a) & (on < b)).any()))
                            fut = nanmean_slice(s, valid, a, b)
                            Y[f"decline_{H}_L{L}"].append(
                                -1 if (np.isnan(fut) or np.isnan(B30))
                                else int(fut <= B30 - DECLINE_DROP))
                            # absolute variant: no dependence on any feature
                            Y[f"declabs_{H}_L{L}"].append(
                                -1 if np.isnan(fut) else int(fut < DECLINE_ABS))
                e += STRIDE_S * HZ

    out = {
        "DR": np.asarray(DR, float), "IH": np.asarray(IH, float),
        "HI": np.asarray(HI, float), "PH": np.asarray(PH, float),
        "groups": np.asarray(groups), "times": np.asarray(times, float),
    }
    for k, v in Y.items():
        out[f"y_{k}"] = np.asarray(v, int)
    out["_gain"] = np.asarray([g for _, _, g in gain_report], float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", nargs="?",
                    default=r"data/brainimmaturity")
    ap.add_argument("--no-physio", action="store_true",
                    help="skip the 200 Hz physiological block (much faster)")
    ap.add_argument("--cache", default=None, help="npz file to cache features in")
    ap.add_argument("--cohort", choices=["cpap", "robin"], default="cpap",
                    help="robin sets annotations_original/, record_duration=10, "
                         "ACTIVITY-MOVE as the artifact mark, ids 001-019 and "
                         "disables the CPAP clock-drift table")
    ap.add_argument("--apnea_type", default=None,
                    help="prediction target, e.g. APNEA-OBSTRUCTIVE (Robin)")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--endpoints", default="apnea,ih,decline,declabs",
                    help="comma-separated subset of apnea,ih,decline")
    ap.add_argument("--with-pr", action="store_true",
                    help="append the pulse-rate block to PH (widens the cache; "
                         "not interchangeable with caches built without it)")
    ap.add_argument("--skip-main", action="store_true",
                    help="run only the lead sweep (the main lead-0 ablation is "
                         "the expensive half and rarely needs repeating)")
    args = ap.parse_args()

    global APNEA_TYPE, ARTIFACT_TYPE, ANNOTATIONS_DIR, RECORD_DURATION
    global USE_DRIFT, IDS, WITH_PR
    WITH_PR = args.with_pr
    if args.cohort == "robin":
        ARTIFACT_TYPE = "ACTIVITY-MOVE"
        ANNOTATIONS_DIR = "annotations_original"
        RECORD_DURATION = 10       # the mis-declared header, Section 4.1 of the thesis
        USE_DRIFT = False          # the drift table is keyed by CPAP patient id
        IDS = ["%03d" % i for i in range(1, 20)]
    if args.apnea_type:
        APNEA_TYPE = args.apnea_type
    if args.ids:
        IDS = list(args.ids)
    print("cohort=%s target=%s artifact=%s ann=%s rec_dur=%s ids=%d"
          % (args.cohort, APNEA_TYPE, ARTIFACT_TYPE, ANNOTATIONS_DIR,
             RECORD_DURATION, len(IDS)), flush=True)

    if args.cache and os.path.exists(args.cache):
        print(f"loading cached features from {args.cache}", flush=True)
        d = dict(np.load(args.cache, allow_pickle=True))
    else:
        print("building features ...", flush=True)
        d = build(args.data_path, not args.no_physio)
        if args.cache:
            np.savez_compressed(args.cache, **d)
            print(f"cached -> {args.cache}", flush=True)

    DR, IH, HI, PH = d["DR"], d["IH"], d["HI"], d["PH"]
    groups, times = d["groups"], d["times"]
    print(f"\n{len(DR)} windows | drift {DR.shape[1]} | ih {IH.shape[1]} | "
          f"history {HI.shape[1]} | physio {PH.shape[1]} features")
    print(f"SpO2 > 100 % (EDF gain artefact): {d['_gain'].mean()*100:.1f} % of "
          f"samples on average, max {d['_gain'].max()*100:.1f} % in a segment\n")

    have_physio = PH.shape[1] > 1
    blocks = {
        "drift (B)": DR,
        "ih (A)": IH,
        "A + B": np.hstack([DR, IH]),
        "history": HI,
        "history + drift": np.hstack([HI, DR]),
        "history + A + B": np.hstack([HI, DR, IH]),
    }
    if have_physio:
        blocks["physio"] = PH
        blocks["all"] = np.hstack([PH, DR, IH, HI])

    # The `apnea` endpoint carries the headline comparison against the
    # clustering table of #13, which reports 1/2/3 min; the two signal-derived
    # endpoints do not need the third horizon to make their point, and each one
    # costs ~15 leave-one-out fits per block.
    HORIZ = {"apnea": HORIZONS_S, "ih": [60, 120], "decline": [60, 120],
             "declabs": [60, 120]}

    def report(endpoint, H, L, block_set):
        y = d[f"y_{endpoint}_{H}_L{L}"]
        keep = y >= 0                      # `decline` marks unusable as -1
        yk, gk, tk = y[keep], groups[keep], times[keep]
        if len(np.unique(yk)) < 2:
            print(f"  horizon {H}s lead {L}s: single-class, skipped", flush=True)
            return
        print(f"\n  horizon {H}s, lead {L}s  (prevalence {yk.mean()*100:.1f} %, "
              f"n={len(yk)})", flush=True)
        for name in block_set:
            X = blocks[name]
            a1, n1 = loo(X[keep], yk, gk)
            a2, n2 = within_temporal(X[keep], yk, gk, tk)
            print(f"    {name:<18} LOO={a1:.3f} (n={n1:2d})   "
                  f"within-pat={a2:.3f} (n={n2:2d})", flush=True)
        # the single-feature reference: is the whole block worth more than the
        # one number that was actually proposed in supervision?
        j = DRIFT_NAMES.index("M-B30")
        a1, _ = loo(DR[keep][:, [j]], yk, gk)
        a2, _ = within_temporal(DR[keep][:, [j]], yk, gk, tk)
        print(f"    {'[M - B30] alone':<18} LOO={a1:.3f}        "
              f"within-pat={a2:.3f}", flush=True)

    all_blocks = list(blocks)
    for endpoint in args.endpoints.split(","):
        if args.skip_main:
            break
        print(f"\n{'='*74}\nENDPOINT: {endpoint}   (lead 0 -- target starts at "
              f"the window end)\n{'='*74}", flush=True)
        for H in HORIZ.get(endpoint, HORIZONS_S):
            report(endpoint, H, 0, all_blocks)

    # ---- the control that separates prediction from detection ---------------
    # Only the signal-derived endpoints need it: `apnea` is scored by a human
    # from the effort belts, so its target does not mechanically continue the
    # SpO2 trace the features are built from.
    # "A + B" and "history + A + B" are in the sweep as well as the lead-0 pass:
    # the headline "all" figure bundles the ih block with the 38-feature physio
    # block, so without them the contribution of the desaturation-burden
    # features to the hypoxemia endpoints cannot be attributed.
    lead_blocks = [b for b in ("drift (B)", "ih (A)", "A + B", "history",
                               "history + drift", "history + A + B",
                               "all") if b in blocks]
    for endpoint in ("ih", "decline", "declabs"):
        if endpoint not in args.endpoints.split(","):
            continue
        print(f"\n{'='*74}\nLEAD SWEEP: {endpoint}  (does it survive a gap "
              f"between window and target?)\n{'='*74}", flush=True)
        for L in LEADS_S:
            report(endpoint, 60, L, lead_blocks)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
