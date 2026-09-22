"""EFFORT BASELINE DRIFT as a predictor -- the Adams/Poets hypothesis.

WHERE THIS COMES FROM
    Poets (Sleep Med 2010, section 4) reports the finding of Adams et al.
    (Pediatr Res 1997): **62 % of events with SpO2 < 80 % were preceded by
    breaths with a tidal volume below 50 % of baseline**. That is a precursor
    claim -- but about VENTILATION, not about saturation. Poets' own timing data
    explain why the distinction matters: the interval from apnea onset to
    desaturation onset has a median of 0.8 s, so SpO2 cannot warn ahead of an
    apnea, whereas a fall in tidal volume precedes the event by construction.

WHY IT HAS NEVER BEEN TESTED HERE
    The precursor block of Section 4.4 of the thesis does compute effort-belt
    envelope statistics -- but `stats(e2)` in scripts/precursor_gbm.py (and in
    physio_features of the SpO2 script) is evaluated WITHIN THE 30-60 s WINDOW
    ONLY: mean, SD, min, max, slope. There is no reference to the infant's own
    longer-run level. Adams' criterion is inherently relative to a baseline, so
    the form that carries the claim was absent.

    This is exactly the blind spot Section 4.6 of the thesis identified for SpO2 (per-window
    z-scoring had removed absolute level by construction) -- and there, the
    relative form `M - B_T` was the one that survived the lead control while the
    window-local features did not. So the same construction is applied here.

WHAT IS DIFFERENT FROM THE SpO2 VERSION
  * RATIO, NOT DIFFERENCE. SpO2 is an absolute percentage, so `M - B` is
    meaningful across recordings. Effort-belt amplitude has an ARBITRARY
    PER-RECORDING GAIN (an inductance belt reports no physical unit), so only a
    ratio is comparable. The headline feature is therefore `log(E / E_T)`, and
    Adams' "below 50 % of baseline" is log(0.5) = -0.69 on that scale.
  * A DROPOUT MASK IS MANDATORY, not optional. A disconnected or frozen belt
    reports near-zero amplitude -- i.e. it looks EXACTLY like the absent
    respiratory effort we are trying to detect. Unmasked, this analysis would
    "predict" apnea from sensor failures. Blocks whose raw variance is zero are
    therefore marked invalid and never enter E or its baseline. (The scale of
    the hazard: figures/01_data_quality found 9.3 h of unflagged dropout.)

ENDPOINTS
    `apnea`   scored APNEA-CENTRAL onset           (the thesis' primary target)
    `sev`     SpO2 < 80 % for >= 5 s -- ADAMS' OWN ENDPOINT, the faithful test
    `ih`      SpO2 < 90 % for >= 5 s
    `decline` mean SpO2 >= 3 % below B30
The lead sweep of Section 4.6 of the thesis is applied unchanged: without a gap between window and
target, a signal-derived endpoint is partly detection.

Usage:
    python scripts/effort_baseline_prediction.py [<data_path>] [--cache f.npz]
                                                 [--ids 001 002] [--endpoints ...]
"""
import argparse
import os
import sys

import numpy as np
from scipy import signal as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from src.neonatal_utils import read_data  # noqa: E402
# The SpO2 script is the reference implementation of the causal baseline, the
# event definitions and the evaluation protocol. Importing rather than copying
# keeps the two analyses literally identical wherever they should be.
import spo2_baseline_prediction as S  # noqa: E402

FS, HZ = S.FS, S.HZ
WIN_S, STRIDE_S = S.WIN_S, S.STRIDE_S
HORIZONS_S, LEADS_S = S.HORIZONS_S, S.LEADS_S
BASE_T_MIN = S.BASE_T_MIN

EFFORT_CHANS = ["Thorax", "Abdomen"]
ENV_HZ = 10                  # decimate to 10 Hz: the belts' native rate, and
                             # Nyquist 5 Hz still covers neonatal breathing
_BP_RESP = sp.butter(2, [0.3 / (ENV_HZ / 2), 3.0 / (ENV_HZ / 2)], "band")
ADAMS_FRAC = 0.5             # Adams et al.: tidal volume < 50 % of baseline
ADAMS_LOOKBACK_S = [60, 300]
EPS = 1e-9


# --------------------------------------------------------------------------
# effort envelope
# --------------------------------------------------------------------------
def effort_context(seg, ch):
    """1 Hz respiratory-effort amplitude for one belt, plus its baselines.

    E(t) = SD of the band-passed belt signal within second t. That is the
    amplitude of the respiratory excursion -- a crude tidal-volume proxy, which
    is the quantity Adams' criterion is stated in.
    """
    x = np.asarray(seg[ch], float)[:: FS // ENV_HZ]          # -> 10 Hz
    n_blocks = len(x) // ENV_HZ
    if n_blocks < 2:
        return None
    xb = x[: n_blocks * ENV_HZ].reshape(n_blocks, ENV_HZ)

    # Validity BEFORE filtering: a block with zero raw variance is a frozen or
    # unplugged belt, not a quiet infant. Filter ringing would otherwise give
    # such a block a small non-zero amplitude and hide it.
    valid = xb.std(axis=1) > 0

    xf = sp.filtfilt(*_BP_RESP, np.nan_to_num(x))
    env = xf[: n_blocks * ENV_HZ].reshape(n_blocks, ENV_HZ).std(axis=1)
    valid &= np.isfinite(env)

    arti = np.asarray(seg["SIGNAL-ARTIFACT"], bool)[:: FS // ENV_HZ]
    arti_b = arti[: n_blocks * ENV_HZ].reshape(n_blocks, ENV_HZ).mean(axis=1) > 0.5

    ctx = {"E": env, "valid": valid, "arti": arti_b, "n": n_blocks}
    base_ok = valid & ~arti_b
    for T in BASE_T_MIN:
        # Same estimator as the SpO2 baseline: an upper percentile of the recent
        # past. Apneas and hypopneas are excursions into the LOWER tail of
        # effort amplitude just as desaturations are for SpO2, so the 75th
        # percentile again tracks "the level between the events".
        ctx["E%d" % T] = S.causal_percentile(env, base_ok, T * 60, S.BASE_PCTL)
    for T, back_s in [(30, 600), (60, 1800)]:
        b = ctx["E%d" % T]
        sh = np.full_like(b, np.nan)
        k = back_s * HZ
        with np.errstate(invalid="ignore", divide="ignore"):
            sh[k:] = np.log((b[k:] + EPS) / (b[:-k] + EPS))
        ctx["dE%d" % T] = sh * (600.0 / back_s)
    return ctx


def effort_features(ctx, e):
    """Where is the effort amplitude now, relative to the infant's own normal."""
    if ctx is None:
        return [np.nan] * N_PER_CHAN
    E, valid = ctx["E"], ctx["valid"]
    w0 = e - WIN_S * HZ
    seg = E[w0:e][valid[w0:e]]
    f = []
    M = float(seg.mean()) if len(seg) else np.nan
    f.append(M)
    for T in BASE_T_MIN:
        B = ctx["E%d" % T][e]
        f.append(B)
        # THE HEADLINE FEATURE. Log-ratio, not difference: belt gain is
        # arbitrary per recording, so only the ratio is comparable across
        # infants. Adams' "< 50 % of baseline" is log(0.5) = -0.69 here.
        f.append(np.log((M + EPS) / (B + EPS))
                 if np.isfinite(M) and np.isfinite(B) else np.nan)
    # Adams' criterion as a burden: what fraction of the recent past was the
    # amplitude below half of its own baseline?
    B30 = ctx["E30"][e]
    for L in ADAMS_LOOKBACK_S:
        lo = max(0, e - L * HZ)
        m = valid[lo:e]
        if m.any() and np.isfinite(B30):
            f.append(float((E[lo:e][m] < ADAMS_FRAC * B30).mean()))
        else:
            f.append(np.nan)
    f += [ctx["dE30"][e], ctx["dE60"][e]]
    if len(seg) >= 5:
        f += [float(seg.std() / (M + EPS)),                      # coeff of var
              float(np.polyfit(np.arange(len(seg)), seg, 1)[0] * 60.0 / (M + EPS)),
              float(np.log((seg.min() + EPS) / (B30 + EPS)))
              if np.isfinite(B30) else np.nan]
    else:
        f += [np.nan] * 3
    return f


PER_CHAN_NAMES = (["M"]
                  + [n for T in BASE_T_MIN for n in ("E%d" % T, "logM/E%d" % T)]
                  + ["adams_frac_%ds" % L for L in ADAMS_LOOKBACK_S]
                  + ["dE30", "dE60", "cv_win", "slope_win", "logmin/E30"])
N_PER_CHAN = len(PER_CHAN_NAMES)
EFFORT_NAMES = ["%s_%s" % (ch, n) for ch in EFFORT_CHANS for n in PER_CHAN_NAMES]


# --------------------------------------------------------------------------
def build(data_path, ids):
    EF, HI, DR = [], [], []
    groups, times = [], []
    Y = {}
    for H in HORIZONS_S:
        for L in LEADS_S:
            for ep in ("apnea", "sev", "ih", "decline"):
                Y["%s_%d_L%d" % (ep, H, L)] = []

    for pid in ids:
        for si, seg in enumerate(read_data(pid, data_path,
                                           clock_drift=S.DRIFT.get(pid))):
            sctx = S.spo2_context(seg)            # endpoints + the SpO2 control
            ectx = {ch: (effort_context(seg, ch) if ch in seg else None)
                    for ch in EFFORT_CHANS}
            n = min([sctx["n"]] + [c["n"] for c in ectx.values() if c])
            step = FS // HZ
            ap_mask = np.asarray(seg["APNEA-CENTRAL"][::step], bool)[:n]
            ds_mask = np.asarray(seg["DESAT"][::step], bool)[:n]
            ap_on, ds_on = S.onsets(ap_mask), S.onsets(ds_mask)
            s, valid, arti = sctx["spo2"], sctx["valid"], sctx["arti"]

            maxH = (max(HORIZONS_S) + max(LEADS_S)) * HZ
            e = WIN_S * HZ
            while e + maxH <= n:
                w0 = e - WIN_S * HZ
                e_ok = any(c is not None and c["valid"][w0:e].mean() >= 0.5
                           for c in ectx.values())
                if arti[w0:e].mean() <= 0.5 and e_ok:
                    row = []
                    for ch in EFFORT_CHANS:
                        row += effort_features(ectx[ch], e)
                    EF.append(row)
                    HI.append(S.history_features(ap_on, ds_on, ap_mask, ds_mask, e))
                    DR.append(S.drift_features(sctx, e))
                    groups.append(pid)
                    times.append(si * 1e9 + e)
                    B30 = sctx["B30"][e]
                    ih_on, sev_on = sctx["ih_onsets"], sctx["sev_onsets"]
                    for H in HORIZONS_S:
                        h = H * HZ
                        for L in LEADS_S:
                            a, b = e + L * HZ, e + L * HZ + h
                            Y["apnea_%d_L%d" % (H, L)].append(
                                int(ap_mask[a:b].any()))
                            Y["ih_%d_L%d" % (H, L)].append(
                                int(((ih_on >= a) & (ih_on < b)).any()))
                            Y["sev_%d_L%d" % (H, L)].append(
                                int(((sev_on >= a) & (sev_on < b)).any()))
                            fut = S.nanmean_slice(s, valid, a, b)
                            Y["decline_%d_L%d" % (H, L)].append(
                                -1 if (np.isnan(fut) or np.isnan(B30))
                                else int(fut <= B30 - S.DECLINE_DROP))
                e += STRIDE_S * HZ
        print("  %s done (%d windows so far)" % (pid, len(EF)), flush=True)

    out = {"EF": np.asarray(EF, float), "HI": np.asarray(HI, float),
           "DR": np.asarray(DR, float),
           "groups": np.asarray(groups), "times": np.asarray(times, float)}
    for k, v in Y.items():
        out["y_" + k] = np.asarray(v, int)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", nargs="?",
                    default=r"data/brainimmaturity")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--endpoints", default="apnea,sev,ih,decline")
    ap.add_argument("--skip-main", action="store_true",
                    help="run only the lead sweep; the lead-0 apnea "
                         "number is not trustworthy on its own")
    args = ap.parse_args()
    ids = args.ids or S.IDS

    if args.cache and os.path.exists(args.cache):
        print("loading cached features from %s" % args.cache, flush=True)
        d = dict(np.load(args.cache, allow_pickle=True))
    else:
        print("building effort features ...", flush=True)
        d = build(args.data_path, ids)
        if args.cache:
            np.savez_compressed(args.cache, **d)
            print("cached -> %s" % args.cache, flush=True)

    EF, HI, DR = d["EF"], d["HI"], d["DR"]
    groups, times = d["groups"], d["times"]
    print("\n%d windows | effort %d | history %d | spo2-drift %d features"
          % (len(EF), EF.shape[1], HI.shape[1], DR.shape[1]))
    print("effort features finite: %.1f %% (NaN = belt dropout or baseline "
          "not yet defined)\n" % (np.isfinite(EF).mean() * 100))

    blocks = {
        "effort (E)": EF,
        "history": HI,
        "history + effort": np.hstack([HI, EF]),
        "spo2-drift (B)": DR,
        "spo2-drift + effort": np.hstack([DR, EF]),
        "all": np.hstack([EF, HI, DR]),
    }

    def report(endpoint, H, L, names):
        y = d["y_%s_%d_L%d" % (endpoint, H, L)]
        keep = y >= 0
        yk, gk, tk = y[keep], groups[keep], times[keep]
        if len(np.unique(yk)) < 2:
            print("  horizon %ds lead %ds: single-class, skipped" % (H, L),
                  flush=True)
            return
        print("\n  horizon %ds, lead %ds  (prevalence %.1f %%, n=%d)"
              % (H, L, yk.mean() * 100, len(yk)), flush=True)
        for name in names:
            X = blocks[name]
            a1, n1 = S.loo(X[keep], yk, gk)
            a2, n2 = S.within_temporal(X[keep], yk, gk, tk)
            print("    %-20s LOO=%.3f (n=%2d)   within-pat=%.3f (n=%2d)"
                  % (name, a1, n1, a2, n2), flush=True)
        # The single features that carry Adams' claim, on their own.
        for probe in ("Thorax_logM/E30", "Thorax_adams_frac_300s"):
            j = EFFORT_NAMES.index(probe)
            a1, _ = S.loo(EF[keep][:, [j]], yk, gk)
            a2, _ = S.within_temporal(EF[keep][:, [j]], yk, gk, tk)
            print("    [%s] alone  LOO=%.3f   within-pat=%.3f" % (probe, a1, a2),
                  flush=True)

    eps = args.endpoints.split(",")
    for endpoint in eps:
        if args.skip_main:
            break
        print("\n%s\nENDPOINT: %s   (lead 0)\n%s" % ("=" * 74, endpoint, "=" * 74),
              flush=True)
        for H in HORIZONS_S:
            report(endpoint, H, 0, list(blocks))

    # `apnea` IS IN THIS LIST, and that is a deliberate departure from the
    # SpO2 script. There the sweep skips `apnea` on the grounds that the
    # target is human-scored and so does not mechanically continue the SpO2
    # trace the features are built from. That argument does NOT transfer
    # here: the annotator scored central apnea FROM THORAX/ABDOMEN MOVEMENT
    # (personal communication, see scripts/make_gap_histogram.py) -- the very
    # signal these features measure. At lead 0 an apnea beginning just after
    # the window is by definition a continuation of the effort trace, so the
    # lead-0 apnea number is part prediction, part detection, and must never
    # be reported on its own.
    for endpoint in ("apnea", "sev", "ih", "decline"):
        if endpoint not in eps:
            continue
        print("\n%s\nLEAD SWEEP: %s\n%s" % ("=" * 74, endpoint, "=" * 74),
              flush=True)
        for L in LEADS_S:
            report(endpoint, 60, L,
                   ["effort (E)", "history", "spo2-drift (B)",
                    "spo2-drift + effort", "all"])
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
