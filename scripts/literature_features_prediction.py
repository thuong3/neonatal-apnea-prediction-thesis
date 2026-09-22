"""Literature-backed features for apnea / apnea-prone-state prediction.

Adds three feature families from the neonatal literature (see LITERATURE.md):
  (L1) HR-SpO2 cross-correlation at lags <=30 s      -- Fairchild & Lake 2018
  (L2) time-series "shape" features on SpO2 and HR   -- Pre-Vent HCTSA idea
       (autocorrelation, low-frequency power, spectral entropy)
  (L3) cardiorespiratory coupling (effort <-> HR)    -- Varisco 2024

We compare feature GROUPS separately (an ablation) so it is clear what helps:
  physio(base) | +literature(L1-L3) | +event-history(clustering) | all
Two tasks: state prediction (apnea in next 2 min) and event prediction (15 s
before onset). Evaluation is leakage-safe: cross-patient leave-one-out and
within-patient temporal split with a buffer gap.

Usage: python scripts/literature_features_prediction.py <data_path>
"""
import os
import sys
import warnings

import numpy as np
from scipy import signal as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from src.neonatal_utils import create_time_windows, load_clock_drift, read_data  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset_v2"
FS = 200
IDS = [f"{i:03d}" for i in range(1, 16)]

# CPAP cohort: annotations run ~158 ppm fast against the signals, so the
# event masks this script reads must be drift-corrected too, not just the
# ones the network trains on. See scripts/measure_clock_drift.py.
DRIFT = load_clock_drift()
ADVERSE, CUTTER = ["APNEA-CENTRAL"], ["SIGNAL-ARTIFACT"]
_BP = sp.butter(2, [5 / (FS / 2), 40 / (FS / 2)], "band")


# ----------------------- small helpers -----------------------
def ds(x, target):
    """Downsample a 200 Hz array to `target` Hz by block-averaging."""
    k = FS // target
    x = np.asarray(x, float)
    return x[: len(x) // k * k].reshape(-1, k).mean(1) if len(x) >= k else x


def stats(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return [0, 0, 0, 0, 0]
    return [x.mean(), x.std(), x.min(), x.max(), np.polyfit(np.arange(len(x)), x, 1)[0]]


def var_stats(x):
    d = np.diff(np.asarray(x, float))
    return [np.std(d) if len(d) else 0.0, np.mean(np.abs(d)) if len(d) else 0.0]


def hrv_from_peaks(pk):
    if len(pk) < 6:
        return [0] * 5
    rr = np.diff(pk) / FS * 1000.0
    rr = rr[(rr >= 250) & (rr <= 700)]
    if len(rr) < 4:
        return [0] * 5
    d = np.diff(rr)
    return [rr.mean(), rr.std(), np.sqrt(np.mean(d ** 2)), np.polyfit(np.arange(len(rr)), rr, 1)[0], float(np.mean(np.abs(d) > 20))]


# ----------------------- PHYSIO (baseline) -----------------------
def physio(seg, sl, peaks):
    f = []
    for ch, st in [("SpO2", 100), ("HR", 200), ("PCO2", 100)]:
        f += stats(seg[ch][sl][::st]) + var_stats(seg[ch][sl][::st])
    for ch in ["Thorax", "Abdomen", "CPAP", "ESO"]:
        env = np.abs(seg[ch][sl].astype(float))
        e2 = ds(env, 2)
        f += stats(e2) + var_stats(e2)
    pw = peaks[(peaks >= sl.start) & (peaks < sl.stop)]
    f += hrv_from_peaks(pw)
    return f


# ----------------------- L1: HR-SpO2 cross-correlation -----------------------
def xcorr_hr_spo2(seg, sl):
    a, b = ds(seg["HR"][sl], 1), ds(seg["SpO2"][sl], 1)   # both at 1 Hz
    n = min(len(a), len(b))
    if n < 10:
        return [0, 0, 0, 0]
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    cc = np.correlate(a, b, "full") / n            # normalised cross-correlation
    lags = np.arange(-(n - 1), n)
    m = np.abs(lags) <= 30                          # lags up to +/-30 s
    ccm = cc[m]
    return [ccm.max(), ccm.min(), float(lags[m][np.argmax(ccm)]), float(np.mean(np.abs(ccm)))]


# ----------------------- L2: time-series shape features -----------------------
def ts_shape(x, target=2):
    x = ds(x, target)
    x = x - x.mean()
    if len(x) < 16 or x.std() < 1e-9:
        return [0, 0, 0, 0, 0]
    xn = x / (x.std() + 1e-9)

    def ac(lag):
        return float(np.corrcoef(xn[:-lag], xn[lag:])[0, 1]) if 0 < lag < len(xn) else 0.0

    f, P = sp.periodogram(xn, fs=target)
    P = P / (P.sum() + 1e-9)
    spec_ent = float(-np.sum(P[P > 0] * np.log(P[P > 0])))
    lowf = float(P[(f > 0) & (f <= 0.04)].sum())   # slow / periodic-breathing band
    return [ac(1 * target), ac(5 * target), ac(10 * target), spec_ent, lowf]


def ts_features(seg, sl):
    return ts_shape(seg["SpO2"][sl]) + ts_shape(seg["HR"][sl])


# ----------------------- L3: cardiorespiratory coupling -----------------------
def coupling(seg, sl):
    e = np.abs(ds(seg["Thorax"][sl], 2))
    h = ds(seg["HR"][sl], 2)
    n = min(len(e), len(h))
    if n < 16 or e[:n].std() < 1e-9 or h[:n].std() < 1e-9:
        return [0, 0]
    e, h = e[:n], h[:n]
    corr = np.nan_to_num(float(np.corrcoef(e, h)[0, 1]))
    try:
        fco, C = sp.coherence(e, h, fs=2.0, nperseg=min(64, n))
        coh = np.nan_to_num(float(np.nanmean(C[(fco >= 0.01) & (fco <= 0.2)])))
    except Exception:
        coh = 0.0
    return [corr, coh]


def lit_features(seg, sl):
    f = xcorr_hr_spo2(seg, sl) + ts_features(seg, sl) + coupling(seg, sl)
    return list(np.nan_to_num(np.array(f, float)))


# ----------------------- event history (clustering) -----------------------
def onsets(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int)]))
    return np.where(d == 1)[0]


def history(ap_on, ds_on, ap_mask, ds_mask, e):
    f = []
    for L in [300 * FS, 600 * FS]:
        lo = max(0, e - L)
        f += [int(((ap_on >= lo) & (ap_on < e)).sum()), float(ap_mask[lo:e].mean()),
              int(((ds_on >= lo) & (ds_on < e)).sum()), float(ds_mask[lo:e].mean())]
    pa, pd = ap_on[ap_on < e], ds_on[ds_on < e]
    f += [float((e - pa[-1]) / FS) if len(pa) else e / FS, float((e - pd[-1]) / FS) if len(pd) else e / FS]
    return f


def seg_peaks(ekg):
    fl = np.abs(sp.filtfilt(*_BP, np.asarray(ekg, float)))
    pk, _ = sp.find_peaks(fl, distance=int(0.25 * FS), height=np.percentile(fl, 90))
    return pk


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)


def sw(yy):
    p, n = int(yy.sum()), int((yy == 0).sum())
    return np.where(yy == 1, len(yy) / (2 * p), len(yy) / (2 * n))


def loo(X, y, g):
    a = []
    for pid in IDS:
        tr, te = g != pid, g == pid
        if len(set(y[te].tolist())) < 2:
            continue
        c = gbm(); c.fit(X[tr], y[tr], sample_weight=sw(y[tr]))
        a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
    return np.nanmean(a)


def within(X, y, g, t):
    a = []
    for pid in IDS:
        idx = g == pid
        o = np.argsort(t[idx]); Xp, yp = X[idx][o], y[idx][o]; n = len(yp)
        tr, te = slice(0, int(0.60 * n)), slice(int(0.70 * n), n)
        if yp[tr].sum() >= 5 and yp[te].sum() >= 5 and (yp[te] == 0).sum() >= 5:
            c = gbm(); c.fit(Xp[tr], yp[tr], sample_weight=sw(yp[tr]))
            a.append(roc_auc_score(yp[te], c.predict_proba(Xp[te])[:, 1]))
    return np.nanmean(a)


def report(name, PH, LI, HI, y, g, t):
    groups = {"physio": PH, "+literature": np.concatenate([PH, LI], 1),
              "+history": np.concatenate([PH, HI], 1), "all": np.concatenate([PH, LI, HI], 1)}
    print(f"\n### {name} (prevalence {y.mean()*100:.0f}%)", flush=True)
    for gn, X in groups.items():
        print(f"  {gn:<12}: LOO={loo(X, y, g):.3f} | within-pat temporal={within(X, y, g, t):.3f}", flush=True)


def main():
    # ---- TASK A: apnea-prone STATE (apnea in next 120 s), 60 s sliding window
    print("Building STATE features ...", flush=True)
    WIN, STRIDE, HOR = 60 * FS, 30 * FS, 120 * FS
    PH, LI, HI, y, g, t = [], [], [], [], [], []
    for pid in IDS:
        for si, seg in enumerate(read_data(pid, DATA, clock_drift=DRIFT.get(pid))):
            ap, dsm, art = seg["APNEA-CENTRAL"].astype(bool), seg["DESAT"].astype(bool), seg["SIGNAL-ARTIFACT"].astype(bool)
            apo, dso, pk, n = onsets(ap), onsets(dsm), seg_peaks(seg["EKG"]), len(ap)
            s = 0
            while s + WIN + HOR <= n:
                e = s + WIN; sl = slice(s, e)
                if art[sl].mean() <= 0.5:
                    PH.append(physio(seg, sl, pk)); LI.append(lit_features(seg, sl)); HI.append(history(apo, dso, ap, dsm, e))
                    y.append(int(ap[e:e + HOR].any())); g.append(pid); t.append(si * 1e9 + s)
                s += STRIDE
    PH, LI, HI = np.array(PH, float), np.array(LI, float), np.array(HI, float)
    y, g, t = np.array(y), np.array(g), np.array(t, float)
    print(f"{len(y)} windows | physio {PH.shape[1]} | literature {LI.shape[1]} | history {HI.shape[1]}", flush=True)
    report("STATE next 2 min", PH, LI, HI, y, g, t)

    # ---- TASK B: EVENT prediction 15 s before onset (physio vs +literature)
    print("\nBuilding EVENT features (15 s lead) ...", flush=True)
    LAG, W2, AWAY = 3000, 6000, 36000
    PH2, LI2, y2, g2, t2 = [], [], [], [], []
    for pid in IDS:
        for si, seg in enumerate(read_data(pid, DATA, clock_drift=DRIFT.get(pid))):
            anti = 1 - np.column_stack([seg[e] for e in ADVERSE]).max(1)
            cut = 1 - np.column_stack([seg[e] for e in (ADVERSE + CUTTER)]).max(1)
            pk = seg_peaks(seg["EKG"])
            for r in create_time_windows(anti, cut, W2, AWAY, LAG):
                PH2.append(physio(seg, r["slice"], pk)); LI2.append(lit_features(seg, r["slice"]))
                y2.append(r["label"]); g2.append(pid); t2.append(si * 1e9 + r["slice"].start)
    PH2, LI2 = np.array(PH2, float), np.array(LI2, float)
    y2, g2, t2 = np.array(y2), np.array(g2), np.array(t2, float)
    z = np.zeros((len(y2), 0))
    report("EVENT 15 s before", PH2, LI2, z, y2, g2, t2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
