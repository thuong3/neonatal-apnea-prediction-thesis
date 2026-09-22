"""Apnea prediction from event CLUSTERING (event-history paradigm).

Distinct from the physiological-precursor approach: apneas cluster in time, so
"how much apnea/desaturation happened recently" predicts whether the next few
minutes contain an apnea. This models the apnea point-process autocorrelation
rather than a physiological precursor.

Design (leakage-safe):
  * sliding 60 s window; predict apnea in the next H seconds.
  * event-history features look only at the PAST (up to window end), strictly
    before the prediction horizon; reset at each analysis-segment start.
  * three feature sets are compared: physiological only / history only / both.
  * evaluation: cross-patient (leave-one-out) AND within-patient temporal split
    with a buffer gap (train earlier, test later, middle dropped) to avoid
    train/test adjacency.

Caveat (for the thesis): a positive result here means "an unstable, apnea-prone
period tends to continue" -- NOT that a physiological precursor announces the
next event. See Section 4.4 of the thesis.

Usage: python scripts/clustering_prediction.py <data_path>
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
from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset_v2"
FS = 200
WIN, STRIDE = 60 * FS, 30 * FS
HORIZONS = [60 * FS, 120 * FS, 180 * FS]
MAXH = max(HORIZONS)
LOOKBACKS = [300 * FS, 600 * FS]          # 5 and 10 min event history
MAXLB = max(LOOKBACKS)
IDS = [f"{i:03d}" for i in range(1, 16)]

# CPAP cohort: annotations run ~158 ppm fast against the signals, so the
# event masks this script reads must be drift-corrected too, not just the
# ones the network trains on. See scripts/measure_clock_drift.py.
DRIFT = load_clock_drift()
_BP = sp.butter(2, [5 / (FS / 2), 40 / (FS / 2)], "band")


# ---------- physiological features (same set as the extra-signal analysis) ----
def stats(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return [0, 0, 0, 0, 0]
    return [x.mean(), x.std(), x.min(), x.max(), np.polyfit(np.arange(len(x)), x, 1)[0]]


def var_stats(x):
    d = np.diff(np.asarray(x, float))
    return [np.std(d) if len(d) else 0.0, np.mean(np.abs(d)) if len(d) else 0.0]


def periodic_power(e):
    if len(e) < 8:
        return 0.0
    f, P = sp.periodogram(e - e.mean(), fs=2.0)
    return float(P[(f >= 0.01) & (f <= 0.1)].sum() / (P.sum() + 1e-9))


def hrv(peaks_in_win):
    if len(peaks_in_win) < 6:
        return [0] * 5
    rr = np.diff(peaks_in_win) / FS * 1000.0
    rr = rr[(rr >= 250) & (rr <= 700)]
    if len(rr) < 4:
        return [0] * 5
    d = np.diff(rr)
    return [rr.mean(), rr.std(), np.sqrt(np.mean(d ** 2)),
            np.polyfit(np.arange(len(rr)), rr, 1)[0], float(np.mean(np.abs(d) > 20))]


def physio_features(seg, s, peaks):
    sl = slice(s, s + WIN)
    f = []
    for ch, st in [("SpO2", 100), ("HR", 200), ("PCO2", 100)]:
        f += stats(seg[ch][sl][::st]) + var_stats(seg[ch][sl][::st])
    for ch in ["Thorax", "Abdomen", "CPAP", "ESO"]:
        env = np.abs(seg[ch][sl].astype(float))
        k = FS // 2
        e2 = env[: len(env) // k * k].reshape(-1, k).mean(1) if len(env) >= k else env
        f += stats(e2) + var_stats(e2) + [periodic_power(e2)]
    pw = peaks[(peaks >= s) & (peaks < s + WIN)]
    f += hrv(pw)
    return f


# ---------- event-history features (the clustering signal) --------------------
def onsets(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int)]))
    return np.where(d == 1)[0]


def history_features(ap_on, ds_on, ap_mask, ds_mask, e):
    f = []
    for L in LOOKBACKS:
        lo = max(0, e - L)
        f.append(int(((ap_on >= lo) & (ap_on < e)).sum()))     # #apnea onsets
        f.append(float(ap_mask[lo:e].mean()))                  # apnea time fraction
        f.append(int(((ds_on >= lo) & (ds_on < e)).sum()))     # #desat onsets
        f.append(float(ds_mask[lo:e].mean()))                  # desat time fraction
    prev_ap = ap_on[ap_on < e]
    prev_ds = ds_on[ds_on < e]
    f.append(float((e - prev_ap[-1]) / FS) if len(prev_ap) else float(e / FS))   # s since last apnea
    f.append(float((e - prev_ds[-1]) / FS) if len(prev_ds) else float(e / FS))   # s since last desat
    return f


def seg_peaks(ekg):
    f = np.abs(sp.filtfilt(*_BP, np.asarray(ekg, float)))
    pk, _ = sp.find_peaks(f, distance=int(0.25 * FS), height=np.percentile(f, 90))
    return pk


def gbm():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)


def sw(yy):
    p, n = int(yy.sum()), int((yy == 0).sum())
    return np.where(yy == 1, len(yy) / (2 * p), len(yy) / (2 * n))


def loo(X, y, groups):
    a = []
    for pid in IDS:
        tr, te = groups != pid, groups == pid
        if len(set(y[te].tolist())) < 2:
            continue
        c = gbm(); c.fit(X[tr], y[tr], sample_weight=sw(y[tr]))
        a.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
    return np.nanmean(a)


def within_temporal(X, y, groups, times):
    a = []
    for pid in IDS:
        idx = groups == pid
        o = np.argsort(times[idx])
        Xp, yp = X[idx][o], y[idx][o]
        n = len(yp)
        tr = slice(0, int(0.60 * n))
        te = slice(int(0.70 * n), n)          # 10% buffer gap between train and test
        ytr, yte = yp[tr], yp[te]
        if ytr.sum() >= 5 and yte.sum() >= 5 and (yte == 0).sum() >= 5:
            c = gbm(); c.fit(Xp[tr], ytr, sample_weight=sw(ytr))
            a.append(roc_auc_score(yte, c.predict_proba(Xp[te])[:, 1]))
    return np.nanmean(a)


def main():
    print("Building features (physiological + event-history) ...", flush=True)
    PH, HI, groups, times = [], [], [], []
    Y = {h: [] for h in HORIZONS}
    for pid in IDS:
        for si, seg in enumerate(read_data(pid, DATA, clock_drift=DRIFT.get(pid))):
            ap_mask = seg["APNEA-CENTRAL"].astype(bool)
            ds_mask = seg["DESAT"].astype(bool)
            arti = seg["SIGNAL-ARTIFACT"].astype(bool)
            ap_on, ds_on = onsets(ap_mask), onsets(ds_mask)
            peaks = seg_peaks(seg["EKG"])
            n = len(ap_mask)
            s = 0
            while s + WIN + MAXH <= n:
                e = s + WIN
                if arti[s:e].mean() <= 0.5:
                    PH.append(physio_features(seg, s, peaks))
                    HI.append(history_features(ap_on, ds_on, ap_mask, ds_mask, e))
                    groups.append(pid); times.append(si * 1e9 + s)
                    for h in HORIZONS:
                        Y[h].append(int(ap_mask[e:e + h].any()))
                s += STRIDE
    PH, HI = np.array(PH, float), np.array(HI, float)
    groups, times = np.array(groups), np.array(times, float)
    Y = {h: np.array(v) for h, v in Y.items()}
    BOTH = np.concatenate([PH, HI], axis=1)
    print(f"{len(PH)} windows | physio {PH.shape[1]} feat | history {HI.shape[1]} feat\n", flush=True)

    for h in HORIZONS:
        y = Y[h]
        print(f"### horizon {h // FS}s (prevalence {y.mean()*100:.0f}%)", flush=True)
        for name, X in [("physio-only ", PH), ("history-only", HI), ("both        ", BOTH)]:
            print(f"  {name}:  LOO={loo(X, y, groups):.3f}  |  within-pat temporal={within_temporal(X, y, groups, times):.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
