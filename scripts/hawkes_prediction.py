"""Hawkes self-exciting point-process model for apnea-prone-state prediction.

Central apneas cluster in time. A Hawkes process models the apnea ONSET times
directly: each apnea transiently raises the conditional intensity (rate) of the
next one, which then decays exponentially:

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

mu = baseline rate, alpha = excitation, beta = decay. Parameters are fit by
maximum likelihood. The intensity lambda(t) at a window end is used as the
prediction score for "an apnea occurs in the next H seconds". This is the
principled, generative version of the ad-hoc event-history features
(clustering_prediction.py) and is an original contribution of the thesis.

Leakage-safe evaluation:
  * cross-patient: fit (mu,alpha,beta) on the OTHER patients, score the test
    patient's windows using that patient's own preceding onsets;
  * within-patient temporal: fit on the first 60 % of each segment, score
    windows in the last 40 %.

Usage: python scripts/hawkes_prediction.py <data_path>
"""
import os
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset_v2"
FS = 200
IDS = [f"{i:03d}" for i in range(1, 16)]

# CPAP cohort: annotations run ~158 ppm fast against the signals, so the
# event masks this script reads must be drift-corrected too, not just the
# ones the network trains on. See scripts/measure_clock_drift.py.
DRIFT = load_clock_drift()
WIN_START, STRIDE = 60.0, 30.0          # seconds
HORIZONS = [60.0, 120.0]                # predict apnea in next 1 / 2 min


def hawkes_nll(p, seglist):
    """Negative log-likelihood of a univariate exponential Hawkes process."""
    mu, al, be = p
    if mu <= 1e-9 or al <= 1e-9 or be <= 1e-9 or al >= be:   # stationarity: alpha<beta
        return 1e12
    tot = 0.0
    for ev, T in seglist:
        comp = mu * T + (al / be) * np.sum(1 - np.exp(-be * (T - ev))) if len(ev) else mu * T
        if len(ev) == 0:
            tot += comp
            continue
        A = 0.0
        ll = 0.0
        for i in range(len(ev)):
            if i > 0:
                A = np.exp(-be * (ev[i] - ev[i - 1])) * (1 + A)
            lam = mu + al * A
            ll += np.log(lam) if lam > 0 else -50.0
        tot += comp - ll
    return tot


def fit_hawkes(seglist):
    n = sum(len(ev) for ev, _ in seglist)
    T = sum(t for _, t in seglist)
    base = max(n / T, 1e-4) if T > 0 else 1e-3
    best = None
    # Apneas cluster over MINUTES, so start beta small (slow decay):
    # beta in {0.002, 0.01, 0.05} = decay timescales ~500 / 100 / 20 s, with
    # alpha kept below beta (stationarity).
    for x0 in ([base * 0.6, 0.0015, 0.002], [base * 0.6, 0.007, 0.01],
               [base * 0.6, 0.03, 0.05], [base * 0.8, 0.0007, 0.001]):
        r = minimize(hawkes_nll, x0, args=(seglist,), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def intensity(t, ev, mu, al, be):
    e = ev[ev < t]
    return mu + al * np.sum(np.exp(-be * (t - e)))


def score_windows(seglist, mu, al, be, H, t_lo_frac=0.0):
    scores, labels = [], []
    for ev, T in seglist:
        t = max(WIN_START, t_lo_frac * T)
        while t + H <= T:
            scores.append(intensity(t, ev, mu, al, be))
            labels.append(int(np.any((ev > t) & (ev <= t + H))))
            t += STRIDE
    return np.array(scores), np.array(labels)


def main():
    print("Reading apnea onset times ...", flush=True)
    data = {}
    for pid in IDS:
        segs = []
        for seg in read_data(pid, DATA, clock_drift=DRIFT.get(pid)):
            m = seg["APNEA-CENTRAL"].astype(bool)
            on = np.where(np.diff(np.concatenate([[0], m.astype(int)])) == 1)[0] / FS
            segs.append((on.astype(float), len(m) / FS))
        data[pid] = segs
        print(f"  {pid}: {sum(len(o) for o, _ in segs)} apnea onsets", flush=True)

    for H in HORIZONS:
        cross, wtemp = [], []
        for pid in IDS:
            # cross-patient: fit on others, score this patient's windows
            train = [s for q in IDS if q != pid for s in data[q]]
            mu, al, be = fit_hawkes(train)
            sc, lb = score_windows(data[pid], mu, al, be, H)
            if len(set(lb.tolist())) > 1:
                cross.append(roc_auc_score(lb, sc))
            # within-patient temporal: fit on first 60 %, score last 40 %
            trainseg = [(ev[ev < 0.6 * T], 0.6 * T) for ev, T in data[pid]]
            if sum(len(ev) for ev, _ in trainseg) >= 5:
                mu2, al2, be2 = fit_hawkes(trainseg)
                sc2, lb2 = score_windows(data[pid], mu2, al2, be2, H, t_lo_frac=0.70)
                if len(set(lb2.tolist())) > 1:
                    wtemp.append(roc_auc_score(lb2, sc2))
        print(f"\nHORIZON {int(H)}s:  cross-patient AUC = {np.nanmean(cross):.3f}  |  "
              f"within-patient temporal AUC = {np.nanmean(wtemp):.3f}", flush=True)
    print("(reference: event-history GBM was cross-patient ~0.64-0.68)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
