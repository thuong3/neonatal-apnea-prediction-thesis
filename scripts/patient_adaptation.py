"""PATIENT-SPECIFIC ADAPTATION: train on the new infant's OWN first K apneas.

THE PROPOSAL (supervision, Aug 2026)
    Do not hold the new infant out completely. Give the model the infant's first
    K apneas as TRAINING data, then ask it to predict apneas K+1 onward. Because
    K events are nothing against ~1500 pooled ones, oversample them: duplicate
    each calibration window M times so the fit actually feels this infant.

WHY THIS IS NOT A RE-RUN OF SECTION 4.4 OR SECTION 4.5 OF THE THESIS
    Both earlier personalisation experiments left the FIT untouched:
      * #25 re-expressed features against the infant's own normal and moved the
        decision threshold; the model was still trained on the other 14 infants.
      * Section 4.5 of the thesis (`adabn_patient`) re-estimated BatchNorm
        statistics on the infant's early windows -- label-free, no gradient step.
    Neither ever put a LABELLED window of the target infant into the training
    objective. That is precisely what is new here, and the oversampling factor M
    is the knob that decides whether those labels are audible at all.

THREE DESIGN POINTS, EACH OF WHICH WOULD INVALIDATE THE GRID IF GOT WRONG

  (1) THE EVALUATION SET MUST NOT MOVE WITH K. Read literally, the proposal
      tests K = 0 on all of the infant and K = 10 on only its later apneas, so
      the rows differ in TWO ways and any trend could be "late apneas are
      easier". Every cell here is therefore scored on the SAME windows: those
      after the K_MAX-th event, whatever K the row used for training. The K = 0
      row is the paired control and differs from every other row only in what
      entered the fit. This is the same discipline `zero_shot_late` enforces in
      Section 4.5 of the thesis, where it was also not optional.

  (2) WHAT GETS OVERSAMPLED CHANGES WHAT IS BEING MEASURED. Three arms are run:
        `positives` -- literally the proposal: only the K apnea-positive windows
                       are added, each M times.
        `segment`   -- the whole calibration period up to the K-th event,
                       positives AND negatives, all M times.
        `donor`     -- THE NEGATIVE CONTROL, see below.
      The first two are different experiments. `positives` raises the positive
      prior, which mostly moves the OPERATING POINT -- and AuROC is
      threshold-invariant (#25.3), so it has limited room to move the ranking by
      construction. `segment` is what can teach "this infant's normal vs this
      infant's apnea", the contrast a per-patient model would actually need.
      Reporting only one of them would either strawman the proposal or quietly
      replace it.

      `donor` EXISTS BECAUSE THE FIRST RUN CAME BACK POSITIVE ON ROBIN. Adding
      the target infant's early segment at weight M does two things at once: it
      supplies PATIENT-SPECIFIC information, and it upweights a block of
      EARLY-RECORDING windows. Only the first is personalisation; the second
      would help (or hurt) just as much with anybody's early windows -- early
      recording is quieter, better-attached, less movement-contaminated, and a
      tree that leans on it may simply be better regularised.

      The control therefore holds the row count and the weight fixed and moves
      only the identity: for target infant p it adds the first-K segment of a
      DIFFERENT infant q, truncated to the size of p's own, at the same M.

      Its one unavoidable asymmetry, stated rather than hidden: every non-target
      window is ALREADY in the training pool, so q's rows enter as duplicates
      while p's are new. No control can escape that -- novel non-target data
      does not exist under leave-one-out. `donor` therefore answers the question
      it can answer, which is the one that matters here: is the gain explained
      by the mere presence of N heavily-weighted early-recording rows, whoever
      they belong to? If `segment` and `donor` gain alike, the answer is yes and
      the effect is not adaptation.

  (3) M IS APPLIED AS A SAMPLE WEIGHT, NOT AS DUPLICATED ROWS. Mathematically
      identical for a gradient-boosted tree, and it avoids a real bug: `gbm()`
      uses early stopping with a RANDOM 10 % validation split, so literal
      duplicates would land in both halves and the stopping criterion would be
      scored on rows it had trained on. Weights also let M be swept continuously.
      Class balancing is computed BEFORE M is applied, so M is purely the
      personalisation strength and not a second, hidden class prior.

LEAKAGE DISCIPLINE
    Features here are all CAUSAL (`causal_mean`, `causal_percentile` in
    spo2_baseline_prediction.py look strictly into the past), so a test window
    drawing on the calibration period is the intended mechanism, not leakage.
    The direction that WOULD leak is a training label reaching forward into the
    scored region: a calibration window at time t carries a label covering
    [t+lead, t+lead+H). A buffer of `--buffer-min` minutes is therefore skipped
    after the K_MAX-th event before scoring starts, comfortably longer than
    lead + horizon. The other infants contribute all of their windows, as in
    every leave-one-out table in this study.

POSITIVE CONTROL (why the Robin cohort is run too)
    Given Sections 4.4 and 4.5 of the thesis the CPAP answer is likely another null,
    and a null is only worth writing down if the method can be shown to work
    where a precursor exists. The identical protocol is therefore run on the
    Robin-sequence cohort, where this same feature pipeline predicts obstructive
    apnea at 0.664 rather than the CPAP 0.574. "Adaptation helps where there is
    something to adapt to, and this cohort has nothing" is a result;
    "adaptation did not help" on its own is not.

CODE BOUNDARY
    BA analysis code, in scripts/ with the rest of it. It imports the original
    pipeline only through spo2_baseline_prediction (feature cache, `gbm()`,
    `sample_weights()`) and modifies nothing under src/.

Usage:
    # CPAP (target cohort)
    python scripts/patient_adaptation.py \
        --cache outputs/spo2_baseline/features_v2.npz \
        --out outputs/patient_adaptation/cpap.json

    # Robin (positive control)
    python scripts/patient_adaptation.py \
        --cache outputs/robin_severity/robin_obstructive.npz \
        --out outputs/patient_adaptation/robin_obstructive.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import spo2_baseline_prediction as S  # noqa: E402

STRIDE_S = S.STRIDE_S
K_GRID = [0, 1, 3, 5, 10]
M_GRID = [1, 5, 10, 25, 50, 100]
ARMS = ["segment", "positives", "donor"]
BUFFER_MIN = 5           # skipped between calibration and scoring
MIN_TEST_POS = 5         # a patient needs this much left to be scorable at all
MIN_TEST_NEG = 5


def chronological(groups, times, pid):
    """Indices of this patient's windows in true recording order.

    `times` is built as `si * 1e9 + sample_index` (spo2_baseline_prediction.py),
    so a plain argsort orders correctly ACROSS segments as well as within one.
    """
    sel = np.where(groups == pid)[0]
    return sel[np.argsort(times[sel])]


def event_ends(pos, seg):
    """Index of the LAST window of each apnea episode, in order.

    Consecutive positive windows are one event, and a run is broken at a segment
    boundary -- two recording segments are not contiguous in time, so a positive
    run closing segment 0 and one opening segment 1 are two events, not one.

    Taking the END of the run rather than its start is what makes "the model may
    use the first K apneas" honest: the whole of the K-th event then sits inside
    the calibration period, so the cut never falls in the middle of an event the
    model is afterwards scored on.
    """
    ends, n, i = [], len(pos), 0
    while i < n:
        if pos[i]:
            j = i
            while j + 1 < n and pos[j + 1] and seg[j + 1] == seg[j]:
                j += 1
            ends.append(j)
            i = j + 1
        else:
            i += 1
    return np.asarray(ends, dtype=int)


def build_patient_plan(d, endpoint, horizon, lead, k_max, buffer_win):
    """Per patient: chronological order, event cuts, and the FIXED eval slice.

    The eval slice is defined once, by k_max, and reused for every (K, M, arm)
    cell -- design point (1). A patient that cannot supply k_max events, or has
    too little left to score afterwards, is dropped from ALL cells rather than
    from some, so every row of the grid is computed over identical patients.
    """
    groups, times = d["groups"], d["times"]
    y_det = d["y_%s_%d_L0" % (endpoint, horizon)]      # event present == onset marker
    y_tgt = d["y_%s_%d_L%d" % (endpoint, horizon, lead)]

    plans, dropped = {}, []
    for pid in sorted(set(groups.tolist())):
        o = chronological(groups, times, pid)
        seg = np.floor(times[o] / 1e9).astype(int)
        ends = event_ends(y_det[o] == 1, seg)
        if len(ends) < max(k_max, 1):
            dropped.append((pid, "only %d events (< k_max)" % len(ends)))
            continue
        eval_start = ends[k_max - 1] + 1 + buffer_win
        ev = o[eval_start:]
        ev = ev[y_tgt[ev] >= 0]                        # -1 marks unusable
        n_pos = int((y_tgt[ev] == 1).sum())
        n_neg = int((y_tgt[ev] == 0).sum())
        if n_pos < MIN_TEST_POS or n_neg < MIN_TEST_NEG:
            dropped.append((pid, "%d pos / %d neg left after the cut" % (n_pos, n_neg)))
            continue
        plans[pid] = dict(order=o, ends=ends, eval_idx=ev, n_events=len(ends))
    return plans, dropped, y_tgt


def donor_of(pid, pids):
    """The infant whose early segment stands in for `pid`'s in the `donor` arm.

    The next scored infant, cyclically. Fixed rather than random so the control
    is reproducible, and a rotation rather than one shared donor so no single
    recording's quirks drive every fold.
    """
    return pids[(pids.index(pid) + 1) % len(pids)]


def calib_rows(plan, k, arm, y_tgt):
    """Rows of the target infant that enter the fit for this (K, arm) cell."""
    if k == 0:
        return np.empty(0, dtype=int)
    cut = plan["ends"][k - 1]
    rows = plan["order"][: cut + 1]
    rows = rows[y_tgt[rows] >= 0]
    if arm == "positives":
        rows = rows[y_tgt[rows] == 1]
    return rows


def run_cohort(d, args):
    # `--blocks` exists to answer a question the first grid could not: is the
    # null a property of ADAPTATION, or of the feature set it was run on? A
    # model that has nothing to adapt on cannot be adapted, so the sweep is
    # repeated per feature family -- SpO2 (DR+IH), physiology (PH), event
    # history (HI) -- as well as on all of them together.
    X = np.hstack([d[b] for b in args.blocks])
    groups = d["groups"]
    print("feature blocks %s -> %d columns\n" % ("+".join(args.blocks), X.shape[1]))
    buffer_win = int(np.ceil(args.buffer_min * 60 / STRIDE_S))
    plans, dropped, y = build_patient_plan(
        d, args.endpoint, args.horizon, args.lead, args.k_max, buffer_win)

    pids = sorted(plans)
    print("patients scored: %d   dropped: %d" % (len(pids), len(dropped)))
    for pid, why in dropped:
        print("   drop %s -- %s" % (pid, why))
    print("buffer %g min = %d windows; eval slice fixed by k_max=%d\n"
          % (args.buffer_min, buffer_win, args.k_max))
    print("%-5s %8s %8s %10s %8s" % ("pid", "events", "eval_n", "eval_prev", "hours"))
    for pid in pids:
        ev = plans[pid]["eval_idx"]
        print("%-5s %8d %8d %10.3f %8.1f"
              % (pid, plans[pid]["n_events"], len(ev), y[ev].mean(),
                 len(plans[pid]["order"]) * STRIDE_S / 3600))
    print()

    # K = 0 is a single cell: with no target rows, M has nothing to multiply.
    cells = [("control", 0, 1)]
    for arm in args.arms:
        for k in [k for k in args.k_grid if k > 0]:
            for m in args.m_grid:
                cells.append((arm, k, m))

    auc, pooled, ncal = {}, {}, {}
    t0 = time.time()
    for ci, (arm, k, m) in enumerate(cells):
        per, sc, ys, sizes = {}, [], [], []
        for pid in pids:
            plan = plans[pid]
            ev = plan["eval_idx"]
            src = np.where(groups != pid)[0]
            src = src[y[src] >= 0]
            if arm == "donor":
                # Same row count, same weight, WRONG INFANT -- design (2).
                # Truncated to the target's own calibration size so the two arms
                # differ in identity alone and not in how much was added.
                q = donor_of(pid, pids)
                n_own = len(calib_rows(plan, k, "segment", y))
                dr = calib_rows(plans[q], k, "segment", y)
                # Tile before truncating: a donor with a SHORTER calibration
                # period would otherwise contribute fewer rows than the target
                # did, and the control would be testing row count rather than
                # identity -- the exact confound it exists to remove.
                if len(dr) and n_own:
                    dr = np.tile(dr, int(np.ceil(n_own / len(dr))))
                cal = dr[:n_own]
            else:
                cal = calib_rows(plan, k, arm, y)
            rows = np.concatenate([src, cal])
            yr = y[rows]
            # class balance FIRST, personalisation strength on top -- design (3)
            w = S.sample_weights(yr)
            if len(cal):
                w[len(src):] *= m
            c = S.gbm()
            c.fit(X[rows], yr, sample_weight=w)
            p = c.predict_proba(X[ev])[:, 1]
            per[pid] = float(roc_auc_score(y[ev], p))
            sc.append(p)
            ys.append(y[ev])
            sizes.append(len(cal))
        auc[(arm, k, m)] = per
        pooled[(arm, k, m)] = float(roc_auc_score(np.concatenate(ys),
                                                  np.concatenate(sc)))
        ncal[(arm, k, m)] = float(np.mean(sizes))
        print("[%2d/%2d] %-10s K=%-3d M=%-4d  cal_rows %5.1f  mean %.3f  "
              "pooled %.3f  (%.0fs)"
              % (ci + 1, len(cells), arm, k, m, ncal[(arm, k, m)],
                 float(np.mean(list(per.values()))), pooled[(arm, k, m)],
                 time.time() - t0), flush=True)
    return pids, auc, pooled, ncal, plans


def report(pids, auc, pooled, ncal, args):
    base = auc[("control", 0, 1)]
    b = np.array([base[p] for p in pids])
    print("\n" + "=" * 78)
    print("K = 0 control (no target data):  mean %.3f   pooled %.3f   n = %d infants"
          % (b.mean(), pooled[("control", 0, 1)], len(pids)))
    print("=" * 78)
    for arm in args.arms:
        print("\narm = %s" % arm)
        print("cell = mean AuROC / delta vs control / Wilcoxon p / infants improved")
        print("\n%-4s %-6s | " % ("K", "rows")
              + " | ".join("%-22s" % ("M = %d" % m) for m in args.m_grid))
        print("-" * (14 + 25 * len(args.m_grid)))
        for k in [k for k in args.k_grid if k > 0]:
            cells = []
            for m in args.m_grid:
                a = np.array([auc[(arm, k, m)][p] for p in pids])
                dif = a - b
                if np.allclose(dif, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(a, b).pvalue)
                cells.append("%.3f %+.3f p%.3f %2d/%d"
                             % (a.mean(), dif.mean(), p, int((dif > 0).sum()), len(pids)))
            print("%-4d %-6.0f | " % (k, ncal[(arm, k, args.m_grid[0])])
                  + " | ".join("%-22s" % c for c in cells))
    print("\n`rows` is the mean number of target-infant windows added to the fit "
          "(before weighting).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="feature npz from spo2_baseline_prediction")
    ap.add_argument("--out", default=None,
                    help="json to write per-patient AuROCs to (for the figure)")
    ap.add_argument("--endpoint", default="apnea")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lead", type=int, default=60)
    ap.add_argument("--k-max", type=int, default=10,
                    help="fixes the evaluation slice for EVERY cell -- design (1)")
    ap.add_argument("--k-grid", type=int, nargs="*", default=K_GRID)
    ap.add_argument("--m-grid", type=int, nargs="*", default=M_GRID)
    ap.add_argument("--arms", nargs="*", default=ARMS, choices=ARMS)
    ap.add_argument("--buffer-min", type=float, default=BUFFER_MIN)
    ap.add_argument("--blocks", nargs="*", default=["DR", "IH", "HI", "PH"],
                    choices=["DR", "IH", "HI", "PH"],
                    help="feature families to use: DR/IH = SpO2, PH = physiology "
                         "(HR, PCO2, Thorax, Abdomen, pressure, and PR if the "
                         "cache was built with it), HI = event history")
    args = ap.parse_args()

    d = dict(np.load(args.cache, allow_pickle=True))
    print("cache %s | %d windows | endpoint %s H=%d L=%d\n"
          % (args.cache, len(d["groups"]), args.endpoint, args.horizon, args.lead))
    pids, auc, pooled, ncal, plans = run_cohort(d, args)
    report(pids, auc, pooled, ncal, args)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        payload = dict(
            cache=args.cache, endpoint=args.endpoint, horizon=args.horizon,
            lead=args.lead, k_max=args.k_max, buffer_min=args.buffer_min,
            arms=args.arms, k_grid=args.k_grid, m_grid=args.m_grid,
            blocks=args.blocks, pids=pids,
            auc={"%s|%d|%d" % key: val for key, val in auc.items()},
            pooled={"%s|%d|%d" % key: val for key, val in pooled.items()},
            n_events={p: int(plans[p]["n_events"]) for p in pids},
            n_eval={p: int(len(plans[p]["eval_idx"])) for p in pids})
        out = os.path.splitext(args.out)[0] + ".json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=1)
        print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
