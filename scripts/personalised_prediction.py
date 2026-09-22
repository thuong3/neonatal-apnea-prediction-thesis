"""PER-PATIENT CALIBRATION: "let it see the first few events, then predict."

THE PROPOSAL (supervision, Aug 2026)
    "It doesn't have to predict it completely blind. Set a number -- three -- the
    first 3 apneas from this patient it's allowed to use to adapt its internal
    thresholds. And then it should predict it."

WHY THAT SPLITS INTO TWO EXPERIMENTS
    AuROC is THRESHOLD-INVARIANT. It scores the ranking of windows, not where the
    alarm is cut. So a method that only adapts a decision threshold cannot move
    AuROC at all -- not weakly, but by construction. Run naively, the proposal
    would report "no change" and look refuted when it was never measured.

    The proposal therefore has to be separated into the two things it contains:

    (a) FEATURE calibration -- re-express each feature relative to this infant's
        own normal. This DOES change the ranking, so AuROC can see it. It is also
        the generalisation of `M - B_T` (#23), the one construction in this study
        that produced a positive result, from a single channel to every feature.
        It needs NO LABELS, so it can use the whole calibration period rather
        than only the 3 events -- strictly more information than was asked for.

    (b) THRESHOLD calibration -- use the first K events to place the operating
        point. Invisible to AuROC, so it is reported as SENSITIVITY and ALARMS
        PER HOUR, which is what a bedside device is actually judged on and a
        number this study does not otherwise report.

WHY K IS SWEPT AND THE BUDGET IS MATCHED (#c, #d -- added Aug 2026)
    Two things make the bare (a)/(b) comparison unfair to the proposal, in
    opposite directions, and both had to be fixed before the null could be
    believed:

    (c) K = 3 IS AN ARBITRARY NUMBER. Reporting one K cannot distinguish "per-
        patient operating points do not help" from "three events are too few to
        estimate one". K is therefore swept (1, 3, 5, 10) so the answer is a
        curve rather than a point. This cohort has ~8-20 scored events per
        infant, so K = 10 is close to the ceiling of what could ever be offered.

    (d) THE OPERATING POINTS WERE NOT COMPARABLE. A personalised threshold that
        fires 8.6 times an hour cannot be compared with a global one that fires
        5.3 times an hour -- more alarms buy more sensitivity for free, so the
        comparison was measuring the alarm budget, not the rule. All rules are
        therefore also compared at a MATCHED alarm budget (2, 6, 12 alarms/h).

        That comparison adds the rule the proposal is really competing against:
        a per-patient threshold set from the infant's own score distribution to
        hit a target alarm rate, which needs NO LABELS AT ALL. If the 3 events
        cannot beat that, they are not buying information -- they are buying an
        alarm rate, and the label-free rule buys it more cheaply.

LEAKAGE DISCIPLINE
    Evaluation starts strictly after BOTH the normalisation period and the K-th
    calibration event. No window used for either purpose is ever scored. Every
    threshold is estimated on calibration-period windows only -- the per-patient
    rules from that patient's own calibration period, the global rule from the
    pooled calibration periods of the OTHER patients (leave-one-out, so the
    held-out infant never contributes to its own threshold).

CODE BOUNDARY
    This file is BA analysis code and lives in scripts/ with the rest of it. It
    imports the ORIGINAL pipeline only through its public API (src.neonatal_utils
    via the feature cache) and does not modify src/: neither Julius Vetter's
    original modules (neonatal_utils, train_neonatal, modules, misc_utils,
    plotting_utils) nor the domain-adaptation entry point
    src/train_domain_adaptation.py, which is a separate experiment (see
    Section 4.5 of the thesis) and shares no code with this one.

Usage:
    python scripts/personalised_prediction.py --cache outputs/spo2_baseline/features_v2.npz
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import spo2_baseline_prediction as S  # noqa: E402

STRIDE_S = S.STRIDE_S
NORM_MIN = 30                      # unsupervised feature-calibration period
N_NORM = (NORM_MIN * 60) // STRIDE_S
K_EVENTS = 3                       # the supervisor's number
K_SWEEP = [1, 3, 5, 10]            # ... and the sweep around it
BUDGETS = [2.0, 6.0, 12.0]         # alarms/h, for the matched-budget comparison
DUTY_MAX = 0.25                    # reject a rule that alarms >25 % of the time
HORIZON_S, LEAD_S = 60, 60
TARGET_SENS = 0.80                 # global operating point, for comparison


def per_patient_z(X, idx_order, n_norm):
    """Z-score every feature using only this patient's first `n_norm` windows.

    Unsupervised: it never looks at a label, so the calibration period does not
    have to contain an event. NaN-safe; a feature with no variation in the
    calibration period is left on its original scale rather than blown up.
    """
    calib = X[idx_order[:n_norm]]
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(calib, axis=0)
        sd = np.nanstd(calib, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return (X - mu) / sd


def alarm_episodes(fired):
    """Number of distinct alarms: consecutive firing windows are one alarm."""
    f = np.asarray(fired, dtype=np.int8)
    if not f.any():
        return 0
    return int((np.diff(np.concatenate([[0], f])) == 1).sum())


def alarm_rate(scores, thr, hours):
    return alarm_episodes(scores >= thr) / hours


def duty_cycle(scores, thr):
    """Fraction of time the alarm is on. The guard against a degenerate rule."""
    return float((scores >= thr).mean())


def thr_for_rate(per_patient, target_rate, duty_max=DUTY_MAX, n_levels=200):
    """Threshold whose MEAN per-patient alarm rate comes closest to `target_rate`.

    `per_patient` is a list of (scores, hours); the mean is taken over its
    entries, so passing one entry gives a per-patient threshold and passing the
    other infants gives a leave-one-out global one. Averaging per-patient rates
    is not the same as counting episodes in the concatenated array -- that would
    divide one patient's worth of hours into every patient's episodes.

    Searched over a quantile grid rather than solved analytically because the
    alarm rate is NOT monotone in the threshold: lowering it merges neighbouring
    firing windows into a single longer episode, so the episode count rises,
    peaks, and then FALLS again.

    That non-monotonicity has a degenerate solution which has to be excluded
    explicitly: a threshold below every score fires on all of it, which is one
    single endless episode, i.e. an apparently excellent ~0.05 alarms/h while
    the alarm is on permanently. Candidates are therefore restricted to those
    whose duty cycle stays <= `duty_max`. Without this guard the rule reports a
    low alarm rate and a sensitivity near 1.0 and looks like the best method in
    the table while being clinically meaningless.

    Label-free by construction -- it looks only at score distributions.
    """
    cand = np.unique(np.quantile(np.concatenate([s for s, _ in per_patient]),
                                 np.linspace(0.0, 1.0, n_levels)))
    best, best_err = None, np.inf
    for t in cand:
        if np.mean([duty_cycle(s, t) for s, _ in per_patient]) > duty_max:
            continue
        r = np.mean([alarm_rate(s, t, h) for s, h in per_patient])
        err = abs(r - target_rate)
        if err < best_err:
            best_err, best = err, float(t)
    # Every candidate alarms too much: fall back to the most conservative one.
    return float(cand[-1]) if best is None else best


def score_patients(d, endpoint, personalise):
    """Leave-one-patient-out fit; return each patient's full chronological scores.

    The model does not depend on K -- K only moves the evaluation cutoff -- so
    the GBM is fitted ONCE per patient here and every K reuses these scores.
    """
    X_all = np.hstack([d["DR"], d["IH"], d["HI"], d["PH"]])
    y_all = d["y_%s_%d_L%d" % (endpoint, HORIZON_S, LEAD_S)]
    groups, times = d["groups"], d["times"]

    keep = y_all >= 0
    X_all, y_all, groups, times = X_all[keep], y_all[keep], groups[keep], times[keep]

    order = {}
    X = X_all.copy()
    for pid in S.IDS:
        sel = np.where(groups == pid)[0]
        if len(sel) == 0:
            continue
        o = sel[np.argsort(times[sel])]
        order[pid] = o
        if personalise and len(o) > N_NORM:
            X[o] = per_patient_z(X_all[o], np.arange(len(o)), N_NORM)

    out = {}
    for pid in S.IDS:
        if pid not in order:
            continue
        o = order[pid]
        if len(o) <= N_NORM + 10:
            continue
        tr = groups != pid
        c = S.gbm()
        c.fit(X[tr], y_all[tr], sample_weight=S.sample_weights(y_all[tr]))
        out[pid] = dict(scores=c.predict_proba(X[o])[:, 1], y=y_all[o])
    return out


def cutoff_for_k(y_p, k):
    """First index that may be evaluated: after the norm period AND the K-th event.

    Returns (cutoff, calib_event_indices) or None if this patient has too few
    events to both calibrate on K of them and still leave a testable remainder.
    """
    pos = np.where(y_p[N_NORM:] == 1)[0] + N_NORM
    if len(pos) < k + 5:
        return None
    calib_pos = pos[:k]
    return int(calib_pos[-1]) + 1, calib_pos


def rows_for_k(ps, k):
    """Per-patient evaluation rows at calibration size K. Model already fitted."""
    rows = []
    for pid, rec in ps.items():
        y_p, s_p = rec["y"], rec["scores"]
        cut = cutoff_for_k(y_p, k)
        if cut is None:
            continue
        cutoff, calib_pos = cut
        y_ev, s_ev = y_p[cutoff:], s_p[cutoff:]
        if len(np.unique(y_ev)) < 2 or len(y_ev) < 50:
            continue
        hours = len(y_ev) * STRIDE_S / 3600.0

        # The proposal, read literally: the alarm must have fired on all K
        # events it was shown, so the threshold is the lowest of their scores.
        thr_p = float(s_p[calib_pos].min())
        rows.append(dict(
            patient=pid, n_eval=len(y_ev), prevalence=float(y_ev.mean()),
            auc=roc_auc_score(y_ev, s_ev),
            thr_pers=thr_p,
            sens_pers=float((s_ev >= thr_p)[y_ev == 1].mean()),
            alarms_h_pers=alarm_rate(s_ev, thr_p, hours),
            duty_pers=duty_cycle(s_ev, thr_p),
            scores=s_ev, y=y_ev, hours=hours,
            calib_scores=s_p[:cutoff], calib_hours=cutoff * STRIDE_S / 3600.0,
        ))
    return rows


def global_threshold(rows, target_sens):
    """One threshold shared by every patient, at the target pooled sensitivity."""
    pos = np.concatenate([r["scores"][r["y"] == 1] for r in rows])
    return float(np.quantile(pos, 1.0 - target_sens))


def matched_budget(rows, budget):
    """Compare three thresholding rules at the SAME alarm budget.

    All three see calibration-period windows only:

      global    one threshold for every infant, tuned on the pooled calibration
                periods of the OTHER infants (leave-one-out) to hit `budget`
      rate      per-patient, from this infant's own calibration-period score
                distribution -- personalised but LABEL-FREE
      events    per-patient, from the K labelled events (the proposal). Its
                alarm rate is whatever it is; it cannot be tuned to a budget,
                which is itself part of the finding.
    """
    out = {k: {"sens": [], "rate": [], "duty": []}
           for k in ("global", "rate", "events")}
    for r in rows:
        others = [q for q in rows if q["patient"] != r["patient"]]
        thr_g = thr_for_rate([(q["calib_scores"], q["calib_hours"])
                              for q in others], budget)
        thr_r = thr_for_rate([(r["calib_scores"], r["calib_hours"])], budget)

        for key, thr in (("global", thr_g), ("rate", thr_r),
                         ("events", r["thr_pers"])):
            out[key]["sens"].append(
                float((r["scores"] >= thr)[r["y"] == 1].mean()))
            out[key]["rate"].append(alarm_rate(r["scores"], thr, r["hours"]))
            out[key]["duty"].append(duty_cycle(r["scores"], thr))
    return out


def matched_duty(rows):
    """Same alarm BURDEN as the K-event rule, with and without the labels.

    The K-event threshold cannot be tuned to a budget, so (d) can only put it
    beside rules that were tuned to a different one. This isolates the value of
    the K labels exactly: for each infant, measure the duty cycle its K-event
    threshold implies on the CALIBRATION period, then give the other two rules
    that same duty cycle and ask who is more sensitive on the evaluation period.

    If the labelled events buy real information, the `events` rule wins here. If
    they only buy a lower threshold, all three land together -- which is what a
    ranking at chance predicts, since an ROC on the diagonal trades sensitivity
    for alarm burden one-for-one no matter how the threshold is chosen.
    """
    out = {k: {"sens": [], "duty": []} for k in ("global", "rate", "events")}
    targets = [duty_cycle(r["calib_scores"], r["thr_pers"]) for r in rows]
    mean_target = float(np.mean(targets))

    for r, target in zip(rows, targets):
        others = [q for q in rows if q["patient"] != r["patient"]]
        pooled = np.concatenate([q["calib_scores"] for q in others])
        # Quantiles of calibration scores hit the requested duty by construction.
        thr_g = float(np.quantile(pooled, 1.0 - mean_target))
        thr_r = float(np.quantile(r["calib_scores"], 1.0 - target))

        for key, thr in (("global", thr_g), ("rate", thr_r),
                         ("events", r["thr_pers"])):
            out[key]["sens"].append(
                float((r["scores"] >= thr)[r["y"] == 1].mean()))
            out[key]["duty"].append(duty_cycle(r["scores"], thr))
    return out, mean_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="outputs/spo2_baseline/features_v2.npz")
    ap.add_argument("--endpoints", default="declabs,apnea")
    args = ap.parse_args()

    d = dict(np.load(args.cache, allow_pickle=True))
    print("cache: %s | %d windows" % (args.cache, len(d["groups"])))
    print("calibration: %d min unsupervised feature normalisation (%d windows) "
          "+ first K events for the threshold" % (NORM_MIN, N_NORM))
    print("evaluation: strictly after both. horizon %ds, lead %ds\n"
          % (HORIZON_S, LEAD_S))

    for endpoint in args.endpoints.split(","):
        print("=" * 78)
        print("ENDPOINT: %s" % endpoint)
        print("=" * 78)

        scored = {}
        for personalise in (False, True):
            ps = score_patients(d, endpoint, personalise)
            if not ps:
                print("  no evaluable patients")
                break
            scored["personalised" if personalise else "global"] = ps
        if len(scored) < 2:
            continue

        res = {k: rows_for_k(v, K_EVENTS) for k, v in scored.items()}

        # ---- (a) does per-patient feature calibration change the RANKING? ----
        print("\n(a) FEATURE calibration -- measured by AuROC (K=%d)" % K_EVENTS)
        print("    %-8s %-14s %-14s %s" % ("patient", "global feats",
                                           "personalised", "delta"))
        gr = {r["patient"]: r for r in res["global"]}
        pr = {r["patient"]: r for r in res["personalised"]}
        common = [p for p in S.IDS if p in gr and p in pr]
        dif = []
        for p in common:
            delta = pr[p]["auc"] - gr[p]["auc"]
            dif.append(delta)
            print("    %-8s %-14.3f %-14.3f %+.3f"
                  % (p, gr[p]["auc"], pr[p]["auc"], delta))
        g_mean = float(np.mean([gr[p]["auc"] for p in common]))
        p_mean = float(np.mean([pr[p]["auc"] for p in common]))
        print("    %-8s %-14.3f %-14.3f %+.3f   (n=%d patients)"
              % ("MEAN", g_mean, p_mean, p_mean - g_mean, len(common)))
        if len(common) >= 5:
            from scipy.stats import wilcoxon
            w = wilcoxon([pr[p]["auc"] for p in common],
                         [gr[p]["auc"] for p in common])
            print("    Wilcoxon signed-rank: W=%.0f, p=%.3f, better in %d/%d"
                  % (w.statistic, w.pvalue, sum(x > 0 for x in dif), len(dif)))

        # ---- (b) does the personalised THRESHOLD beat one global cut? --------
        print("\n(b) THRESHOLD calibration -- invisible to AuROC, so: "
              "sensitivity and alarm rate")
        rows = res["personalised"]
        gthr = global_threshold(rows, TARGET_SENS)
        print("    global threshold set at %.1f %% pooled sensitivity = %.4f"
              % (100 * TARGET_SENS, gthr))
        print("    NOTE: the two rules fire at DIFFERENT rates, so this "
              "comparison is confounded")
        print("          by the alarm budget. See (d) for the matched-budget "
              "version.")
        print("    %-8s %-22s %s" % ("", "personalised thr", "one global thr"))
        print("    %-8s %-10s %-11s %-10s %s"
              % ("patient", "sens", "alarms/h", "sens", "alarms/h"))
        agg = [[], [], [], []]
        for r in rows:
            fired_g = r["scores"] >= gthr
            sg = float(fired_g[r["y"] == 1].mean())
            ag = alarm_episodes(fired_g) / r["hours"]
            print("    %-8s %-10.2f %-11.1f %-10.2f %.1f"
                  % (r["patient"], r["sens_pers"], r["alarms_h_pers"], sg, ag))
            agg[0].append(r["sens_pers"]); agg[1].append(r["alarms_h_pers"])
            agg[2].append(sg); agg[3].append(ag)
        print("    %-8s %-10.2f %-11.1f %-10.2f %.1f"
              % ("MEAN", np.mean(agg[0]), np.mean(agg[1]),
                 np.mean(agg[2]), np.mean(agg[3])))

        # ---- (c) how many of the patient's own events are needed? -----------
        print("\n(c) K-SWEEP -- is 3 too few, or is the whole idea flat?")
        print("    threshold = must have fired on all K calibration events "
              "(the proposal, literally)")
        print("    read `sens` against `duty`: the threshold is a MINIMUM over "
              "K scores, so it")
        print("    can only fall as K grows. Sensitivity bought by alarming "
              "more of the time is")
        print("    not sensitivity bought by information -- see (e).")
        print("    %-4s %-7s %-9s %-13s %-13s %-8s %-10s %s"
              % ("K", "n_pat", "eval_h", "AuROC glob", "AuROC pers",
                 "sens", "alarms/h", "duty"))
        for k in K_SWEEP:
            rg = rows_for_k(scored["global"], k)
            rp = rows_for_k(scored["personalised"], k)
            if not rp:
                print("    %-4d  (no evaluable patients)" % k)
                continue
            print("    %-4d %-7d %-9.1f %-13.3f %-13.3f %-8.2f %-10.1f %.2f"
                  % (k, len(rp), np.mean([r["hours"] for r in rp]),
                     np.mean([r["auc"] for r in rg]) if rg else float("nan"),
                     np.mean([r["auc"] for r in rp]),
                     np.mean([r["sens_pers"] for r in rp]),
                     np.mean([r["alarms_h_pers"] for r in rp]),
                     np.mean([r["duty_pers"] for r in rp])))

        # ---- (d) the same alarm budget for every rule ------------------------
        print("\n(d) MATCHED ALARM BUDGET -- the fair comparison (K=%d, "
              "personalised feats)" % K_EVENTS)
        print("    every threshold estimated on CALIBRATION windows only")
        print("    'rate' is per-patient but LABEL-FREE: it never sees an event")
        print("    'duty' = fraction of time alarming; a rule is only "
              "meaningful if it stays low")
        print("    %-9s %-22s %-8s %-10s %s"
              % ("budget", "rule", "sens", "alarms/h", "duty"))
        labels = {"global": "one global thr",
                  "rate": "per-pt, label-free",
                  "events": "per-pt, from K=%d events" % K_EVENTS}
        for b in BUDGETS:
            m = matched_budget(rows, b)
            for key in ("global", "rate", "events"):
                print("    %-9s %-22s %-8.2f %-10.1f %.2f"
                      % ("%.1f" % b, labels[key],
                         np.mean(m[key]["sens"]), np.mean(m[key]["rate"]),
                         np.mean(m[key]["duty"])))
        print("    (the 'from K events' rule has no budget to hit -- its rate "
              "is whatever it is,")
        print("     which is itself part of the finding, so it repeats "
              "unchanged on every row)")

        # ---- (e) same alarm burden, with and without the K labels ------------
        md, mean_target = matched_duty(rows)
        print("\n(e) MATCHED ALARM BURDEN -- what do the K=%d labels actually buy?"
              % K_EVENTS)
        print("    the K-event threshold implies a duty cycle of %.2f on the "
              "calibration period;" % mean_target)
        print("    the other two rules are given that same duty cycle, so only "
              "the labels differ")
        print("    %-24s %-8s %s" % ("rule", "sens", "duty"))
        for key, lab in (("events", "per-pt, from K=%d events" % K_EVENTS),
                         ("rate", "per-pt, label-free"),
                         ("global", "one global thr")):
            print("    %-24s %-8.2f %.2f"
                  % (lab, np.mean(md[key]["sens"]), np.mean(md[key]["duty"])))
        if len(rows) >= 5:
            from scipy.stats import wilcoxon
            w = wilcoxon(md["events"]["sens"], md["rate"]["sens"])
            print("    events vs label-free: W=%.0f, p=%.3f, better in %d/%d"
                  % (w.statistic, w.pvalue,
                     sum(a > b for a, b in zip(md["events"]["sens"],
                                               md["rate"]["sens"])), len(rows)))
        print()


if __name__ == "__main__":
    main()
