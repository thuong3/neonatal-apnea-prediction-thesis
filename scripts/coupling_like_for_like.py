"""APNEA-TO-DESATURATION COUPLING UNDER ONE CRITERION APPLIED TO BOTH COHORTS.

Answering this question from the desaturation annotations would confound it:
the two cohorts do not mean the same thing by "desaturation".
Sievers (2008) marked a fall to 80 % or below,
whereas the Robin recordings follow the 2020 AASM criteria, under which a fall
of at least 3 % within five seconds is marked. `scripts/desat_criterion_check.py`
measures the gap on the recordings -- a scored desaturation reaches a median
nadir of 73.8 % in the Brainimmaturity cohort against 90.2 % in Robin, and only
10.0 % of the Robin events would satisfy Sievers' rule.

This script removes the confound by ignoring the desaturation annotations
entirely. It uses only the apnea marks and the saturation trace, and applies one
rule to both cohorts. A scored central apnea counts as coupled if, within the
linking window, the saturation

  * falls at least DROP_PP points below the pre-apnea baseline   ("3-point"),
  * or reaches ABS_PCT or below                                  ("<= 80 %").

The baseline is the median over the 30 s preceding the onset. For each apnea the
LATENCY to the first qualifying sample is stored, so the curve over linking
windows is the empirical CDF of that latency and costs nothing extra.

CHANCE CONTROL
    A longer window couples more events for a trivial reason, since saturation
    in these infants wobbles on its own. The dashed bands are surrogates: the
    identical test applied at random onsets within the same analysis block,
    which preserves the saturation trace and destroys only the timing relation.
    The gap between the solid curve and the band is the coupling that is
    actually about the apnea.

WHAT IT SHOWS
    Panel (a): the coupling is present in both cohorts and roughly three times
    weaker in the Brainimmaturity recordings. Panel (b): at the threshold that
    carries the clinical risk the ordering reverses, because these infants begin
    from a lower baseline.

Usage:
    python scripts/coupling_like_for_like.py
    python scripts/coupling_like_for_like.py --out_dir thesis/figures
"""
import argparse
import os
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
matplotlib.rc_file(os.path.join(ROOT, "matplotlibrc"))
from src.neonatal_utils import read_data  # noqa: E402

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"
FS = 200
BASELINE_S = 30
DROP_PP = 3.0
ABS_PCT = 80.0
MAX_S = 120.0                # longest linking window swept
REPORT_S = 20.0              # window the thesis quotes; coverage is required here
N_SURROGATE = 20             # random onsets per real apnea

WINDOWS = np.array([0., 1., 2., 3., 5., 7.5, 10., 15., 20., 30., 45., 60.,
                    90., 120.])
MARK_S = (5.0, 20.0)

CPAP, ROBIN = "Brainimmaturity", "Robin sequence"
COHORTS = {
    CPAP: dict(
        path="data/"
             "dataset_brainimmaturity",
        sub="annotations", ids=["%03d" % i for i in range(1, 16)],
        record_duration=None),
    ROBIN: dict(
        path="data/robin_sequence/"
             "data/neonatal_robin_polysomnography",
        sub="annotations_original", ids=["%03d" % i for i in range(1, 20)],
        record_duration=10),
}

COLOR = {CPAP: "#2a78d6", ROBIN: "#eb6834"}
C_GUIDE = "#c9c8c3"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "thesis/figures"))
    p.add_argument("--out", default="outputs/apnea_desat_coupling_like.txt")
    p.add_argument("--report-at", type=float, default=REPORT_S,
                   help="linking window for the printed table")
    return p.parse_args()


def annotations(path, sub, pid):
    f = os.path.join(path, sub, "annotations%s.txt" % pid)
    if not os.path.exists(f):
        return None
    a = pd.read_csv(f, sep="\t")
    a = a[a["type"] != "type"].copy()
    t0 = datetime.strptime(a["timestamp"].iloc[0], TS_FMT)
    a["t"] = [(datetime.strptime(s, TS_FMT) - t0).total_seconds()
              for s in a["timestamp"]]
    return a


def latencies(x, onset_rel):
    """(latency to a DROP_PP fall, latency to ABS_PCT), inf if neither occurs.

    An apnea is measurable only when the saturation is actually recorded around
    it, so at least half of the baseline window and at least half of the first
    REPORT_S seconds after the onset must carry a physiological value. Without
    that second condition an apnea whose oximeter dropped out would be counted
    as uncoupled rather than excluded, which biases every figure downward in
    the cohort with more dropout.
    """
    i0 = int(round(onset_rel * FS))
    i1 = int(round((onset_rel + MAX_S) * FS))
    b0 = i0 - BASELINE_S * FS
    if b0 < 0 or i1 > len(x) or i1 <= i0:
        return None
    base = x[b0:i0]
    base = base[(base > 40) & (base <= 100)]
    if len(base) < 0.5 * BASELINE_S * FS:
        return None
    b = float(np.median(base))
    ev = x[i0:i1].copy()
    ok = (ev > 40) & (ev <= 100)
    if ok[:int(REPORT_S * FS)].mean() < 0.5:
        return None
    ev[~ok] = np.nan
    t = np.arange(len(ev)) / FS
    hit_drop = np.where((b - ev) >= DROP_PP)[0]
    hit_abs = np.where(ev <= ABS_PCT)[0]
    return (t[hit_drop[0]] if len(hit_drop) else np.inf,
            t[hit_abs[0]] if len(hit_abs) else np.inf)


def curve(lat, windows):
    lat = np.asarray(lat, dtype=float)
    return np.array([np.mean(lat <= w) for w in windows])


def collect(rng):
    """{cohort: {'drop': [...], 'abs': [...], 'sur_drop': [...], ...}}"""
    out = {}
    for name, cfg in COHORTS.items():
        d = {k: [] for k in ("drop", "abs", "sur_drop", "sur_abs")}
        for pid in cfg["ids"]:
            a = annotations(cfg["path"], cfg["sub"], pid)
            if a is None:
                continue
            try:
                segments = read_data(pid, cfg["path"],
                                     annotations_dir=cfg["sub"],
                                     record_duration=cfg["record_duration"])
            except Exception as exc:                      # noqa: BLE001
                print(f"  {pid}: {exc}")
                continue
            starts = a.loc[a["type"] == "ANALYSIS-START", "t"].values
            ap = a.loc[a["type"] == "APNEA-CENTRAL", "t"].values
            for seg, s0 in zip(segments, starts):
                if "SpO2" not in seg:
                    continue
                x = np.asarray(seg["SpO2"], dtype=float)
                span = len(x) / FS
                in_seg = ap[(ap >= s0) & (ap < s0 + span)] - s0
                for rel in in_seg:
                    m = latencies(x, rel)
                    if m is not None:
                        d["drop"].append(m[0])
                        d["abs"].append(m[1])
                lo, hi = BASELINE_S, span - MAX_S
                if len(in_seg) == 0 or hi <= lo:
                    continue
                for _ in range(N_SURROGATE * len(in_seg)):
                    m = latencies(x, float(rng.uniform(lo, hi)))
                    if m is not None:
                        d["sur_drop"].append(m[0])
                        d["sur_abs"].append(m[1])
        out[name] = d
    return out


def make_figure(data, out_dir):
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, wspace=0.26, top=0.80, bottom=0.16,
                          left=0.075, right=0.985)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    panels = [("drop", "sur_drop",
               "(a) Falls %g points below baseline" % DROP_PP,
               "Central apneas with a %g-point fall [%%]" % DROP_PP, 100),
              ("abs", "sur_abs",
               "(b) Reaches %g %% or below" % ABS_PCT,
               "Central apneas reaching %g %% [%%]" % ABS_PCT, 40)]

    for ax, (key, skey, title, ylab, ymax) in zip(axes, panels):
        for w in MARK_S:
            ax.axvline(w, color=C_GUIDE, linewidth=1.0, zorder=0)
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks([0, 5, 10, 20, 60, 120])
        ax.set_xticklabels(["0", "5", "10", "20", "60", "120"])
        ax.set_xlim(0, WINDOWS.max())
        ax.set_xlabel("Linking window after apnea onset [s]")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, ymax)
        for name in (CPAP, ROBIN):
            c = COLOR[name]
            ax.plot(WINDOWS, 100 * curve(data[name][skey], WINDOWS), color=c,
                    linestyle="--", linewidth=1.2,
                    label="%s: chance" % name)
            ax.plot(WINDOWS, 100 * curve(data[name][key], WINDOWS), color=c,
                    marker="o", label="%s: measured (n=%d)"
                    % (name, len(data[name][key])))
        if key == "drop":
            h, l = ax.get_legend_handles_labels()
            ax.legend([h[1], h[0], h[3], h[2]], [l[1], l[0], l[3], l[2]],
                      loc="upper left", handlelength=1.6)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "desat_coupling_signal.%s" % ext),
                    bbox_inches="tight")
    plt.close(fig)
    print("figure -> %s/desat_coupling_signal.pdf" % out_dir)


def main():
    args = parse_args()
    data = collect(np.random.default_rng(0))
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out(f"Coupling within {args.report_at:.0f} s of a scored central apnea "
        f"onset, measured on the")
    out(f"saturation trace, ignoring the DESAT annotations. Baseline = median "
        f"over the")
    out(f"preceding {BASELINE_S} s. Chance = the same test at random onsets in "
        f"the same block.")
    out()
    w = np.array([args.report_at])
    for name in (CPAP, ROBIN):
        d = data[name]
        out(f"{name}  --  {len(d['drop'])} central apneas measured")
        out(f"  falls >= {DROP_PP:.0f} points below baseline   "
            f"{100 * curve(d['drop'], w)[0]:5.1f} %   "
            f"(chance {100 * curve(d['sur_drop'], w)[0]:.1f} %)")
        out(f"  reaches {ABS_PCT:.0f} % or below             "
            f"{100 * curve(d['abs'], w)[0]:5.1f} %   "
            f"(chance {100 * curve(d['sur_abs'], w)[0]:.1f} %)")
        out()

    rows = []
    for name in (CPAP, ROBIN):
        for ww in WINDOWS:
            rows.append(dict(
                cohort=name, window_s=ww,
                pct_drop=100 * curve(data[name]["drop"], [ww])[0],
                pct_drop_chance=100 * curve(data[name]["sur_drop"], [ww])[0],
                pct_abs=100 * curve(data[name]["abs"], [ww])[0],
                pct_abs_chance=100 * curve(data[name]["sur_abs"], [ww])[0],
                n=len(data[name]["drop"])))
    csv = os.path.join("outputs", "apnea_desat_coupling_like.csv")
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"sweep -> {csv}")

    make_figure(data, args.out_dir)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
