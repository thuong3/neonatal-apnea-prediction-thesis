"""WHY A CENTRAL APNEA IS VISIBLE IN NASAL PRESSURE AND NOT IN MASK PRESSURE.

The thesis rests on one channel substitution. Vetter et al. (2024) found nasal
pressure the most informative modality; the Brainimmaturity recordings carry no
airflow channel at all, and the slot is filled by the pressure inside the CPAP
mask. Section 2.2 states the consequence in prose -- a pause in the infant's own
effort does not produce the drop a true airflow channel would show, because the
flow generator holds the mask at its set pressure regardless of what the infant
does. This figure shows it on the recordings instead of asserting it.

Two columns, one scored central apnea each:

  LEFT   a Robin-sequence infant. Nasal pressure, then both effort belts. The
         effort stops and the airflow channel stops with it.
  RIGHT  a Brainimmaturity infant. CPAP mask pressure, then both effort belts.
         The effort stops in the same way and the pressure trace does not move:
         it stays at the set pressure, in cmH2O, throughout the pause.

The right-hand event is drawn from a block on continuous positive pressure and
not from one of the two synchronised-ventilation blocks, following the same
restriction Section 3.2 applies to the band-power measurement. The blocks are
labelled by scripts/detect_ventilation_mode.py. Under synchronised ventilation
the contrast is if anything starker -- the machine keeps inflating at about 10
per minute straight through the pause, so the channel carries a breathing-like
rhythm while the infant is not breathing at all -- but that is the ventilator's
rhythm and a reader is entitled to object that it is not the CPAP case. Pass
--mode nippv to see it.

No cohort records both channels, so the two columns are necessarily different
infants from different studies. That is the point rather than a weakness of the
figure: the contrast being drawn is between two instrumentations, not between
two patients.

WHICH EVENT IS DRAWN
    Not the clearest one. Every scored central apnea in each cohort that meets
    the admissibility rules below is a candidate, and the event drawn is the one
    whose effort collapse is CLOSEST TO ITS COHORT'S MEDIAN -- a typical event,
    not a favourable one. The number of candidates and the median it was picked
    against are printed with the figure, so the selection can be checked. Pass
    --pick extreme to see the other end of the distribution; the qualitative
    contrast between the columns does not depend on it.

    Admissible means: a duration inside that cohort's own scoring rule (see
    `dur` in COHORTS -- the two rules differ, and the Robin pauses are shorter
    BY DEFINITION rather than by accident); the whole 40-second drawing window
    inside one analysis block; no other scored apnea in the 20 s of breathing
    before the onset; and no SIGNAL-ARTIFACT, SIGNAL-QUALITY-LOW or
    ACTIVITY-MOVE span anywhere in the window.

WHAT THE NUMBERS ON THE PANELS MEAN
    Each trace is annotated with its breathing-band (0.5-2 Hz) RMS during the
    apnea as a percentage of the same quantity over the 20 s before it. That is
    the claim of the figure reduced to one number per panel, computed from the
    trace that is drawn and nothing else. The mask-pressure panel is also the
    one axis that keeps real tick values, because the width of that axis -- a
    third of a cmH2O around the set pressure -- is the whole explanation.

    The band filter is a zero-phase filtfilt. This is a descriptive statistic
    over a window that is already fixed, not a model input, so the objection of
    Section 4.8 to zero-phase filtering does not apply here.

    Belt and nasal-pressure amplitudes are in arbitrary units -- the EDF
    declares uV and "m" for gain settings that were never calibrated to a volume
    or a flow -- so only the ratio within a panel carries meaning, never the
    absolute height. Mask pressure is the exception and is genuinely in cmH2O.

Usage:
    python scripts/make_flow_vs_mask_figure.py
    python scripts/make_flow_vs_mask_figure.py --replot            # from cache
    python scripts/make_flow_vs_mask_figure.py --simple            # talk version
    python scripts/make_flow_vs_mask_figure.py --pick extreme
    python scripts/make_flow_vs_mask_figure.py --mode nippv --replot
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
matplotlib.rc_file(os.path.join(ROOT, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402

FS = 200                     # every channel is up-sampled to this by read_data
PRE_S = 20.0                 # breathing shown, and measured, before the onset
POST_S = 20.0                # shown after the onset
BAND = (0.5, 2.0)            # neonatal breathing band, as used in Section 3.2

CPAP, ROBIN = "Brainimmaturity", "Robin sequence"

# `flow` is the channel occupying the airflow slot in each cohort. Everything
# else about the two columns is identical.
COHORTS = {
    ROBIN: dict(
        path="data/robin_sequence/"
             "data/neonatal_robin_polysomnography",
        sub="annotations_original", record_duration=10, drift=False,
        ids=["%03d" % i for i in range(1, 20)],
        flow="NP", flow_label="Nasal pressure", flow_unit="a.u.",
        # The AASM rule scores a central apnea in an infant at two breath
        # cycles, so these events have a median duration of 4.0 s and only 3 of
        # 742 reach 10 s. Applying Sievers' 10 s floor here would admit almost
        # nothing and would be the wrong rule for this cohort besides.
        dur=(4.0, 12.0),
        column="(a) Nasal pressure, Robin sequence"),
    CPAP: dict(
        path="data/"
             "dataset_brainimmaturity",
        sub="annotations", record_duration=None, drift=True,
        ids=["%03d" % i for i in range(1, 16)],
        flow="CPAP", flow_label="CPAP mask pressure", flow_unit="cmH$_2$O",
        # Sievers' own duration condition, and the upper bound keeps the
        # recovery breaths inside the drawn window.
        dur=(10.0, 25.0),
        column="(b) CPAP mask pressure, Brainimmaturity"),
}

MASKS_MUST_BE_CLEAR = ("SIGNAL-ARTIFACT", "SIGNAL-QUALITY-LOW", "ACTIVITY-MOVE")

# Written by scripts/detect_ventilation_mode.py: which of each infant's four
# analysis blocks ran on synchronised ventilation. The block index is the
# segment index `read_data` returns, which is how band_power.py joins it too.
MODES = os.path.join(ROOT, "outputs/ventilation_mode/modes.csv")

COLOR = {CPAP: "#2a78d6", ROBIN: "#eb6834"}
C_BELT = "#3f3f3c"
C_SPAN = "#e8e6e0"
C_GUIDE = "#c9c8c3"

CACHE = os.path.join(ROOT, "outputs/flow_vs_mask/traces.npz")
CANDIDATES = os.path.join(ROOT, "outputs/flow_vs_mask/candidates.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.path.join(ROOT, "thesis/figures"))
    p.add_argument("--out", default="outputs/flow_vs_mask.txt")
    p.add_argument("--replot", action="store_true",
                   help="redraw from the cached traces, without reading EDFs")
    p.add_argument("--simple", action="store_true",
                   help="talk version: no titles, no annotations, direct labels")
    p.add_argument("--pick", choices=("median", "extreme"), default="median",
                   help="typical event (default) or the deepest effort collapse")
    p.add_argument("--panels", choices=("traces", "distribution", "both"),
                   default="traces",
                   help="traces: the two example columns only, which is the "
                        "background figure (default). distribution: the "
                        "cohort-level curves on their own, as a separate "
                        "figure. both: the two stacked, as one figure.")
    p.add_argument("--cpap-rank", type=int, default=0,
                   help="step to the Nth most typical Brainimmaturity event "
                        "(0 = the most typical); for choosing between equally "
                        "representative examples")
    p.add_argument("--robin-rank", type=int, default=0,
                   help="the same for the Robin column")
    p.add_argument("--mode", choices=("cpap", "nippv", "any"), default="cpap",
                   help="Brainimmaturity block type to draw from (default cpap)")
    p.add_argument("--ids", nargs="*", default=None,
                   help="restrict the scan to these patient ids (both cohorts)")
    return p.parse_args()


def band_rms(x):
    """RMS of `x` inside the neonatal breathing band."""
    if len(x) < 4 * FS:
        return np.nan
    b, a = butter(2, BAND, btype="band", fs=FS)
    return float(np.sqrt(np.mean(filtfilt(b, a, np.asarray(x, float)) ** 2)))


def load_modes():
    """{(patient, block): "NIPPV" | "CPAP"}; empty if the table is missing."""
    if not os.path.exists(MODES):
        print("no %s -- ventilation mode not restricted" % MODES)
        return {}
    m = pd.read_csv(MODES, dtype={"pat": str})
    return {(r.pat, int(r.block)): r.mode for r in m.itertuples(index=False)}


def band_filter(x):
    """The 0.5-2 Hz component of `x`, which is what the percentages measure."""
    b, a = butter(2, BAND, btype="band", fs=FS)
    return filtfilt(b, a, np.asarray(x, float))


def runs(mask):
    """[(start, stop)] of the 1-runs in a 0/1 array, stop exclusive."""
    m = np.asarray(mask, int)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def scan(cfg, cohort, ids, modes):
    """Every admissible central apnea in one cohort, with its collapse ratios."""
    drift = load_clock_drift() if cfg["drift"] else {}
    pre, post = int(PRE_S * FS), int(POST_S * FS)
    rows, traces = [], {}
    for pid in ids:
        try:
            segments = read_data(pid, cfg["path"], annotations_dir=cfg["sub"],
                                 record_duration=cfg["record_duration"],
                                 clock_drift=drift.get(pid))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {cohort} {pid}: {exc}")
            continue
        for si, seg in enumerate(segments):
            need = (cfg["flow"], "Thorax", "Abdomen", "APNEA-CENTRAL")
            if any(k not in seg for k in need):
                continue
            ap = np.asarray(seg["APNEA-CENTRAL"], int)
            for s, e in runs(ap):
                dur = (e - s) / FS
                if not (cfg["dur"][0] <= dur <= cfg["dur"][1]):
                    continue
                w0, w1 = s - pre, s + post
                if w0 < 0 or w1 > len(ap):
                    continue
                if ap[w0:s].any():            # breathing, not a preceding pause
                    continue
                if any(np.asarray(seg[k], int)[w0:w1].any()
                       for k in MASKS_MUST_BE_CLEAR if k in seg):
                    continue
                flow = np.asarray(seg[cfg["flow"]], float)[w0:w1]
                thorax = np.asarray(seg["Thorax"], float)[w0:w1]
                abdomen = np.asarray(seg["Abdomen"], float)[w0:w1]
                belt = thorax + abdomen
                if not (np.isfinite(flow).all() and np.isfinite(belt).all()):
                    continue
                if np.ptp(flow) == 0 or np.ptp(belt) == 0:        # dead channel
                    continue
                d = int(dur * FS)
                r_flow = band_rms(flow[pre:pre + d]) / band_rms(flow[:pre])
                r_belt = band_rms(belt[pre:pre + d]) / band_rms(belt[:pre])
                if not (np.isfinite(r_flow) and np.isfinite(r_belt)):
                    continue
                key = "%s|%s|%d|%d" % (cohort, pid, si, s)
                rows.append(dict(cohort=cohort, patient=pid, segment=si,
                                 onset_sample=s, duration_s=dur,
                                 mode=modes.get((pid, si), "unknown"),
                                 flow_ratio=r_flow, belt_ratio=r_belt, key=key))
                traces[key + "|flow"] = flow
                traces[key + "|thorax"] = thorax
                traces[key + "|abdomen"] = abdomen
    return pd.DataFrame(rows), traces


def restrict(df, mode):
    """Drop Brainimmaturity events outside the requested block type."""
    if mode == "any" or "mode" not in df:
        return df
    want = {"cpap": "CPAP", "nippv": "NIPPV"}[mode]
    keep = (df["cohort"] != CPAP) | (df["mode"] == want)
    if not keep.any():
        return df
    return df[keep].copy()


def summarise(df):
    """{cohort: (n, median flow ratio)} over the events the figure draws from.

    The drawn event is one example; this is what its cohort does. Both go on
    the panel, so that the figure does not rest on the example alone.
    """
    return {c: (len(g), float(g["flow_ratio"].median()))
            for c, g in df.groupby("cohort")}


def choose(df, how, rank=None):
    """One row per cohort: the median collapse, or the deepest one.

    Sorted by patient and onset first, because with an even number of
    candidates the two middle events are EXACTLY equidistant from the median
    and `idxmin` would otherwise return whichever of them the row order
    happened to put first -- which differs between a fresh scan and a --replot
    off the cache, and would silently change the published figure.
    """
    out = {}
    key = ["patient", "segment", "onset_sample"]
    for cohort, g in df.sort_values(key).groupby("cohort"):
        if how == "extreme":
            out[cohort] = g.loc[g["belt_ratio"].idxmin()]
        else:
            # Typical in BOTH ratios, in log space: the event minimising the
            # larger of its two distances from the cohort medians. Selecting on
            # the flow ratio alone picked events whose belts were atypical, and
            # on the belts alone events whose airflow channel was.
            d = []
            for c in ("flow_ratio", "belt_ratio"):
                v = np.log(g[c].clip(lower=1e-3))
                d.append((v - v.median()).abs())
            order = pd.concat(d, axis=1).max(axis=1).sort_values()
            n = min((rank or {}).get(cohort, 0), len(order) - 1)
            out[cohort] = g.loc[order.index[n]]
    return out


def collect(args):
    modes = load_modes()
    frames, traces = [], {}
    for cohort, cfg in COHORTS.items():
        use = args.ids if args.ids else cfg["ids"]
        df, tr = scan(cfg, cohort, use, modes if cfg["drift"] else {})
        print("%s: %d admissible central apneas" % (cohort, len(df)))
        frames.append(df)
        traces.update(tr)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CANDIDATES, index=False)
    np.savez_compressed(CACHE, table=np.array(df.to_json()), **traces)
    print("candidates -> %s" % CANDIDATES)
    return df, traces


def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    df = pd.read_json(str(z["table"]))
    df["patient"] = ["%03d" % int(p) for p in df["patient"]]
    return df, {k: z[k] for k in z.files if k != "table"}


def ylim(x):
    # 1st-99th rather than a full range: one sigh or one movement transient in
    # a 40 s window would otherwise compress the breathing into a flat line.
    lo, hi = np.percentile(x, [1, 99])
    pad = 0.12 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def ecdf_panel(ax, full, mode):
    """The cohort-level claim: where each channel's amplitude ratio lands.

    The traces above are one event each, and one event cannot carry a claim
    about a channel when the spread within a cohort is this wide. This panel
    is every admissible event in both cohorts.
    """
    def curve(v, **kw):
        v = 100 * np.sort(np.asarray(v, float))
        v = np.clip(v, 0.3, 300)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kw)

    cp = full[(full["cohort"] == CPAP) & (full["mode"] != "NIPPV")]
    vent = full[(full["cohort"] == CPAP) & (full["mode"] == "NIPPV")]
    rb = full[full["cohort"] == ROBIN]

    curve(rb["flow_ratio"], color=COLOR[ROBIN], linewidth=1.4,
          label="Nasal pressure, Robin (n=%d)" % len(rb))
    curve(cp["flow_ratio"], color=COLOR[CPAP], linewidth=1.4,
          label="Mask pressure, CPAP blocks (n=%d)" % len(cp))
    if len(vent):
        curve(vent["flow_ratio"], color=COLOR[CPAP], linewidth=1.2,
              linestyle="--",
              label="Mask pressure, ventilated blocks (n=%d)" % len(vent))
    # The belts are the reference: they say the effort really did stop. Kept
    # per cohort rather than pooled, because they do not behave the same --
    # the cardiac artifact and the baseline drift that Section 5.2 describes
    # hold the Brainimmaturity belts up at roughly three times the Robin
    # figure, and pooling would hide exactly that.
    curve(rb["belt_ratio"], color=COLOR[ROBIN], linewidth=1.0, linestyle=":",
          label="Effort belts, Robin")
    curve(cp["belt_ratio"], color=COLOR[CPAP], linewidth=1.0, linestyle=":",
          label="Effort belts, Brainimmaturity")

    ax.axvline(100, color=C_GUIDE, linewidth=1.0, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 250)
    ax.set_xticks([1, 3, 10, 30, 100])
    ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("0.5-2 Hz amplitude during the pause, as a %% of the "
                  "%.0f s before [%%, log scale]" % PRE_S)
    ax.set_ylabel("Fraction of\nevents")
    ax.legend(loc="upper left", handlelength=2.2, fontsize=7,
              labelspacing=0.3, borderpad=0.2)


def make_figure(picked, traces, stats, full, out_dir, simple=False,
                panels="traces"):
    t = np.arange(-PRE_S * FS, POST_S * FS) / FS
    want_traces = panels in ("traces", "both")
    want_dist = panels in ("distribution", "both")

    if panels == "both":
        # Two gridspecs rather than one: the lower panel needs a real gap
        # above it for the x-label of the row it sits under, which a shared
        # hspace cannot give.
        fig = plt.figure(figsize=(7.2, 5.4))
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26,
                              top=0.95 if simple else 0.905, bottom=0.50,
                              left=0.095, right=0.985)
        gs_c = fig.add_gridspec(1, 1, top=0.37, bottom=0.085,
                                left=0.095, right=0.985)
    elif want_traces:
        fig = plt.figure(figsize=(7.2, 3.3))
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.26,
                              top=0.92 if simple else 0.845, bottom=0.145,
                              left=0.095, right=0.985)
        gs_c = None
    else:
        fig = plt.figure(figsize=(7.2, 2.9))
        gs = None
        gs_c = fig.add_gridspec(1, 1, top=0.90 if simple else 0.855,
                                bottom=0.175, left=0.095, right=0.985)

    for col, cohort in enumerate((ROBIN, CPAP) if want_traces else ()):
        cfg, row = COHORTS[cohort], picked[cohort]
        dur = float(row["duration_s"])
        rows = [(cfg["flow_label"], "flow", COLOR[cohort], cfg["flow_unit"]),
                ("Effort belts, summed", "belt", C_BELT, "a.u.")]
        for r, (label, kind, color, unit) in enumerate(rows):
            ax = fig.add_subplot(gs[r, col])
            if kind == "belt":
                # The model's own effort channel is the sum of the two belts
                # (SIGNAL_LABEL_MAP in src/neonatal_utils.py), so the figure
                # shows what the network is given rather than two near-copies.
                x = (traces[row["key"] + "|thorax"]
                     + traces[row["key"] + "|abdomen"])
            else:
                x = traces[row["key"] + "|" + kind]
            ax.axvspan(0, dur, color=C_SPAN, zorder=0)
            ax.axvline(0, color=C_GUIDE, linewidth=1.0, zorder=1)
            # Raw channel faint, its 0.5-2 Hz component solid on top. Without
            # the second line a reader sees the slow excursion the mask
            # pressure makes around an event and reads it as a response to the
            # apnea, when the number quoted below counts only the band.
            ax.plot(t, x, color=color, linewidth=0.7, alpha=0.30)
            ax.plot(t, band_filter(x) + np.median(x), color=color,
                    linewidth=0.8)
            ax.set_xlim(-PRE_S, POST_S)
            ax.set_ylim(*ylim(np.concatenate(
                [x, band_filter(x) + np.median(x)])))
            ax.set_ylabel(label if simple else "%s\n[%s]" % (label, unit))
            if kind == "flow" and unit.startswith("cmH"):
                # The set point is the explanation, so this one axis keeps its
                # real numbers while the arbitrary-unit traces drop theirs.
                # The two ticks are the span of the trace: a reader who sees
                # them a few tenths of a cmH2O apart has the whole story.
                lo, hi = np.percentile(x, [5, 95])
                nd = 1 if hi - lo >= 0.5 else 2
                ax.set_yticks(np.round([lo, hi], nd))
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FormatStrFormatter("%%.%df" % nd))
            else:
                ax.set_yticks([])
            if r == 0 and not simple:
                # pad clears the apnea-duration label, which sits just above
                # the axes and would otherwise share this line.
                ax.set_title(cfg["column"], loc="left", pad=14)
                if col == 0:
                    # Named on the figure rather than only in the caption,
                    # since the two lines are the first thing a reader asks
                    # about. The filtered line is zero-mean and drawn at the
                    # raw median, so its height means nothing and its
                    # amplitude everything -- hence "component", not "signal".
                    ax.plot([], [], color=color, alpha=0.30, linewidth=0.7,
                            label="as recorded")
                    ax.plot([], [], color=color, linewidth=0.8,
                            label="0.5-2 Hz component")
                    ax.legend(loc="upper left", fontsize=6.5,
                              handlelength=1.6, labelspacing=0.25,
                              borderpad=0.15, borderaxespad=0.2)
            if r == 0:
                # Above the axes, where it cannot land on the trace.
                ax.text(dur / 2, 1.005, "scored central apnea, %.0f s" % dur,
                        transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=7, color="#5a5a55")
            if r == 1:
                ax.set_xlabel("Time from apnea onset [s]")
            else:
                ax.set_xticklabels([])
            if simple:
                continue
            if kind == "flow":
                # Bottom right on an opaque patch, so the trace may run
                # underneath without either becoming unreadable.
                n, med = stats[cohort]
                ax.text(0.99, 0.05,
                        "0.5-2 Hz amplitude during the pause, as a %% of the "
                        "%.0f s before\nthis event %.0f %%      cohort median "
                        "%.1f %% over %d events"
                        % (PRE_S, 100 * row["flow_ratio"], 100 * med, n),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="#5a5a55", linespacing=1.4,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=1.5))

    if want_dist:
        ax = fig.add_subplot(gs_c[0, 0])
        ecdf_panel(ax, full, None)
        if not simple:
            title = ("(c) Every admissible event in both cohorts"
                     if panels == "both"
                     else "Every scored central apnea in both cohorts")
            ax.set_title(title, loc="left", pad=6)

    os.makedirs(out_dir, exist_ok=True)
    stem = ("flow_vs_mask_distribution" if panels == "distribution"
            else "flow_vs_mask_pressure") + ("_simple" if simple else "")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (stem, ext)),
                    bbox_inches="tight")
    plt.close(fig)
    print("figure -> %s/%s.pdf" % (out_dir, stem))


def main():
    args = parse_args()
    if args.replot and os.path.exists(CACHE):
        df, traces = load_cache()
    else:
        df, traces = collect(args)
    if df.empty:
        raise SystemExit("no admissible apnea found in either cohort")

    full = df                      # every cohort and block, for panel (c)
    df = restrict(df, args.mode)   # what the example traces come from
    picked = choose(df, args.pick,
                    rank={CPAP: args.cpap_rank, ROBIN: args.robin_rank})
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("Breathing-band (%.1f-%.1f Hz) RMS during a scored central apnea, as a "
        "fraction of" % BAND)
    out("the same quantity over the %.0f s of breathing before it. One event "
        "per cohort," % PRE_S)
    out("selected as the %s effort collapse among the admissible events."
        % args.pick)
    out("Brainimmaturity blocks restricted to: %s." % args.mode)
    out()
    for cohort in (ROBIN, CPAP):
        g, row = df[df["cohort"] == cohort], picked[cohort]
        out("%s  --  %d admissible central apneas, median belt ratio %.3f"
            % (cohort, len(g), g["belt_ratio"].median()))
        out("  drawn: patient %s, block %d (%s), onset sample %d, %.0f s"
            % (row["patient"], row["segment"], row.get("mode", "n/a"),
               row["onset_sample"], row["duration_s"]))
        out("  effort belts         %5.1f %%  of the preceding %.0f s"
            % (100 * row["belt_ratio"], PRE_S))
        out("  %-20s %5.1f %%"
            % (COHORTS[cohort]["flow_label"], 100 * row["flow_ratio"]))
        out()

    make_figure(picked, traces, summarise(df), full, args.out_dir,
                simple=args.simple, panels=args.panels)

    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("written -> %s" % path)


if __name__ == "__main__":
    main()
