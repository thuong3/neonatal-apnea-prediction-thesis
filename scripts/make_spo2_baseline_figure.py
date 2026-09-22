"""Figure: the TWO oxygenation phenomena, drawn on real recordings.

Makes the supervisor's point visually -- that "SpO2 deteriorates" means two
different things that the current 30 s z-scored window cannot tell apart:

  A  frank desaturation / intermittent hypoxemia (IH): short dips below
     90 % (Dormishian et al. 2023: < 90 % for >= 5 s; severe < 80 %);
  B  a decline of the BASELINE -- the level between the dips -- over tens of
     minutes, with few or no frank desaturations.

Panel 1  SpO2 at 1 Hz with the causal 30 min baseline B30 overlaid, the 90/85/80 %
         thresholds, and the scored central apneas marked.
Panel 2  The predictor itself: M - B30 (window mean minus long-term baseline),
         which is what separates B from a merely low-but-stable recording.
Panel 3  IH burden (phenomenon A) over a rolling 30 min lookback, so the two
         can be read against each other on the same time axis.

Usage:
    python scripts/make_spo2_baseline_figure.py --patient 010 --segment 3
    python scripts/make_spo2_baseline_figure.py --patient 010 --segment 3 --simple
    python scripts/make_spo2_baseline_figure.py --survey     # all patients, drift table
"""
import argparse
import os
import sys
import warnings

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rc_file(os.path.join(_ROOT, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

from src.neonatal_utils import load_clock_drift, read_data  # noqa: E402
from scripts.spo2_baseline_prediction import (  # noqa: E402
    FS, HZ, IH_THRESHOLD, IH_SEVERE, IH_MIN_DUR_S, IH_MERGE_GAP_S, WIN_S,
    episodes, spo2_context, onsets,
)

DEFAULT_DATA = r"data/brainimmaturity"
C_SPO2, C_BASE, C_A, C_B = "#3a3a3a", "#c1272d", "#1f77b4", "#c1272d"
C_IH = "#7a1f6d"   # the desaturation row, and severe IH in panel 3


def rolling_frac(x, w):
    """Causal fraction of `x` (0/1) over the past w samples."""
    cs = np.concatenate([[0.0], np.cumsum(x.astype(float))])
    hi = np.arange(len(x))
    lo = np.maximum(0, hi - w)
    return (cs[hi] - cs[lo]) / np.maximum(1, hi - lo)


def draw(pid, si, data_path, out_dir):
    drift = load_clock_drift()
    segs = read_data(pid, data_path, clock_drift=drift.get(pid))
    if si >= len(segs):
        raise SystemExit(f"patient {pid} has {len(segs)} segments, {si} requested")
    seg = segs[si]
    ctx = spo2_context(seg)
    s, valid, B = ctx["spo2"], ctx["valid"], ctx["B30"]
    step = FS // HZ
    ap = np.asarray(seg["APNEA-CENTRAL"][::step], bool)
    t = np.arange(len(s)) / 3600.0

    sp_plot = np.where(valid, s, np.nan)
    # M - B30 on the same sliding window the model uses
    w = WIN_S * HZ
    cs = np.concatenate([[0.0], np.cumsum(np.where(valid, s, 0.0))])
    cv = np.concatenate([[0.0], np.cumsum(valid.astype(float))])
    hi = np.arange(len(s))
    lo = np.maximum(0, hi - w)
    cnt = cv[hi] - cv[lo]
    M = np.where(cnt >= w // 2, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    diff = M - B

    below90 = rolling_frac(valid & (s < IH_THRESHOLD), 30 * 60 * HZ) * 100
    below80 = rolling_frac(valid & (s < IH_SEVERE), 30 * 60 * HZ) * 100

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})

    ax = axes[0]
    ax.plot(t, sp_plot, color=C_SPO2, lw=0.4, label="SpO$_2$ (1 Hz)")
    ax.plot(t, B, color=C_BASE, lw=2.0,
            label="causal 30 min baseline $B_{30}$ (dips excluded)")
    for thr, ls in [(90, "--"), (85, ":"), (80, "-.")]:
        ax.axhline(thr, color="0.55", lw=0.8, ls=ls)
        ax.text(t[-1], thr, f" {thr}%", color="0.4", va="center", fontsize=6)
    # Ticks along the top rather than full-height hairlines: at lw 0.3 and
    # alpha 0.25 they were invisible behind the 1 Hz trace, which made the
    # title promise a mark the eye could not find.
    apx = np.array([a / 3600.0 for a in onsets(ap)])
    ax.vlines(apx, 100.6, 102, color=C_A, lw=1.0, alpha=0.9, zorder=4)
    ax.set_ylim(60, 102)
    ax.set_ylabel("SpO$_2$ (%)")
    ax.legend(loc="lower left", ncol=1)
    ax.set_title(f"Patient {pid}, segment {si}: the two phenomena "
                 f"(blue ticks on top = {len(apx)} scored central apneas)")

    ax = axes[1]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(t, diff, color=C_B, lw=0.8)
    ax.fill_between(t, 0, diff, where=np.isfinite(diff) & (diff < 0),
                    color=C_B, alpha=0.25, lw=0)
    ax.set_ylabel("$M - B_{30}$ (%)")
    lo_d = np.nanpercentile(diff, 0.5) - 1
    hi_d = max(2.0, np.nanpercentile(diff, 99.5))
    ax.set_ylim(lo_d, hi_d + 0.35 * (hi_d - lo_d))   # headroom for the label
    ax.text(0.005, 0.88, "phenomenon B: window mean below own baseline",
            transform=ax.transAxes, color=C_B, fontsize=7, va="top")

    ax = axes[2]
    ax.plot(t, below90, color=C_A, lw=1.2, label="time < 90 % (IH)")
    ax.plot(t, below80, color="#7a1f6d", lw=1.2, label="time < 80 % (severe IH)")
    ax.set_ylabel("% of past 30 min")
    ax.set_xlabel("time within analysis segment (h)")
    ax.set_ylim(0, max(1.0, float(np.nanmax(below90))) * 1.55)  # headroom for labels
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.005, 0.94, "phenomenon A: frank desaturation burden",
            transform=ax.transAxes, color=C_A, fontsize=7, va="top")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"spo2_two_phenomena_pat{pid}_seg{si}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.png / .pdf")

    fin = np.isfinite(B)
    print(f"  baseline B30 range within this segment: "
          f"{np.nanmin(B[fin]):.1f} - {np.nanmax(B[fin]):.1f} % "
          f"(span {np.nanmax(B[fin]) - np.nanmin(B[fin]):.1f} %)")
    print(f"  time < 90 %: {(valid & (s < 90)).sum() / max(1, valid.sum()) * 100:.2f} %"
          f" | scored central apneas: {len(onsets(ap))}")


def draw_simple(pid, si, data_path, out_dir):
    """One panel, for talking over: the trace, its baseline, and two event rows.

    The apneas and the desaturations get a row each above the trace, so the two
    can be compared tick for tick. Under CPAP most desaturations have no apnea
    above them, which is the thing the three-panel version only implies.
    """
    drift = load_clock_drift()
    segs = read_data(pid, data_path, clock_drift=drift.get(pid))
    if si >= len(segs):
        raise SystemExit(f"patient {pid} has {len(segs)} segments, {si} requested")
    seg = segs[si]
    ctx = spo2_context(seg)
    s, valid, B = ctx["spo2"], ctx["valid"], ctx["B30"]
    t = np.arange(len(s)) / 3600.0
    ap = onsets(np.asarray(seg["APNEA-CENTRAL"][::FS // HZ], bool))
    ds = episodes(valid & (s < IH_THRESHOLD), IH_MIN_DUR_S * HZ,
                  IH_MERGE_GAP_S * HZ)

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(t, np.where(valid, s, np.nan), color=C_SPO2, lw=0.4, label="SpO$_2$")
    ax.plot(t, B, color=C_BASE, lw=2.0, label="30 min baseline")
    ax.axhline(IH_THRESHOLD, color="0.55", lw=0.8, ls="--")
    ax.text(t[-1] + 0.03, IH_THRESHOLD, "90%", color="0.4", va="center",
            fontsize=7.5)

    # Two event rows, one above the other, same time axis as the trace.
    rows = [(ap, 106.5, 109.5, C_A, f"central apneas ({len(ap)})"),
            ([e[0] for e in ds], 102.5, 105.5, C_IH,
             f"desaturations < 90 % ({len(ds)})")]
    for xs, y0, y1, colour, label in rows:
        ax.vlines([x / 3600.0 for x in xs], y0, y1, color=colour, lw=1.1,
                  alpha=0.9, clip_on=False)
        ax.text(t[-1] + 0.03, (y0 + y1) / 2, label, color=colour, fontsize=7.5,
                va="center", clip_on=False)

    ax.set_ylim(60, 110)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.spines["left"].set_bounds(60, 100)
    ax.set_ylabel("SpO$_2$ (%)")
    ax.set_xlabel("time (h)")
    ax.legend(loc="lower left", ncol=2, fontsize=8)

    fig.subplots_adjust(left=0.085, right=0.755, top=0.97, bottom=0.145)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"spo2_simple_pat{pid}_seg{si}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {base}.png / .pdf  ({len(ap)} apneas, {len(ds)} desats)")


def survey(data_path):
    """Per-segment baseline span, to pick the clearest examples and to quantify
    how much baseline movement the cohort actually contains."""
    drift = load_clock_drift()
    print(f"{'pat':>4} {'seg':>4} {'hours':>6} {'B30 min':>8} {'B30 max':>8} "
          f"{'span':>6} {'%<90':>6} {'apneas':>7}")
    rows = []
    for pid in [f"{i:03d}" for i in range(1, 16)]:
        for si, seg in enumerate(read_data(pid, data_path, clock_drift=drift.get(pid))):
            ctx = spo2_context(seg)
            s, valid, B = ctx["spo2"], ctx["valid"], ctx["B30"]
            fin = np.isfinite(B)
            if fin.sum() < 600:
                continue
            lo_, hi_ = float(np.nanmin(B[fin])), float(np.nanmax(B[fin]))
            nap = len(onsets(np.asarray(seg["APNEA-CENTRAL"][::FS // HZ], bool)))
            f90 = (valid & (s < 90)).sum() / max(1, valid.sum()) * 100
            rows.append((pid, si, hi_ - lo_))
            print(f"{pid:>4} {si:>4} {len(s)/3600:>6.2f} {lo_:>8.1f} {hi_:>8.1f} "
                  f"{hi_-lo_:>6.1f} {f90:>6.2f} {nap:>7d}")
    rows.sort(key=lambda r: -r[2])
    print("\nlargest baseline spans (best figure candidates):")
    for pid, si, sp_ in rows[:8]:
        print(f"  patient {pid} segment {si}: {sp_:.1f} %")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", default="010")
    p.add_argument("--segment", type=int, default=3)
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--out_dir", default=os.path.join(_ROOT, "figures", "06_spo2_baseline"))
    p.add_argument("--survey", action="store_true")
    p.add_argument("--simple", action="store_true",
                   help="one-panel version for presenting")
    a = p.parse_args()
    if a.survey:
        survey(a.data_path)
    elif a.simple:
        draw_simple(a.patient, a.segment, a.data_path, a.out_dir)
    else:
        draw(a.patient, a.segment, a.data_path, a.out_dir)


if __name__ == "__main__":
    main()
