"""B4's seventh channel: what it is, why it should have worked, and why it did not.

B4 adds `effort(t) / median(effort over [t-120 s, t-15 s])` as a seventh input
channel (`config/experiment/effort_baseline_channel.yaml`). It scored 0.5721
against `norm_none`'s 0.5747 -- inside seed spread, and short of the 0.60
registered before the run (Section 4.4 of the thesis SS33.2).

A bar chart of 0.5721 against 0.5747 shows the verdict and none of the reason.
This figure shows the reason, in four panels:

  a  THE CHANNEL ON REAL SIGNAL. The summed belt envelope with the causal
     trailing median it is divided by, and the resulting channel underneath,
     with the scored apneas marked and the two published criteria drawn as
     lines: B4's own "below 20 % of the preceding breaths" and Adams et al.'s
     "below 50 % of baseline", the claim the channel was built to encode.
  b  WHY IT SHOULD HAVE WORKED. The network's input window is 30 s; the
     baseline reaches 120 s back. The channel is the only route by which
     anything older than the window enters the model, which is exactly why
     SS27.6's -0.025 for per-window standardisation does not bound it.
  c  WHY IT DID NOT. The channel's distribution over pre-apnea windows against
     control windows, pooled over all 15 infants. If effort declined before
     central apneas, these separate. They do not.
  d  THE CRITERION, PRICED. How often each rule actually fires in a pre-apnea
     window and in a control window. Adams' finding was made on events with
     SpO2 < 80 % in a different population; this is what it is worth here.

Panels c and d are single-feature rank statistics over a window summary, so they
are a FLOOR, not a model result -- the network sees a 30 s waveform per channel
and can combine channels non-additively. Read them the way SS27.4 reads its
preview: evidence about where the information is, not proof that no architecture
could use the channel. The sweep was run anyway and agrees (SS33.2).

The envelope and the trailing baseline are recomputed here so panel a can draw
them, then ASSERTED to reproduce `effort_baseline_ratio` exactly. If that assert
ever fires, this figure has drifted from the channel the network was given and
the numbers on it are not about B4 any more.

Usage:
    python scripts/make_effort_channel_figure.py                    # all 15
    python scripts/make_effort_channel_figure.py --ids 001 002      # quick
    python scripts/make_effort_channel_figure.py --replot           # from cache
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
matplotlib.rc_file(os.path.join(REPO, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

from src.neonatal_utils import (  # noqa: E402
    DECIMATION_CHAIN,
    EFFORT_BASE_FROM_S,
    EFFORT_BASE_TO_S,
    EFFORT_ENV_WIN_S,
    EFFORT_LOG_CLIP,
    EFFORT_MIN_HISTORY_S,
    TARGET_FREQ,
    NeoNatal,
    _apply_chain,
    apply_robust,
    effort_baseline_ratio,
    load_clock_drift,
    read_data,
    robust_stats,
)

DEFAULT_DATA = "data/brainimmaturity"
CACHE = os.path.join(REPO, "outputs/effort_channel/windows.npz")
TRACE_CACHE = os.path.join(REPO, "outputs/effort_channel/trace.npz")

# The B4 scale floors, copied from config/experiment/effort_baseline_channel.yaml.
# `Thorax_sum` is the one that matters: without it patient 009 block 2's dead
# Abdomen belt enters the sum in raw units and 55.9 % of that block's channel
# samples change.
SCALE_FLOOR = {"Abdomen": 5.228, "CPAP": 0.09263, "PR": 5.468,
               "Thorax": 3.273, "Thorax_sum": 0.3637}

# The channel is log2(ratio) clipped to +-3 and divided by 3, so a ratio r maps
# to log2(r)/3. Both published criteria are fixed points on that scale.
B4_CRITERION = np.log2(0.20) / EFFORT_LOG_CLIP     # -0.774, "below 20 % of baseline"
ADAMS_CRITERION = np.log2(0.50) / EFFORT_LOG_CLIP  # -0.333, Adams et al. 1997

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTE = "#898781"
GRID = "#e1e0d9"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--replot", action="store_true",
                   help="plot from the cache, read no EDFs")
    p.add_argument("--trace_id", default="010")
    p.add_argument("--out", default="figures/03_results/effort_channel_b4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# the channel, with its intermediates exposed
# ---------------------------------------------------------------------------
def channel_with_parts(seg, artifact_free, scale_floor):
    """(channel, envelope, baseline) at 5 Hz, plus the channel at TARGET_FREQ.

    Reproduces `effort_baseline_ratio` step for step. The duplication is
    deliberate and is checked: panel a needs the envelope and the baseline,
    which the library function does not return, and a figure drawn from a
    paraphrase of the channel would not be a figure about B4.
    """
    fs = TARGET_FREQ
    n = min(len(seg["Thorax"]), len(seg["Abdomen"]))

    belts = []
    for channel in ("Thorax", "Abdomen"):
        x = np.asarray(seg[channel][:n], dtype=float)
        dec = _apply_chain(x, DECIMATION_CHAIN[channel])
        mask = None
        if artifact_free is not None and len(artifact_free):
            idx = np.clip((np.arange(len(dec)) * (len(artifact_free) / len(dec))
                           ).astype(int), 0, len(artifact_free) - 1)
            mask = np.asarray(artifact_free)[idx].astype(bool)
        stats = robust_stats(dec, mask)
        belts.append(apply_robust(x, stats, scale_floor.get(channel)))
    summed = belts[0] + belts[1]

    step = fs // 5
    env_win = max(1, int(EFFORT_ENV_WIN_S * fs))
    sq = pd.Series(summed ** 2).rolling(env_win, min_periods=1, center=True).mean()
    env = np.sqrt(sq.to_numpy())[::step]

    span = int((EFFORT_BASE_FROM_S - EFFORT_BASE_TO_S) * 5)
    gap = int(EFFORT_BASE_TO_S * 5)
    min_per = max(1, int(EFFORT_MIN_HISTORY_S * 5))
    base = pd.Series(env).rolling(span, min_periods=min_per).median().shift(gap)

    live = np.isfinite(env) & (env > 0)
    dead_floor = 0.05 * float(np.median(env[live])) if live.any() else 0.0
    b = base.to_numpy()
    ok = np.isfinite(b) & (b > dead_floor)
    ratio = np.ones_like(env, dtype=float)
    np.divide(env, b, out=ratio, where=ok)
    ratio[~ok] = 1.0
    ratio[ratio <= 0] = 1.0

    chan5 = np.clip(np.log2(ratio), -EFFORT_LOG_CLIP, EFFORT_LOG_CLIP) / EFFORT_LOG_CLIP
    return chan5, env, b, np.repeat(chan5, step)[:n]


def artifact_free_mask(seg, cutter_events):
    """Boolean mask at TARGET_FREQ, from the CUTTER events only -- never the
    adverse events, which would let the labels reach the statistics."""
    if not cutter_events:
        return None
    return (1 - np.column_stack([seg[e] for e in cutter_events]).max(axis=1)).astype(bool)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect(pat_id, data_path, cfg, drift, want_trace):
    signal_dict = read_data(
        pat_id, data_path,
        annotations_dir=cfg.get("annotations_dir", "annotations"),
        record_duration=cfg.get("record_duration", None),
        clock_drift=drift.get(pat_id))
    segments = [signal_dict] if isinstance(signal_dict, dict) else signal_dict

    ds = NeoNatal(
        pat_id, signal_dict, dataset_mode="list",
        signal_types=cfg.signal_types, adverse_events=cfg.adverse_events,
        cutter_events=cfg.cutter_events, time_window=cfg.time_window,
        lag=cfg.lag, away=cfg.away,
        norm_per_window=False, norm_per_block=False)
    df = ds.time_window_df
    if df is None or not len(df):
        return [], None

    rows, trace = [], None
    for si, seg in enumerate(segments):
        af = artifact_free_mask(seg, cfg.cutter_events)
        chan5, env, base, chan = channel_with_parts(seg, af, SCALE_FLOOR)

        # The whole point of recomputing: prove it is the same channel.
        reference = effort_baseline_ratio(seg, artifact_free=af,
                                          scale_floor=SCALE_FLOOR)
        m = min(len(reference), len(chan))
        if not np.allclose(chan[:m], reference[:m], atol=1e-12, equal_nan=True):
            raise SystemExit(
                f"patient {pat_id} segment {si}: the recomputed channel does not "
                "match effort_baseline_ratio. This figure would misrepresent B4; "
                "reconcile the two before using it.")

        sub = df[df["segment_idx"] == si] if "segment_idx" in df else df
        for _, r in sub.iterrows():
            sl = r["slice"]
            w = chan[sl]
            w = w[np.isfinite(w)]
            if not len(w):
                continue
            # Time BELOW the threshold, not "ever crosses it". The RMS envelope
            # dips between breaths, so an any-sample test fires in ~2/3 of
            # control windows and measures the breathing cycle rather than the
            # criterion. A fraction is also what SS24 carries as its burden
            # feature, so this matches the project's own reading of Adams.
            rows.append((int(r["label"]), float(w.mean()), float(w.min()),
                         float((w <= B4_CRITERION).mean()),
                         float((w <= ADAMS_CRITERION).mean())))

        if want_trace and trace is None and (sub["label"] == 1).any():
            # A stretch centred on a scored apnea, long enough that the 105 s
            # baseline window is visible next to the 30 s model window.
            pos = sub[sub["label"] == 1]
            centre = int(pos.iloc[len(pos) // 2]["slice"].stop)
            half = int(210 * 5)
            c5 = centre // (TARGET_FREQ // 5)
            lo, hi = max(0, c5 - half), min(len(chan5), c5 + half)
            if hi - lo > 400:
                apnea = np.asarray(seg["APNEA-CENTRAL"], dtype=float)[::TARGET_FREQ // 5]
                trace = dict(
                    env=env[lo:hi], base=base[lo:hi], chan=chan5[lo:hi],
                    apnea=apnea[lo:hi] if len(apnea) >= hi else np.zeros(hi - lo),
                    t=(np.arange(lo, hi) - c5) / 5.0,
                    pat=pat_id, seg=si)
    return rows, trace


def build_cache(args):
    cfg = OmegaConf.load(os.path.join(REPO, args.dataset))
    drift = load_clock_drift()
    ids = args.ids or list(cfg.ids)

    all_rows, trace = [], None
    for pid in ids:
        want = trace is None and (pid == args.trace_id or args.trace_id not in ids)
        try:
            rows, tr = collect(pid, args.data_path, cfg, drift, want)
        except SystemExit:
            raise
        except Exception as exc:                      # noqa: BLE001
            print(f"  {pid}: skipped ({type(exc).__name__}: {exc})")
            continue
        all_rows += [(pid,) + r for r in rows]
        if tr is not None and trace is None:
            trace = tr
        print(f"  {pid}: {len(rows)} windows")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    arr = np.array([r[1:] for r in all_rows], dtype=float)
    np.savez(CACHE, pids=np.array([r[0] for r in all_rows]), values=arr)
    if trace is not None:
        np.savez(TRACE_CACHE, **{k: np.asarray(v) for k, v in trace.items()})
    print(f"cached {len(all_rows)} windows -> {CACHE}")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def panel_trace(ax_top, ax_bot, tr):
    t = tr["t"]
    ax_top.plot(t, tr["env"], lw=0.8, color=INK2, label="summed belt envelope")
    ax_top.plot(t, tr["base"], lw=1.8, color=ORANGE,
                label="causal trailing median, [t-120 s, t-15 s]")
    ax_top.set_ylabel("effort\n(a.u.)")
    ax_top.legend(loc="upper right", fontsize=7, ncol=1,
                  framealpha=0.9)
    ax_top.set_title(
        f"a   The channel on real signal — patient {str(tr['pat'])}, "
        "the scored apnea shaded",
        loc="left", fontweight="bold", fontsize=8.5)

    ax_bot.plot(t, tr["chan"], lw=0.9, color=BLUE)
    ax_bot.axhline(0, color=MUTE, lw=1)
    for y, lab, style in [
        (ADAMS_CRITERION, "Adams: below 50 % of baseline", (0, (4, 3))),
        (B4_CRITERION, "B4: below 20 % of baseline", (0, (1, 2))),
    ]:
        ax_bot.axhline(y, color=ORANGE, lw=1.1, linestyle=style)
        ax_bot.text(t[-1], y, "  " + lab, va="center", ha="left", fontsize=6.5,
                    color=ORANGE, clip_on=False)
    ax_bot.set_ylabel("channel\n$\\log_2(e/\\bar e)/3$")
    ax_bot.set_xlabel("seconds relative to the end of the prediction window")
    ax_bot.set_ylim(-1.05, 1.05)

    for ax in (ax_top, ax_bot):
        a = np.asarray(tr["apnea"], dtype=float)
        lo, hi = ax.get_ylim()
        if a.any():
            ax.fill_between(t, lo, hi, where=a > 0.5, color=ORANGE,
                            alpha=0.22, lw=0, zorder=0)
        # The 30 s the network actually receives, so the shaded apnea and the
        # window the model had to predict it from are both visible at once.
        ax.axvspan(-30, 0, color=BLUE, alpha=0.10, lw=0, zorder=0)
        ax.set_ylim(lo, hi)
        ax.set_xlim(t[0], t[-1])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)

    ax_top.text(-15, ax_top.get_ylim()[1], "model window ", ha="right",
                va="top", fontsize=6.5, color=BLUE)
    # Shared time axis: only the lower row carries tick labels.
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlabel("")


def panel_geometry(ax):
    ax.set_title("b   Why it could have helped", loc="left",
                 fontweight="bold", fontsize=8.5)
    # Same second-axis as panel a: t = 0 ends the 30 s window, the apnea starts
    # 15 s later. The two spans are the point of the panel, so they get y tick
    # labels rather than text sitting inside the plot beside them.
    bars = [
        ("the network's\ninput window", -30, 0, BLUE, 1.0),
        ("the channel's\nbaseline", -120, -15, ORANGE, 0.0),
    ]
    for _label, x0, x1, colour, y in bars:
        ax.barh(y, x1 - x0, left=x0, height=0.38, color=colour)
        ax.text((x0 + x1) / 2, y, f"{int(x1 - x0)} s", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

    ax.axvline(0, color=INK, lw=1.4)
    ax.axvspan(15, 25, color=ORANGE, alpha=0.18, lw=0)
    ax.text(20, 1.95, "apnea", ha="center", va="bottom", fontsize=7, color=ORANGE)
    ax.annotate("", xy=(15, 1.72), xytext=(0, 1.72),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(7.5, 1.78, "15 s", ha="center", va="bottom", fontsize=7, color=INK2)

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([b[0] for b in bars], fontsize=7.5, color=INK2,
                       linespacing=1.4)
    ax.set_xlim(-132, 40)
    ax.set_ylim(-0.75, 2.25)
    ax.set_xticks([-120, -90, -60, -30, 0, 30])
    ax.set_xlabel("seconds relative to the end of the prediction window")
    ax.tick_params(axis="y", length=0)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.text(0.0, -0.30,
            "The baseline reaches 105 s further back than the window, so the "
            "channel is the only\nroute by which anything older than 30 s "
            "reaches the model.",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color=INK2, linespacing=1.45)


def panel_distribution(ax, values):
    lab, mean = values[:, 0], values[:, 1]
    pos, neg = mean[lab == 1], mean[lab == 0]
    lo, hi = np.percentile(mean, [0.5, 99.5])
    bins = np.linspace(lo, hi, 46)
    ax.hist(neg, bins=bins, density=True, color=BLUE, alpha=0.55,
            label=f"control windows (n = {len(neg)})")
    ax.hist(pos, bins=bins, density=True, histtype="step", lw=1.6, color=ORANGE,
            label=f"15 s before an apnea (n = {len(pos)})")
    auc = roc_auc_score(lab, -mean)   # declining direction: less effort = apnea
    ax.axvline(0, color=MUTE, lw=1)
    ax.set_xlabel("channel, averaged over the 30 s window")
    ax.set_ylabel("density")
    ax.set_title(f"c   Why it did not — single-feature AuROC {auc:.3f}",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return auc


def panel_criterion(ax, values):
    lab = values[:, 0]
    out = []
    for j, name in ((3, "below 20 %\n(B4's rule)"), (4, "below 50 %\n(Adams et al.)")):
        out.append((name,
                    100 * values[lab == 1, j].mean(),
                    100 * values[lab == 0, j].mean()))
    x = np.arange(len(out))
    w = 0.36
    ax.bar(x - w / 2, [o[1] for o in out], w, color=ORANGE, label="15 s before an apnea")
    ax.bar(x + w / 2, [o[2] for o in out], w, color=BLUE, label="control window")
    for xi, o in zip(x, out):
        ax.text(xi - w / 2, o[1] + 0.5, f"{o[1]:.1f} %", ha="center", va="bottom",
                fontsize=7, color=INK)
        ax.text(xi + w / 2, o[2] + 0.5, f"{o[2]:.1f} %", ha="center", va="bottom",
                fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([o[0] for o in out], fontsize=7.5, color=INK2, linespacing=1.4)
    ax.set_ylabel("share of the 30 s window\nspent below the threshold (%)")
    ax.set_title("d   The criterion, priced", loc="left", fontweight="bold",
                 fontsize=8.5)
    # Headroom for the value labels, then the legend over the shorter pair --
    # the taller bars sit on the right, so upper-left is the free corner.
    ax.set_ylim(0, max(max(o[1], o[2]) for o in out) * 1.45)
    ax.legend(loc="upper left", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return out


def make_figure(out_base):
    d = np.load(CACHE, allow_pickle=True)
    values = d["values"]
    tr = dict(np.load(TRACE_CACHE, allow_pickle=True)) if os.path.exists(TRACE_CACHE) else None

    # Two full-width rows for the trace (envelope over channel, shared time
    # axis), then the three argument panels side by side underneath.
    fig = plt.figure(figsize=(12.6, 8.0))
    # Nested, because the two trace rows must sit flush against each other on a
    # shared time axis while the argument panels below need a wide gutter. One
    # gridspec cannot give two different hspaces.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.36,
                             left=0.070, right=0.905, top=0.935, bottom=0.085)
    top_gs = outer[0].subgridspec(2, 1, hspace=0.10)
    bot_gs = outer[1].subgridspec(1, 3, wspace=0.34)
    if tr is not None:
        panel_trace(fig.add_subplot(top_gs[0]), fig.add_subplot(top_gs[1]), tr)
    panel_geometry(fig.add_subplot(bot_gs[0]))
    auc = panel_distribution(fig.add_subplot(bot_gs[1]), values)
    fired = panel_criterion(fig.add_subplot(bot_gs[2]), values)

    print(f"single-feature AuROC (window mean): {auc:.4f}")
    for name, p, n in fired:
        print(f"  {name.replace(chr(10), ' '):28s} pre-apnea {p:5.1f} %   control {n:5.1f} %")

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_base}.png / .pdf")


def main():
    args = parse_args()
    if not args.replot:
        build_cache(args)
    make_figure(os.path.join(REPO, args.out))


if __name__ == "__main__":
    main()
