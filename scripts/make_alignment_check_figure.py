"""THE alignment check: the same recording, early / middle / late in the night,
drawn without and with the annotation-clock drift correction.

This is the figure that settles "do our overlays sit where RemLogic says".
A fixed offset cannot be checked by looking at one event -- any constant can be
made to fit a single mark. What identifies a DRIFT is that the error GROWS with
time into the recording, so the test has to compare early against late. That is
the whole layout: three rows down the night, uncorrected on the left, corrected
on the right.

Left column, top to bottom: the mark starts on the feature and walks off it.
Right column: it stays put. If the correction were wrong in slope, the right
column would still walk; if it were wrong by a constant, the whole right column
would sit off by the same amount in every row.

DESAT against SpO2 is the default because it needs no interpretation: a
desaturation mark either sits on the falling edge or it does not, and a
supervisor can check it without knowing anything about the model. `--event_type
APNEA-CENTRAL --channel ESO` gives the respiratory-effort version instead (needs
dataset_v2, which is the export that carries the esophageal catheter).

Usage:
    python scripts/make_alignment_check_figure.py --pat_id 011
    python scripts/make_alignment_check_figure.py --pat_id 015 \
        --event_type APNEA-CENTRAL --channel ESO \
        --data_path ".../dataset_v2"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyedflib  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import load_clock_drift  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_BAD = "#c0392b"
C_GOOD = "#1e8449"
C_TRACE = "#2c3e50"
C_TEXT = "#52514e"

# EDF labels per logical channel, in preference order (the two exports of this
# cohort label the same sensors differently).
# "Belts" is a pair drawn in one panel: it is what the annotator actually read,
# and the rule was that BOTH had to be flat, which only a paired view shows.
CHANNEL_LABELS = {
    "Belts": ["Thorax", "Abdomen"],
    "SpO2": ["SpO2 Radical", "SpO2"],
    "ESO": ["?sophagusdruck", "Oesophagusdruck"],
    "Thorax": ["Thorax"],
    "Abdomen": ["Abdomen"],
    "CPAP": ["CPAP Pressure"],
}
CONTEXT_S = 45.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity")
    p.add_argument("--pat_id", default="011")
    # Defaults to the annotator's own criterion: an APNEA mark, shown against
    # the two belts they scored it from. `--event_type DESAT --channel SpO2`
    # gives the independent cross-check on the oximeter instead.
    p.add_argument("--event_type", default="APNEA-CENTRAL")
    p.add_argument("--channel", default="Belts", choices=sorted(CHANNEL_LABELS))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "figures/05_signal_checks"))
    p.add_argument("--out_name", default=None)
    return p.parse_args()


def read_channel(pat_id, data_path, channel):
    """(t0, [traces], fs). `Belts` returns both traces, everything else one."""
    edf = pyedflib.EdfReader(
        os.path.join(data_path, "signals", f"signals{pat_id}.edf"))
    try:
        labels = edf.getSignalLabels()
        wanted = CHANNEL_LABELS[channel]
        if channel == "Belts":
            missing = [w for w in wanted if w not in labels]
            if missing:
                raise SystemExit(f"patient {pat_id} is missing {missing}")
            idx = [labels.index(w) for w in wanted]
            return (pd.Timestamp(edf.getStartdatetime()),
                    [edf.readSignal(i) for i in idx],
                    float(edf.getSampleFrequency(idx[0])))
        for want in wanted:
            if want in labels:
                i = labels.index(want)
                return (pd.Timestamp(edf.getStartdatetime()),
                        [edf.readSignal(i)], float(edf.getSampleFrequency(i)))
        raise SystemExit(
            f"patient {pat_id} has no {channel} channel "
            f"(looked for {wanted}, file has {labels})")
    finally:
        edf._close()


ISOLATION_S = 40.0


def pick_events(anno, event_type, n_samples, fs):
    """One ISOLATED event early, one in the middle, one late.

    Isolation is the whole trick. These infants spend long stretches in periodic
    breathing, desaturating every ~12 s; inside such a stretch a mark placed at
    the right time and a mark placed one cycle out look equally correct, so the
    panel proves nothing either way. (The drift itself is not an artefact of
    that periodicity -- re-measuring on isolated marks only reproduces the same
    slope, 117-240 ppm across recordings -- but a FIGURE has to be readable by
    eye, and a lone desaturation with flat SpO2 on both sides is.)

    Percentiles rather than the literal first and last, so that every chosen
    event still has CONTEXT_S of signal on both sides to draw.
    """
    ev = anno[anno["type"] == event_type].sort_values("s")
    ev = ev[(ev["s"] > CONTEXT_S + 60) & (ev["s"] < n_samples / fs - CONTEXT_S - 60)]
    if len(ev) < 3:
        raise SystemExit(f"only {len(ev)} usable {event_type} marks in this recording")

    t = ev["s"].to_numpy()
    gap_before = np.diff(t, prepend=-np.inf)
    gap_after = np.diff(t, append=np.inf)
    lone = ev[(gap_before > ISOLATION_S) & (gap_after > ISOLATION_S)]
    if len(lone) >= 3:
        ev = lone
    else:
        print(f"WARNING: only {len(lone)} isolated {event_type} marks; falling back "
              "to all of them, so the panels may sit inside periodic breathing "
              "and be ambiguous to read.")
    idx = [int(q * (len(ev) - 1)) for q in (0.10, 0.50, 0.90)]
    return ev.iloc[idx]


def main():
    args = parse_args()
    drift = load_clock_drift()
    params = drift.get(args.pat_id, {})
    slope = params.get("slope_ppm", 0.0)
    shift = params.get("intercept_s", 0.0)
    scale = 1.0 + slope * 1e-6

    t0, traces, fs = read_channel(args.pat_id, args.data_path, args.channel)
    anno = pd.read_csv(
        os.path.join(args.data_path, "annotations", f"annotations{args.pat_id}.txt"),
        sep="\t")
    anno["s"] = (pd.to_datetime(anno["timestamp"]) - t0).dt.total_seconds()
    events = pick_events(anno, args.event_type, len(traces[0]), fs)

    fig, axes = plt.subplots(3, 2, figsize=(9.6, 6.8), sharex=True)
    for r, ev in enumerate(events.itertuples(index=False)):
        # Uncorrected: the mark is read at face value. Corrected: it is placed
        # where this recording's measured clock drift says it belongs. The
        # SIGNAL is identical in both columns -- only the mark moves.
        positions = [("uncorrected", ev.s, C_BAD),
                     ("drift-corrected", ev.s * scale + shift, C_GOOD)]
        for c, (title, pos, colour) in enumerate(positions):
            ax = axes[r, c]
            i0 = int((pos - CONTEXT_S) * fs)
            i1 = int((pos + CONTEXT_S) * fs)
            i0, i1 = max(0, i0), min(len(traces[0]), i1)
            tt = np.arange(i0, i1) / fs - pos
            if len(traces) == 1:
                ax.plot(tt, traces[0][i0:i1], lw=0.9, color=C_TRACE, zorder=3)
            else:
                # Two belts, each in its own lane and scaled by its own range:
                # the raw gains differ, and a shared axis would flatten one
                # against the other and hide exactly the "both flat" condition.
                blend = matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes)
                for k, (tr, nm) in enumerate(zip(traces, CHANNEL_LABELS[args.channel])):
                    seg = np.asarray(tr[i0:i1], dtype=float)
                    lane = 0.72 - 0.42 * k
                    span = np.nanmax(np.abs(seg - np.nanmean(seg))) or 1.0
                    ax.plot(tt, lane + 0.20 * (seg - np.nanmean(seg)) / span,
                            lw=0.9, color=C_TRACE if k == 0 else "#7f8c8d",
                            transform=blend, zorder=3)
                    ax.text(-CONTEXT_S + 1, lane + 0.21, nm, fontsize=6.5,
                            color=C_TRACE if k == 0 else "#7f8c8d",
                            transform=blend, va="bottom")
                ax.set_ylim(0, 1)
                ax.set_yticks([])
            ax.axvspan(0.0, float(ev.duration), color=colour, alpha=0.22, zorder=1)
            ax.axvline(0.0, color=colour, lw=1.4, zorder=4)
            ax.set_xlim(-CONTEXT_S, CONTEXT_S)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=10, color=colour,
                             fontweight="bold")
            if c == 0:
                hours = ev.s / 3600.0
                ax.set_ylabel(f"{hours:.1f} h\n{args.channel}", fontsize=8.5)
            applied = pos - ev.s
            ax.text(0.985, 0.06, f"Drift {applied:+.1f} s" if c else "Drift 0 s",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=colour)
    for ax in axes[-1]:
        ax.set_xlabel(f"seconds relative to the start of the {args.event_type} mark")
    if args.channel == "Belts":
        fig.text(0.012, 0.028,
                 "The scorer's criterion: Thorax AND Abdomen flat at the same "
                 "time for at least 10 s, so the mark has to sit exactly on the "
                 "stretch where both belts are quiet.",
                 ha="left", va="top", fontsize=7.5, color=C_TEXT)

    # Not every recording drifts. Patient 009 measures 2.5 ppm -- its clock is
    # simply accurate -- and asserting a drift over its panels would be a
    # caption contradicting its own figure.
    total = slope * 1e-6 * 24 * 3600
    if abs(total) < 2.0:
        headline = (f"Patient {args.pat_id}: **no** appreciable clock drift "
                    f"({slope:.0f} ppm) — the annotation sits correctly through "
                    "the whole recording")
    else:
        headline = (f"Patient {args.pat_id}: the annotation drifts away from the "
                    f"signal over the recording ({slope:.0f} ppm, {total:.0f} s/24 h)")
    fig.suptitle(headline.replace("**", ""),
                 x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold")
    fig.text(
        0.012, 0.935,
        f"Three {args.event_type} marks from the same recording: early, middle, late. "
        f"On the left as they stand in the annotation file, on the right\n"
        f"corrected by the measured clock drift. On the left the mark slides further off "
        f"the event as the recording goes on, which is exactly what no fixed\n"
        f"offset can repair. On the right it sits at the same place in the signal in all three rows.",
        ha="left", va="top", fontsize=8, color=C_TEXT)

    bottom = 0.075 if args.channel == "Belts" else 0.0
    fig.tight_layout(rect=[0, bottom, 1, 0.895])
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.out_name or f"alignment_check_pat{args.pat_id}_{args.event_type}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"{name}.{ext}"), dpi=150)
    plt.close(fig)

    print(f"patient {args.pat_id}: {slope:.1f} ppm, intercept {shift:+.2f} s")
    for ev in events.itertuples(index=False):
        print(f"  {args.event_type} at {ev.s / 3600.0:5.1f} h -> "
              f"drift applied {ev.s * scale + shift - ev.s:+6.2f} s")
    print(f"written: {os.path.join(args.out_dir, name)}.pdf/.png")


if __name__ == "__main__":
    main()
