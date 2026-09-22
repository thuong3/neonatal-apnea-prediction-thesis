"""Quantify the GAPS in the CPAP cohort: recorded time that carries no usable
label, either because it lies outside an ANALYSIS-START/STOP block (never
scored) or because it is marked SIGNAL-ARTIFACT (scored as unusable).

Why this exists: both kinds of gap are dropped before training -- the
un-analysed stretches never enter `read_data` as a segment, the artifact
stretches are the `cutter_events` of config/dataset/neonatal.yaml -- so they
directly shrink the usable data.

The EDF signals run CONTINUOUSLY through every gap (spot checks of SpO2 /
Thorax / CPAP pressure / heart rate inside the un-analysed stretches show
ordinary, non-flat data). The gaps are gaps in the ANNOTATION, not holes in
the recording.

WHAT THE ANNOTATOR EXPLAINED (personal communication, Aug 2026)
    Every infant was ventilated with FOUR different CPAP devices in turn
    (Stephanie with and without frequency, Infant Flow, Infant Flow with
    water seal), each for ~6 h, giving one 24 h measurement per infant. The
    four ANALYSIS blocks this script finds are exactly those four device
    phases; the gaps between them are the DEVICE CHANGEOVERS. Scoring was
    originally stored as four separate per-device files -- the export used
    here has all four merged into one annotation file per infant, which is
    why all four blocks are present.
    Primary endpoint of the original study: central apnoeas counted from
    thorax/abdomen movement. SpO2 and heart rate were recorded but did not
    enter the primary endpoint.
    Still undocumented: what SIGNAL-ARTIFACT stands for in each case.

Usage:
    python scripts/make_gap_histogram.py \
        --data_path "data/brainimmaturity" \
        [--out_dir figures/01_data_quality]
"""
import argparse
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pyedflib import highlevel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rc_file(os.path.join(ROOT, "matplotlibrc"))

TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"

# Colour by the role of the span, not by patient: the two gap kinds are the
# question this figure poses, the analysed remainder is context.
C_UNSCORED = "#2a78d6"  # outside any ANALYSIS block -> device changeover
C_ARTIFACT = "#eb6834"  # SIGNAL-ARTIFACT inside an analysed block
C_USABLE = "#c9c8c3"    # analysed and artifact-free -> the data we train on
C_TEXT = "#52514e"

# Leading/trailing remainders below this are the ordinary ragged edge of a
# recording (the scorer stops a few seconds before the file ends), not a gap
# anybody needs to explain. Reported separately rather than silently dropped.
EDGE_TOL_S = 60.0

BEFORE, BETWEEN, AFTER = (
    "before first block (set-up)",
    "between blocks (device change)",
    "after last block",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        default="data/brainimmaturity",
        help="folder containing signals/ and annotations/",
    )
    p.add_argument("--out_dir", default=os.path.join(ROOT, "figures/01_data_quality"))
    p.add_argument(
        "--ids",
        nargs="*",
        default=[f"{i:03d}" for i in range(1, 16)],
    )
    return p.parse_args()


def merge_intervals(intervals):
    """Union of (start, end) pairs -- artifact spans may overlap each other."""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap(iv, blocks):
    """Total length of `iv` that falls inside any of `blocks`."""
    return sum(max(0.0, min(b, iv[1]) - max(a, iv[0])) for a, b in blocks)


def read_one(pat_id, data_path):
    """Analysis blocks, artifact spans and recording length of one patient.

    All times are seconds relative to the EDF start; `t0` keeps the wall-clock
    origin so the table can report real timestamps.
    """
    ann = pd.read_csv(
        os.path.join(data_path, "annotations", f"annotations{pat_id}.txt"), sep="\t"
    )
    hdr = highlevel.read_edf_header(
        os.path.join(data_path, "signals", f"signals{pat_id}.edf"),
        read_annotations=False,
    )
    t0 = hdr["startdate"]
    rec_len = float(hdr["Duration"])

    rel = np.array(
        [
            (datetime.strptime(ts, TS_FMT) - t0).total_seconds()
            for ts in ann["timestamp"]
        ]
    )
    ann = ann.assign(t=rel)

    blocks, current = [], None
    for row in ann.itertuples(index=False):
        if row.type == "ANALYSIS-START":
            current = row.t
        elif row.type == "ANALYSIS-STOP":
            if current is None:
                raise ValueError(f"{pat_id}: ANALYSIS-STOP without START")
            blocks.append((current, row.t + row.duration))
            current = None
    if current is not None:
        raise ValueError(f"{pat_id}: ANALYSIS-START without STOP")
    blocks = merge_intervals(blocks)

    artifacts = merge_intervals(
        [
            (row.t, row.t + row.duration)
            for row in ann.itertuples(index=False)
            if row.type == "SIGNAL-ARTIFACT"
        ]
    )
    return dict(
        pat_id=pat_id, t0=t0, rec_len=rec_len, blocks=blocks, artifacts=artifacts
    )


def gaps_of(rec):
    """Every stretch of recording that lies outside the ANALYSIS blocks."""
    edges = [(0.0, rec["blocks"][0][0], BEFORE)]
    for i in range(len(rec["blocks"]) - 1):
        edges.append((rec["blocks"][i][1], rec["blocks"][i + 1][0], BETWEEN))
    edges.append((rec["blocks"][-1][1], rec["rec_len"], AFTER))
    return [(a, b, kind) for a, b, kind in edges if b > a]


def collect(ids, data_path):
    recs = [read_one(p, data_path) for p in ids]
    rows = []
    for rec in recs:
        for a, b, kind in gaps_of(rec):
            rows.append(
                dict(
                    patient=rec["pat_id"],
                    kind="not analysed",
                    position=kind,
                    start_clock=(rec["t0"] + pd.Timedelta(seconds=a)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    end_clock=(rec["t0"] + pd.Timedelta(seconds=b)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    start_h_into_recording=round(a / 3600, 3),
                    duration_min=round((b - a) / 60, 2),
                )
            )
        for a, b in rec["artifacts"]:
            inside = overlap((a, b), rec["blocks"])
            rows.append(
                dict(
                    patient=rec["pat_id"],
                    kind="SIGNAL-ARTIFACT",
                    position=(
                        "inside analysis block"
                        if inside > 0.5 * (b - a)
                        else "outside analysis block"
                    ),
                    start_clock=(rec["t0"] + pd.Timedelta(seconds=a)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    end_clock=(rec["t0"] + pd.Timedelta(seconds=b)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    start_h_into_recording=round(a / 3600, 3),
                    duration_min=round((b - a) / 60, 2),
                )
            )
    return recs, pd.DataFrame(rows).sort_values(
        ["patient", "start_h_into_recording"]
    )


def budget(recs):
    """Per patient: recorded / usable / artifact / un-analysed hours."""
    rows = []
    for rec in recs:
        analysed = sum(b - a for a, b in rec["blocks"])
        art_in = sum(overlap(iv, rec["blocks"]) for iv in rec["artifacts"])
        rows.append(
            dict(
                patient=rec["pat_id"],
                recording_h=rec["rec_len"] / 3600,
                not_analysed_h=(rec["rec_len"] - analysed) / 3600,
                artifact_h=art_in / 3600,
                usable_h=(analysed - art_in) / 3600,
                n_blocks=len(rec["blocks"]),
                n_artifacts=len(rec["artifacts"]),
            )
        )
    df = pd.DataFrame(rows)
    df["usable_pct"] = 100 * df["usable_h"] / df["recording_h"]
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(recs, gaps, bud, out_dir):
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.5],
        hspace=0.34,
        wspace=0.26,
        top=0.885,
        bottom=0.11,
        left=0.09,
        right=0.98,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) the device changeovers -------------------------------------------
    # ONE series only: the 45 gaps between consecutive blocks (3 per infant).
    # The set-up period before the first block and the remainder after the
    # last one are a different thing and would only muddle the distribution;
    # they stay visible in panel (c) and are quantified in the printout.
    ung = gaps[gaps["kind"] == "not analysed"]
    changeover = ung.loc[ung["position"] == BETWEEN, "duration_min"].values
    bins = np.arange(0, 130, 10)
    ax_a.hist(changeover, bins=bins, color=C_UNSCORED, edgecolor="white", linewidth=0.8)
    ax_a.set_xlabel("Gap between two consecutive blocks [min]")
    ax_a.set_ylabel("Number of gaps")
    ax_a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax_a.set_title(
        f"(a) Gaps between the analysis blocks\n"
        f"n = {len(changeover)} · median {np.median(changeover):.0f} min · "
        f"total {changeover.sum() / 60:.0f} h",
        loc="left",
    )

    # (b) artifact spans ----------------------------------------------------
    art = gaps[gaps["kind"] == "SIGNAL-ARTIFACT"]
    d = art["duration_min"].values
    log_bins = np.logspace(np.log10(max(d.min(), 0.05)), np.log10(d.max() * 1.05), 22)
    ax_b.hist(d, bins=log_bins, color=C_ARTIFACT, edgecolor="white", linewidth=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xticks([0.1, 1, 10, 100])
    ax_b.set_xticklabels(["0.1", "1", "10", "100"])
    ax_b.set_xlabel("Duration of the SIGNAL-ARTIFACT segment [min, log]")
    ax_b.set_ylabel("Number of segments")
    ax_b.set_title(
        f"(b) Segments marked SIGNAL-ARTIFACT\n"
        f"n = {len(art)} · median {np.median(d):.1f} min · "
        f"longest {d.max():.0f} min · total {d.sum() / 60:.0f} h",
        loc="left",
    )

    # (c) where the gaps sit in each recording -----------------------------
    for y, rec in enumerate(recs):
        ax_c.barh(
            y, rec["rec_len"] / 3600, left=0, height=0.72, color=C_UNSCORED, zorder=1
        )
        for a, b in rec["blocks"]:
            ax_c.barh(
                y, (b - a) / 3600, left=a / 3600, height=0.72, color=C_USABLE, zorder=2
            )
        for a, b in rec["artifacts"]:
            for ba, bb in rec["blocks"]:  # only inside analysed stretches
                lo, hi = max(a, ba), min(b, bb)
                if hi > lo:
                    ax_c.barh(
                        y,
                        (hi - lo) / 3600,
                        left=lo / 3600,
                        height=0.72,
                        color=C_ARTIFACT,
                        zorder=3,
                    )
    ax_c.set_yticks(range(len(recs)))
    ax_c.set_yticklabels([r["pat_id"] for r in recs])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, max(r["rec_len"] for r in recs) / 3600 + 0.2)
    ax_c.set_xlabel("Time into the recording [h]")
    ax_c.set_ylabel("Infant")
    ax_c.set_title(
        "(c) Position of the blocks and gaps in each recording "
        f"(usable on average {bud['usable_pct'].mean():.0f} %, "
        f"minimum {bud['usable_pct'].min():.0f} % in infant "
        f"{bud.loc[bud['usable_pct'].idxmin(), 'patient']})",
        loc="left",
    )
    ax_c.legend(
        handles=[
            Patch(color=C_USABLE, label="analysed, artifact-free (used)"),
            Patch(color=C_ARTIFACT, label="SIGNAL-ARTIFACT (discarded)"),
            Patch(color=C_UNSCORED, label="not analysed (discarded)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        handlelength=1.2,
    )

    # No interpretation on the figure -- the covering e-mail carries that.
    total = bud["recording_h"].sum()
    fig.suptitle(
        f"Non-evaluable time in the CPAP cohort "
        f"({len(recs)} recordings, {total:.0f} h in total: "
        f"{100 * bud['not_analysed_h'].sum() / total:.1f} % not analysed, "
        f"{100 * bud['artifact_h'].sum() / total:.1f} % marked as artifact)",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out_dir, f"signal_gaps.{ext}"), format=ext, bbox_inches=None
        )
    plt.close(fig)


def main():
    args = parse_args()
    recs, gaps = collect(args.ids, args.data_path)
    bud = budget(recs)
    make_figure(recs, gaps, bud, args.out_dir)

    gaps.to_csv(os.path.join(args.out_dir, "signal_gaps_table.csv"), index=False)
    bud.round(2).to_csv(
        os.path.join(args.out_dir, "signal_gaps_summary.csv"), index=False
    )

    ung = gaps[
        (gaps["kind"] == "not analysed") & (gaps["duration_min"] * 60 >= EDGE_TOL_S)
    ]
    # Panel (a) plots only the changeovers, so spell out the other two kinds.
    by_pos = ung.groupby("position")["duration_min"].agg(["count", "median", "sum"])
    art = gaps[gaps["kind"] == "SIGNAL-ARTIFACT"]
    outside = gaps[gaps["position"] == "outside analysis block"]
    total = bud["recording_h"].sum()
    print(bud.round(2).to_string(index=False))
    print(
        f"\nrecorded total          {total:7.1f} h"
        f"\n  not analysed          {bud['not_analysed_h'].sum():7.1f} h "
        f"({100 * bud['not_analysed_h'].sum() / total:.1f} %) "
        f"in {len(ung)} gaps > {EDGE_TOL_S:.0f}s"
        f"\n  artifact (in block)   {bud['artifact_h'].sum():7.1f} h "
        f"({100 * bud['artifact_h'].sum() / total:.1f} %) "
        f"in {len(art)} segments"
        f"\n  usable                {bud['usable_h'].sum():7.1f} h "
        f"({100 * bud['usable_h'].sum() / total:.1f} %)"
    )
    print("\nun-analysed time by position (gaps > 60 s):")
    print(
        by_pos.assign(sum_h=(by_pos["sum"] / 60).round(1))
        .drop(columns="sum")
        .round(1)
        .to_string()
    )
    if len(outside):
        print(f"\nnote: {len(outside)} artifact segments lie outside the analysis blocks.")
    print(
        f"\nwritten to {args.out_dir}: signal_gaps.pdf/.png, "
        "signal_gaps_table.csv, signal_gaps_summary.csv"
    )


if __name__ == "__main__":
    main()
