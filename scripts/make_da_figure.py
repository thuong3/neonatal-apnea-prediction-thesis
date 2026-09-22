"""Figure for the domain-adaptation experiment (Section 4.5 of the thesis).

Two panels, one message:

  A  Where the performance is lost. Each row differs from the one above it in
     exactly one respect, so each gap is separately attributable. Colour marks
     the cohort -- the variable that turns out to matter.
  B  No adaptation method beats training on target data alone. Paired
     per-patient differences against the `target_only` control, with 95 % CI.

Panel B's error bars are computed from the PAIRED per-patient differences
(same patient, adapted vs control), not from the spread of the two means --
the patients are the same in both arms, so the paired spread is the honest
uncertainty and is much tighter than the marginal SDs suggest.

Usage:
    python scripts/make_da_figure.py --result_path results/domain_adaptation
"""

import argparse
import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rc_file(os.path.join(REPO, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from da_summary import collect, load_runs, seed_averaged  # noqa: E402

# Reference palette slots 1 and 2 (see the dataviz palette reference): they
# validate all-pairs in both modes. Colour encodes the COHORT, because that is
# the variable the experiment isolates.
C_SOURCE = "#2a78d6"  # blue  -- Robin (source)
C_TARGET = "#eb6834"  # orange -- CPAP (target)
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"

LADDER_LABELS = {
    "zero_shot": "zero-shot",
    "pooled": "pooled (source + target)",
    "finetune_full": "fine-tuning (all weights)",
    "finetune_head": "fine-tuning (head only)",
    "adabn": "AdaBN",
    "coral": "CORAL",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--result_path", required=True)
    p.add_argument("--out", default="figures/03_results/domain_adaptation.png")
    p.add_argument("--control", default="target_only")
    p.add_argument("--diag_5ch", default="diag_5ch")
    p.add_argument("--diag_6ch", default="diag_6ch")
    p.add_argument(
        "--paper_auc",
        type=float,
        default=0.780,
        help="Reproduced published result (6ch, apnea+hypopnea). "
        "Pass a negative value to omit that row.",
    )
    return p.parse_args()


def mean_of(averaged, name):
    """Mean over patients of the seed-averaged per-patient AuROCs."""
    if name not in averaged:
        return None
    return float(np.nanmean(list(averaged[name].values())))


def paired_delta(averaged, variant, control):
    """Per-patient (variant - control) differences, mean, 95% CI and p."""
    shared = sorted(set(averaged[variant]) & set(averaged[control]))
    diffs = np.array(
        [averaged[variant][p] - averaged[control][p] for p in shared], dtype=float
    )
    diffs = diffs[~np.isnan(diffs)]
    mean = float(np.mean(diffs))
    # SE of the paired differences; 1.96 SE is a normal-approximation 95 % CI,
    # adequate at n = 15 for a figure (the reported test is Wilcoxon).
    half = 1.96 * float(np.std(diffs, ddof=1)) / np.sqrt(len(diffs))
    try:
        _, pval = wilcoxon(diffs)
    except ValueError:
        pval = float("nan")
    return mean, half, pval, len(diffs)


def panel_a(ax, averaged_ladder, averaged_diag, args):
    rows = []
    if args.paper_auc >= 0:
        rows.append(
            ("Robin · 6 channels · apnea + hypopnea", args.paper_auc, C_SOURCE, True)
        )
    for key, label in (
        (args.diag_6ch, "Robin · 6 channels · apnea only"),
        (args.diag_5ch, "Robin · 5 shared channels · apnea only"),
    ):
        val = mean_of(averaged_diag, key)
        if val is not None:
            rows.append((label, val, C_SOURCE, False))
    for key, label in (
        ("zero_shot", "CPAP · Robin model unchanged"),
        (args.control, "CPAP · trained on CPAP"),
    ):
        val = mean_of(averaged_ladder, key)
        if val is not None:
            rows.append((label, val, C_TARGET, False))

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ypos = np.arange(len(rows))[::-1]  # first row at the top

    # Bars are anchored at CHANCE (0.5), not at zero or at the axis minimum.
    # An AuROC bar drawn from an arbitrary cut would let length misrepresent
    # value; measured from chance, length is the meaningful quantity ("how far
    # above chance") and the sign flip at zero-shot becomes visible directly.
    chance = 0.5
    for y, (_lab, val, colour, is_ref) in zip(ypos, rows):
        ax.barh(
            y,
            val - chance,
            left=chance,
            height=0.62,
            color="none" if is_ref else colour,
            edgecolor=colour,
            linewidth=1.5,
            # The published row was not re-measured here, so it is drawn hollow
            # rather than presented as one of this experiment's own results.
            hatch="////" if is_ref else None,
        )
        offset = 0.006 if val >= chance else -0.006
        ax.text(val + offset, y, f"{val:.3f}", va="center",
                ha="left" if val >= chance else "right",
                fontsize=8, color=C_INK)

    # The chance baseline is an axis rule, so it is solid ink, not a dashed
    # gridline. It is named in the tick label rather than by floating text,
    # which cannot collide with the title.
    ax.axvline(chance, color=C_INK, linewidth=1.5, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.40, 0.86)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(["0.4", "0.5\nchance", "0.6", "0.7", "0.8"])
    ax.set_xlabel("AuROC (mean over patients)")
    ax.set_title("A  Where the performance is lost", loc="left", fontweight="bold")
    ax.xaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_SOURCE),
        plt.Rectangle((0, 0), 1, 1, color=C_TARGET),
    ]
    ax.legend(handles, ["Robin (source)", "CPAP (target)"],
              loc="lower right", fontsize=8)


def panel_b(ax, averaged, control):
    variants = [v for v in LADDER_LABELS if v in averaged]
    stats = [paired_delta(averaged, v, control) for v in variants]
    ypos = np.arange(len(variants))[::-1]

    for y, v, (mean, half, pval, _n) in zip(ypos, variants, stats):
        # Nothing reaches significance, so nothing is emphasised: colouring a
        # non-significant point would imply an effect the test does not support.
        ax.errorbar(mean, y, xerr=half, fmt="o", color=C_INK2,
                    ecolor=C_MUTED, elinewidth=1.5, capsize=3, markersize=5)
        # p-values live outside the plot area (axes-fraction x), so they cannot
        # collide with a wide confidence interval.
        ax.text(1.03, y, f"{pval:.2f}", va="center", ha="left", fontsize=8,
                color=C_MUTED, transform=ax.get_yaxis_transform(),
                clip_on=False)
    ax.text(1.03, len(variants) - 0.4, "p", va="bottom", ha="left", fontsize=8,
            color=C_MUTED, transform=ax.get_yaxis_transform(), clip_on=False,
            fontstyle="italic")

    ax.axvline(0, color=C_INK, linewidth=1.5, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LADDER_LABELS[v] for v in variants], fontsize=8)
    ax.set_xlim(-0.16, 0.16)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["-0.15", "-0.10", "-0.05", "0\nno effect",
                        "0.05", "0.10", "0.15"])
    ax.set_xlabel(f"Δ AuROC vs. “{control}”  (paired, 95 % CI)")
    ax.set_title("B  No adaptation method beats the control",
                 loc="left", fontweight="bold")
    ax.xaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def main():
    args = parse_args()
    runs = load_runs(args.result_path)
    averaged_ladder = seed_averaged(collect(runs, "results"))
    averaged_diag = seed_averaged(collect(runs, "source_internal"))

    if args.control not in averaged_ladder:
        raise SystemExit(
            f"No '{args.control}' rows found in {args.result_path}. "
            f"Available: {sorted(averaged_ladder)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    panel_a(axes[0], averaged_ladder, averaged_diag, args)
    panel_b(axes[1], averaged_ladder, args.control)
    fig.tight_layout(w_pad=3.0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    # A vector copy for the thesis; the PNG is for slides and quick viewing.
    pdf_out = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"Wrote {args.out} and {pdf_out}")


if __name__ == "__main__":
    main()
