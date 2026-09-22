"""THESIS FIGURE: what hyperparameter tuning did, including dropout (F2).

Two panels, because the question was asked twice and answered twice.

  (a) The nested run (31 Aug 2026, seed 0, `cpap_hp`). Three configurations
      were offered to an inner validation loop on every one of the 15 outer
      folds. Dropout was one of them. Each bar averages only the folds where
      selection picked that configuration -- 8, 4 and 3 of them -- so the three
      are NOT paired and the panel is descriptive. The two rules are the
      numbers that estimate the pipeline: nested selection returned 0.534
      against the untuned 0.555.

  (b) The paired re-run (1 Sep 2026, `cpap_capacity`). `base` and `wide_long`
      as FIXED arms over all 15 infants at 10 seeds, which is what makes a
      signed-rank test defined at all. Per-infant test AuROC, capacity arm
      minus published arm, against the seed-to-seed spread of that same
      contrast. Under PRE_SPECIFICATION.md 1.1 an effect must clear both the
      Wilcoxon and its own spread; this one clears the spread and misses the
      Wilcoxon, so it is not called.

The two panels carry the whole finding: the capacity arm buys 0.188 of training
fit and returns -0.024 on the held-out infant, and dropout changed nothing.

NUMBER PROVENANCE.
  panel (a)  Section 4.3 of the thesis, hard-coded
             below. One seed, and the fold counts are part of the reading.
  panel (b)  parsed from outputs/capacity/capacity_stats_10seeds.txt, which is
             the saved printout of scripts/capacity_stats.py. The result
             pickles it came from live on the VM only, so that file is the
             record -- the figure is built from it rather than from a second
             copy of the numbers that could drift away from it.

Usage:
    python scripts/make_hp_tuning_figure.py [--out_dir figures/03_results]
"""
import argparse
import os
import re

import matplotlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rc_file(os.path.join(REPO, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

# Same palette and ink as make_thesis_figures.py, since this figure sits in the
# same results set. Slots 3-5 are below 3:1 on white, so every bar is labelled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTE = "#898781"
GRID = "#e1e0d9"

CHANCE = 0.5
STATS = os.path.join(REPO, "outputs", "capacity", "capacity_stats_10seeds.txt")

# --- panel (a): the nested run, seed 0 -----------------------------------
# label, folds selection gave it, train, test, colour
CONFIGS = [
    ("published\n20 hidden, 10 epochs", 8, 0.655, 0.565, BLUE),
    ("+ dropout 0.25\n40 epochs", 4, 0.673, 0.555, AQUA),
    ("40 hidden\n40 epochs", 3, 0.871, 0.424, ORANGE),
]
NESTED_TUNED = 0.534      # outer mean of the tuned pipeline
UNTUNED = 0.555           # the same pipeline, no selection


def _read_per_infant(path):
    """Block [3] of the capacity_stats.py printout: id, base, wide_long, delta."""
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found. It is the saved printout of\n"
            "  python scripts/capacity_stats.py --seeds 0 1 2 3 4 5 6 7 8 9 "
            "--logs outputs/\n"
            "run on the VM, where the cpap_capacity pickles live.")
    row = re.compile(
        r"^\s{2}(\d{3})\s+([\d.]+)\s+([\d.]+)\s+([-+][\d.]+)\s+(\d+)\s*$")
    ids, base, wide, delta = [], [], [], []
    for line in open(path, encoding="utf-8"):
        m = row.match(line.rstrip("\n"))
        if m:
            ids.append(m.group(1))
            base.append(float(m.group(2)))
            wide.append(float(m.group(3)))
            delta.append(float(m.group(4)))
    if len(ids) != 15:
        raise SystemExit(f"parsed {len(ids)} infants from {path}, expected 15")
    return ids, np.array(base), np.array(wide), np.array(delta)


def _read_scalar(path, pattern):
    text = open(path, encoding="utf-8").read()
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"could not find {pattern!r} in {path}")
    return float(m.group(1))


def panel_a(ax):
    # Wider bars than the default 0.34, to offset the left margin the
    # three rule labels need.
    w = 0.38
    for xi, (_lab, folds, train, test, colour) in enumerate(CONFIGS):
        ax.bar(xi - w / 2, train - CHANCE, w, bottom=CHANCE, color="none",
               edgecolor=colour, linewidth=1.5, hatch="////",
               label="train (windows the model learned from)" if xi == 0 else None)
        ax.bar(xi + w / 2, test - CHANCE, w, bottom=CHANCE, color=colour,
               edgecolor=colour, linewidth=1.5,
               label="test (held-out infant)" if xi == 0 else None)
        for off, v in ((-w / 2, train), (w / 2, test)):
            up = v >= CHANCE
            ax.text(xi + off, v + (0.010 if up else -0.010), f"{v:.3f}",
                    ha="center", va="bottom" if up else "top", fontsize=7.5,
                    color=colour)
        # Below the deepest bar (0.424) so the value label has the space above.
        ax.text(xi, 0.398, f"{folds} of 15 folds", ha="center",
                va="top", fontsize=7, color=MUTE)

    ax.axhline(CHANCE, color=INK, linewidth=1.5, zorder=3)
    # Left edge: the right edge belongs to the two pipeline rules.
    ax.text(-1.40, 0.496, "chance", ha="left", va="top", fontsize=6.5,
            color=MUTE)

    # The two numbers that estimate the PIPELINE rather than a configuration.
    ax.axhline(UNTUNED, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1,
               zorder=2)
    # Left margin, not the right: right-aligned at 2.60 these two ran straight
    # through the hatching of the 40-hidden train bar.
    ax.text(-1.40, UNTUNED + 0.003, f"untuned {UNTUNED:.3f}",
            ha="left", va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(NESTED_TUNED, color=INK2, linestyle=(0, (4, 3)), linewidth=1.1,
               zorder=2)
    ax.text(-1.40, NESTED_TUNED - 0.003,
            f"nested-tuned {NESTED_TUNED:.3f}",
            ha="left", va="top", fontsize=6.5, color=INK2)

    # The specific request from the 31 Aug meeting, and its answer.
    ax.annotate("dropout, offered on every fold",
                xy=(1 - w / 2, 0.673 + 0.028), xytext=(0.72, 0.845),
                ha="center", fontsize=7, color=AQUA,
                arrowprops=dict(arrowstyle="->", color=AQUA, linewidth=1.1))

    ax.set_xticks(np.arange(len(CONFIGS)))
    ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=7.5, color=INK2,
                       linespacing=1.4)
    # Left margin wide enough for the chance / untuned / nested-tuned labels,
    # which have nowhere to sit inside the plot without crossing a bar.
    ax.set_xlim(-1.42, 2.62)
    ax.set_ylim(0.365, 0.93)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("AuROC")
    ax.set_title("a   Nested selection over three configurations, seed 0\n"
                 "     each bar averages only the folds that chose it, so the "
                 "three are not paired",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6)


def panel_b(ax, ids, delta, spread, mean_cw, pval, wstat, train_base,
            train_wide):
    order = np.argsort(delta)
    ids = [ids[i] for i in order]
    d = delta[order]

    ax.axhspan(-spread, spread, color=GRID, zorder=0)
    ax.text(-0.55, spread + 0.003,
            f"seed-to-seed spread of this contrast ({spread:.3f})",
            ha="left", va="bottom", fontsize=7, color=INK2)

    ax.bar(np.arange(len(d)), d, width=0.62,
           color=[BLUE if v >= 0 else ORANGE for v in d], zorder=2)
    ax.axhline(0, color=INK, linewidth=1.5, zorder=3)
    ax.axhline(mean_cw, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=3)
    ax.text(len(d) - 0.4, mean_cw - 0.005, f"mean {mean_cw:+.3f}", ha="right",
            va="top", fontsize=7, color=INK2)

    worse = int((d < 0).sum())
    ax.text(-0.55, min(d) - 0.014,
            f"worse with capacity in {worse} of {len(d)}. "
            f"W = {wstat:.1f}, p = {pval:.4f}, so the loss is not called.\n"
            f"On the training data the same arms go {train_base:.3f} to "
            f"{train_wide:.3f}, a gain of {train_wide - train_base:+.3f}.",
            ha="left", va="top", fontsize=7, color=INK2, linespacing=1.5,
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(ids, fontsize=7, color=INK2)
    ax.set_xlim(-0.6, len(d) - 0.4)
    ax.set_ylim(min(d) - 0.062, max(max(d), spread) + 0.030)
    ax.set_xlabel("infant")
    ax.set_ylabel("test AuROC, 40x40 minus 20x10")
    ax.set_title("b   The same capacity as a fixed arm on all 15 infants, "
                 "10 seeds\n"
                 "     identical windows and labels, so the differences are "
                 "paired",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures/03_results")
    ap.add_argument("--stats", default=STATS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids, _base, _wide, delta = _read_per_infant(args.stats)
    spread = _read_scalar(args.stats,
                          r"spread of this contrast \(n=\d+\): ([\d.]+)")
    mean_cw = _read_scalar(args.stats, r"count-weighted delta ([-+][\d.]+)")
    wstat = _read_scalar(args.stats, r"W=([\d.]+)")
    pval = _read_scalar(args.stats, r"p=([\d.]+)")
    train_base = _read_scalar(args.stats, r"base\s+mean train AuROC ([\d.]+)")
    train_wide = _read_scalar(args.stats,
                              r"wide_long\s+mean train AuROC ([\d.]+)")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.16,
                        wspace=0.26)
    panel_a(axes[0])
    panel_b(axes[1], ids, delta, spread, mean_cw, pval, wstat,
            train_base, train_wide)

    for ext in ("png", "pdf"):
        path = os.path.join(args.out_dir, f"hp_tuning.{ext}")
        fig.savefig(path, dpi=300)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
