"""THESIS / TALK FIGURES: the three summary figures the argument needs, in English.

Supersedes `make_summary_figures.py`, whose three figures are German and predate
the domain-adaptation runs, the effort baseline and the SpO2 work. Nothing here
is recomputed -- every value is the already-validated number from the methodology
documents, quoted with its source, exactly as `make_summary_figures.py` and
`make_underfitting_figure.py` do. If a number in those documents changes, change
it here too.

  task_map            every approach REPORTED in the thesis, grouped by the
                      task it addresses. The one figure that shows what the
                      project did and what each strand returned. Three approaches
                      were run but are deliberately NOT shown, all cut from
                      Chapter 4 in Sept 2026: the LSTM (0.529) on the
                      supervisor's advice as outdated; the MLP combiner (0.541)
                      as redundant against the capacity experiment; and the
                      45-feature gradient-boosting precursor model (0.559),
                      because the capacity experiment makes the same point more
                      rigorously and the provenance of the 45 features is not
                      defensible in a viva. Do not re-add any of them here
                      without re-adding them to the chapter.
                      NB the "event history + physiology" bar in the
                      apnea-prone-state group IS a gradient-boosting model;
                      only its use as a precursor model was cut.
  horizon_curves      (a) the detection/prediction collapse, train AND test,
                      (b) the apnea-prone state, which is the thing that is
                      predictable.
  horizon_simple      horizon_curves (a) again, for talking to: one curve, the
                      window drawn against a breathing trace, no annotations --
                      the presenter supplies the argument out loud.
  domain_adaptation   (a) where the performance is lost, one attributable step
                      per row, (b) no adaptation method beats the control,
                      (c) adapting to the individual infant does not either.
  norm_ablation       the normalisation 2x2 and the four closing sweeps
                      against their own seed spread.
  spo2_baseline_results
                      what the SpO2 baseline-drift idea actually predicts, and
                      how much of it survives a real gap before the target.
  spo2_simple         the drift alone against the two endpoints, for talking to.

`domain_adaptation` is also produced by `make_da_figure.py` directly from the
result pickles -- prefer that one when the pickles are reachable (they live on
the VM, not in this repo). This version reads Section 4.5 of the thesis SS10 instead,
so the figure can be rebuilt without them.

Usage: python scripts/make_thesis_figures.py [--out_dir figures/03_results]
"""
import argparse
import os

import matplotlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.rc_file(os.path.join(REPO, "matplotlibrc"))
import matplotlib.pyplot as plt  # noqa: E402

# --- palette -------------------------------------------------------------
# Slots 1-5 of the reference categorical palette, in its fixed order. That
# order is documented as passing the adjacent-pair gates in light mode, and a
# contiguous prefix inherits the property because its adjacent pairs are a
# subset. Slots 3-5 sit below 3:1 contrast on a white surface, so the relief
# rule applies: every bar carries a visible value label.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTE = "#898781"
GRID = "#e1e0d9"

CHANCE = 0.5


# =========================================================================
# Fig 1: the task map
# =========================================================================
# Mean test AuROC, leave-one-patient-out. Pooled AuROC is never quoted
# (Section 4.4 of the thesis SS33.6).
#
# NOT all count-weighted, despite what this comment used to claim. The NAM
# rows at 15 s come from the 10-seed capacity rerun and ARE count-weighted
# (SS22.2); detection at 0.870 is the PLAIN per-patient mean of the SS7 lag
# sweep, and the state and saturation groups are plain per-patient means from
# the feature pipeline of SS13/SS23, which never computed a count-weighted
# summary. The two summaries differ by ~0.005 where both exist, so the figure
# is readable as it stands, but do not describe the whole plot as the
# pre-specified primary statistic.
# EXCEPTION, measured Sep 2026 by scripts/spo2_operating_points.py: for the
# oxygenation endpoint the count-weighted mean is 0.770 against the 0.740 shown
# here, a gap of 0.030 rather than ~0.005. Section 4.7 quotes the 0.770 and
# names it as count-weighted, so the two numbers do not collide; the caption of
# Figure 4.1 no longer carries the caveat, because at the size the figure has to
# be to share the chapter opening page there is no room for it.
#
# (section title, section colour, [(row label, AuROC, is_reference), ...])
TASK_MAP = [
    (
        "Reference: the published cohort, Robin sequence",
        BLUE,
        [
            ("Vetter et al., as published", 0.800, True),      # SS22
            ("this pipeline, same data, unchanged", 0.789, False),  # SS22
        ],
    ),
    (
        "Detect an apnea in the current window",
        AQUA,
        [
            ("NAM, 6 channels", 0.870, False),                 # SS7, lag = -3000
        ],
    ),
    (
        "Predict a coming apnea",
        ORANGE,
        [
            ("NAM, 6 channels, 15 s ahead", 0.540, False),      # SS7, lag = +3000
            ("literature-style features, 15 s", 0.554, False),                # SS15
            ("effort / causal 30 min baseline, 60 s ahead", 0.558, False),    # SS24
            ("best of 6 adaptation variants", 0.536, False),    # DA SS10.2
            ("Robin model applied unchanged (zero-shot)", 0.474, False),      # DA SS10.2
        ],
    ),
    (
        "Predict an apnea-prone state, next 1 min",
        YELLOW,
        [
            ("event history + physiology", 0.677, False),      # SS13
            ("event history only", 0.667, False),              # SS13
            ("Hawkes process, 3 parameters", 0.617, False),    # SS16
            ("physiology only", 0.563, False),                 # SS13
        ],
    ),
    (
        "Predict mean saturation below 90 %, next 1 min",
        MAGENTA,
        # SS23.7(c), `declabs` at lead 60 s -- the endpoint is an ABSOLUTE
        # clinical threshold (mean SpO2 < 90 % over the next minute), so no
        # feature can move the goalposts, and the 60 s lead is what removes the
        # persistence/detection component SS23.7(b) warns about. Raw output:
        # outputs/spo2_baseline/declabs_lead_sweep.txt, "horizon 60s, lead 60s".
        # These three replace an earlier set (0.809 / 0.740 / 0.688) that was
        # shifted by one row against that file and had no traceable source.
        [
            ("all features", 0.740, False),                                 # SS23
            (r"$\mathrm{SpO_2}$ baseline drift alone", 0.706, False),        # SS23
            (r"$\mathrm{M-B_{30}}$, a single feature", 0.674, False),        # SS23
        ],
    ),
]


def fig_task_map(out_dir):
    # Lay the rows out top-down with a gap between sections, then flip, so the
    # first section reads at the top. The gap has to hold a two-line heading.
    SECTION_GAP = 1.6
    rows, ypos, heads = [], [], []
    y = 0.0
    for title, colour, entries in TASK_MAP:
        heads.append((title, colour, y))
        for label, auc, is_ref in entries:
            rows.append((label, auc, colour, is_ref))
            ypos.append(y)
            y += 1.0
        y += SECTION_GAP

    total = y - SECTION_GAP
    ypos = [total - 1.0 - v for v in ypos]
    heads = [(t, c, total - 1.0 - a) for t, c, a in heads]

    # Drawn at the width of the thesis text block, so it is included at
    # width=\linewidth with no scaling and every label prints at the point size
    # set here. Scaling the figure down in LaTeX instead shrank the 7.5 pt row
    # labels to about 4 pt on the page.
    fig, ax = plt.subplots(figsize=(6.3, 4.55))
    # A wide left margin: the row labels and the section headings share it, so
    # it has to hold the longest of either without the two ever overlapping.
    LEFT, RIGHT = 0.41, 0.965
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=0.99, bottom=0.15)
    # Headings start at the left edge of the figure, not of the axes. In axis
    # coordinates that is negative, and how negative depends on the margin.
    head_x = (0.012 - LEFT) / (RIGHT - LEFT)

    for yy, (label, auc, colour, is_ref) in zip(ypos, rows):
        # Bars measure from chance, not from zero or from an arbitrary cut, so
        # bar length is the meaningful quantity ("how far above chance") and the
        # sign flip at zero-shot shows up directly as a leftward bar.
        ax.barh(
            yy, auc - CHANCE, left=CHANCE, height=0.62,
            color="none" if is_ref else colour,
            edgecolor=colour, linewidth=1.4,
            # The published number was not measured here, so it is hollow
            # rather than presented as one of this project's own results.
            hatch="////" if is_ref else None,
        )
        offset = 0.005 if auc >= CHANCE else -0.005
        ax.text(auc + offset, yy, f"{auc:.3f}", va="center",
                ha="left" if auc >= CHANCE else "right",
                fontsize=8, color=INK)

    # Section headings sit ABOVE their group, left-aligned across the whole
    # width, with a rule under them. Putting them beside the rows would put two
    # kinds of label in one column, which is what collides.
    for title, colour, top in heads:
        ax.text(head_x, top + 0.88, title,
                transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.5, color=colour,
                fontweight="bold", linespacing=1.35)
        ax.plot([head_x, 1.0], [top + 0.56] * 2,
                transform=ax.get_yaxis_transform(), color=colour,
                linewidth=0.8, alpha=0.45, clip_on=False, zorder=0)

    ax.axvline(CHANCE, color=INK, linewidth=1.5, zorder=3)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK2)
    ax.set_ylim(-0.7, total + 1.4)
    ax.set_xlim(0.42, 0.90)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9"])
    ax.set_xlabel("mean test AuROC, leave-one-patient-out")
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    # No figure title: in the thesis the LaTeX caption directly below says the
    # same thing, and two statements of it read as a mistake.

    _save(fig, out_dir, "task_map", tight=False)


# =========================================================================
# Fig 2: horizon curves
# =========================================================================
# (a) Section 4.4 of the thesis SS7, the lag sweep. lag is in samples at 200 Hz;
#     positive lag = the window ends that many samples BEFORE onset. Plotted on
#     a "window end relative to onset" axis, so time runs left to right.
LAG_SAMPLES = [-3000, -1500, -750, 0, 1000, 3000, 6000, 12000]
WINDOW_END_S = [-s / 200.0 for s in LAG_SAMPLES]   # +15 .. -60 s
LAG_TRAIN = [0.880, 0.739, 0.710, 0.683, 0.667, 0.646, 0.637, 0.633]
LAG_TEST = [0.870, 0.667, 0.623, 0.581, 0.561, 0.540, 0.523, 0.552]

# (b) Section 4.4 of the thesis SS13 (feature sets) and SS16 (Hawkes),
#     cross-patient leave-one-out.
STATE_HORIZONS = [1, 2, 3]
STATE_CURVES = [
    ("event history + physiology", [0.677, 0.639, 0.622], ORANGE),
    ("event history only", [0.667, 0.634, 0.622], BLUE),
    ("Hawkes process (3 parameters)", [0.617, 0.615, np.nan], AQUA),
    ("physiology only", [0.563, 0.547, np.nan], MUTE),
]


def fig_horizon_curves(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    # --- (a) ------------------------------------------------------------
    ax = axes[0]
    # The shaded band is where the window still overlaps the scored event, i.e.
    # where the task is detection rather than prediction. Naming it removes the
    # need to explain why the left-hand points are allowed to be high.
    ax.axvspan(0, 17, color=GRID, zorder=0)
    # The regimes are named ON the axis, below the chance rule where no curve
    # runs. Without them the eye reads the left-to-right climb as "prediction
    # improves", when what actually changes is the TASK: past zero the window
    # starts to contain the apnea, so the model is detecting, not predicting.
    ax.text(-33, 0.462, "predicting\nwindow ends before onset", ha="center",
            va="bottom", fontsize=6.8, color=INK2, linespacing=1.3)
    ax.text(8.5, 0.462, "detecting\nevent inside", ha="center",
            va="bottom", fontsize=6.8, color=INK2, linespacing=1.3)

    ax.plot(WINDOW_END_S, LAG_TRAIN, "o--", color=MUTE, markersize=4,
            label="train (data the model learned from)")
    ax.plot(WINDOW_END_S, LAG_TEST, "o-", color=ORANGE, markersize=4,
            label="test (held-out infant)")

    ax.axhline(CHANCE, color=INK, linewidth=1.2, zorder=1)
    ax.text(-62, 0.507, "chance", fontsize=7, color=INK2, va="bottom")

    # The reading: the train curve falls with the test curve. A bug or an
    # overfit gives the opposite shape -- train stays high, test drops.
    ax.annotate("train falls with test:\nnothing to fit, not overfitting",
                xy=(-15, LAG_TRAIN[5] + 0.006), xytext=(-62, 0.775),
                ha="left", va="center", color=INK, fontsize=7.5,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.1,
                                shrinkA=3, shrinkB=3,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_xlim(-66, 19)
    ax.set_ylim(0.455, 0.95)
    ax.set_xticks([-60, -45, -30, -15, 0, 15])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("window end relative to apnea onset (s)")
    ax.set_ylabel("AuROC")
    ax.set_title("a   The rise is the apnea entering the window,\n"
                 "     not prediction getting better",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    # --- (b) ------------------------------------------------------------
    ax = axes[1]
    for label, vals, colour in STATE_CURVES:
        ax.plot(STATE_HORIZONS, vals, "o-", color=colour, markersize=4,
                label=label)

    # No per-point values: two of the series land on 0.622 at 3 min and the
    # labels would print on top of each other. The one number that carries the
    # panel is the gap at 1 min, so that is the one drawn.
    hi, lo = 0.667, 0.563
    ax.annotate("", xy=(1.14, hi), xytext=(1.14, lo),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.1))
    # One line, set below the Hawkes curve rather than centred on the arrow --
    # centred, it straddles that curve.
    ax.text(1.20, 0.587, "+0.104 from the event history alone",
            va="center", ha="left", fontsize=7.5, color=INK)

    ax.axhline(CHANCE, color=INK, linewidth=1.2, zorder=1)
    ax.text(0.93, 0.507, "chance", fontsize=7, color=INK2, va="bottom")

    ax.set_xlim(0.9, 3.25)
    ax.set_ylim(0.455, 0.95)
    ax.set_xticks(STATE_HORIZONS)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("horizon (min)")
    ax.set_title("b   What is predictable: an apnea-prone state,\n"
                 "     from the event history rather than the physiology",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    fig.tight_layout(w_pad=2.5)
    _save(fig, out_dir, "horizon_curves")


# =========================================================================
# Fig 2b: the same lag sweep, for a reader who knows the physiology
# =========================================================================
# Same numbers as fig_horizon_curves panel (a), same source (SS7). Built to be
# talked over, so it carries no written argument: the train curve is dropped
# (a point about fitting, not about breathing) and the annotations with it.
# What is added is the 30 s window drawn against a breathing trace, because
# "the window ends before the pause" is a picture, not a number.
APNEA_START, APNEA_END = 0.0, 20.0      # a central apnea lasts at least 10 s
WINDOW_S = 30.0                          # the model's input window (SS7)


def fig_horizon_simple(out_dir):
    fig, (ax_s, ax) = plt.subplots(
        2, 1, figsize=(6.6, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 2.4], hspace=0.12))

    xlo, xhi = -70, 33

    # --- top: what the model is actually handed ---------------------------
    ax_s.set_xlim(xlo, xhi)
    ax_s.set_ylim(0, 1)
    for side in ax_s.spines.values():
        side.set_visible(False)
    ax_s.set_yticks([])
    ax_s.tick_params(axis="x", bottom=False, labelbottom=False)

    # Airflow, schematic. The breathing before the pause is drawn unchanged,
    # because that is the finding: the 30 s before an apnea look ordinary.
    t = np.linspace(xlo + 9, xhi, 2000)   # clear of the "airflow" label
    breath = 0.13 * np.sin(2 * np.pi * t / 6.0) + 0.81
    breath[(t >= APNEA_START) & (t <= APNEA_END)] = 0.81
    ax_s.plot(t, breath, color=INK2, linewidth=1.0, solid_capstyle="round")
    ax_s.text(xlo + 1.5, 0.99, "airflow", fontsize=7.5, color=INK2, va="top")

    # The pause itself, marked once and carried down into the curve panel.
    for a in (ax_s, ax):
        a.axvspan(APNEA_START, APNEA_END, color=GRID, zorder=0)
    ax_s.text((APNEA_START + APNEA_END) / 2, 0.99, "apnea", fontsize=7.5,
              color=INK, ha="center", va="top", fontweight="bold")

    # Two windows: one that ends before the pause, one that runs into it.
    # Unlabelled beyond their width -- the presenter says the rest.
    for end_s, y, colour in [(-15.0, 0.34, MUTE), (15.0, 0.06, ORANGE)]:
        ax_s.add_patch(plt.Rectangle((end_s - WINDOW_S, y), WINDOW_S, 0.20,
                                     facecolor="none", edgecolor=colour,
                                     linewidth=1.3, zorder=3))
        ax_s.text(end_s - WINDOW_S / 2, y + 0.10, "30 s window", fontsize=7,
                  color=colour, ha="center", va="center", zorder=4)
        ax_s.plot([end_s, end_s], [0.0, y], color=colour, linewidth=0.8,
                  linestyle=":", zorder=2)
        ax.axvline(end_s, color=colour, linewidth=0.8, linestyle=":", zorder=1)

    # --- bottom: the score, one curve -------------------------------------
    ax.plot(WINDOW_END_S, LAG_TEST, "o-", color=ORANGE, markersize=5, zorder=4)

    ax.axhline(CHANCE, color=INK, linewidth=1.2, zorder=2)
    ax.text(xlo + 1.5, 0.507, "chance", fontsize=7.5, color=INK2, va="bottom")

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.455, 0.95)
    ax.set_xticks([-60, -45, -30, -15, 0, 15, 30])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("window end relative to apnea onset (s)")
    ax.set_ylabel("AuROC")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.97, bottom=0.115)
    _save(fig, out_dir, "horizon_simple", tight=False)


# =========================================================================
# Fig 3: domain adaptation
# =========================================================================
# Section 4.5 of the thesis SS10.3. Each row differs from the one above it in exactly
# one respect, so every gap is separately attributable. Colour marks the cohort,
# because the cohort is the variable the experiment isolates.
DA_LADDER = [
    ("Robin, 6 channels, apnea + hypopnea", 0.780, BLUE, True),
    ("Robin, 6 channels, apnea only", 0.723, BLUE, False),
    ("Robin, 5 shared channels, apnea only", 0.704, BLUE, False),
    ("the same model applied to CPAP", 0.474, ORANGE, False),
    ("CPAP, trained on CPAP", 0.527, ORANGE, False),
]
DA_DELTAS = [  # Section 4.5 of the thesis SS10.2, vs the `target_only` control
    ("fine-tuning, head only", +0.009, 0.720),
    ("fine-tuning, all weights", -0.005, 0.804),
    ("pooled (source + target)", -0.021, 0.389),
    ("CORAL", -0.025, 0.489),
    ("zero-shot", -0.053, 0.303),
    ("AdaBN", -0.056, 0.252),
]
# Section 4.4 of the thesis SS33.6: repeated training on identical config and
# seed moves the result by up to 0.022. Anything inside that band is noise.
SEED_SPREAD = 0.022


# Section 4.5 of the thesis SS11.1, the per-patient rung. Unlike the ladder above,
# the per-patient result pickle IS in the repo, so panel c is computed from the
# stored per-window scores rather than transcribed. One seed.
ADABN_PKL = "results/adabn_patient/adabn_patient_seed0_da_results.pkl"


def _load_adabn_per_patient():
    """Per-infant AuROC for the two test-time-adaptation arms, or None."""
    path = os.path.join(REPO, ADABN_PKL)
    if not os.path.exists(path):
        return None
    import pickle

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    ids = sorted(d["results"]["zero_shot_late"])
    ctrl = d["summary"]["zero_shot_late"]["per_patient_auc"]
    adapt = d["summary"]["adabn_patient"]["per_patient_auc"]
    return ids, np.asarray(ctrl, float), np.asarray(adapt, float)


def fig_domain_adaptation(out_dir):
    fig = plt.figure(figsize=(11.0, 6.4))
    # Two rows: the cohort-level experiment on top, the per-infant rung below.
    # The bottom panel spans the full width because it carries 15 categories.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.75,
                          wspace=0.42, left=0.20, right=0.93,
                          top=0.93, bottom=0.10)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    # --- A: where the performance is lost --------------------------------
    ax = axes[0]
    ypos = np.arange(len(DA_LADDER))[::-1]
    for yy, (label, auc, colour, is_ref) in zip(ypos, DA_LADDER):
        ax.barh(yy, auc - CHANCE, left=CHANCE, height=0.62,
                color="none" if is_ref else colour,
                edgecolor=colour, linewidth=1.4,
                hatch="////" if is_ref else None)
        offset = 0.006 if auc >= CHANCE else -0.006
        ax.text(auc + offset, yy, f"{auc:.3f}", va="center",
                ha="left" if auc >= CHANCE else "right",
                fontsize=8, color=INK)

    # The step deltas, written between the rows they separate. The fourth is
    # the decisive one and is the only one emphasised.
    for i, (delta, note, strong) in enumerate([
        (-0.057, "narrower label set", False),
        (-0.019, "loss of the airflow channel", False),
        (-0.230, "the cohort itself", True),
        (+0.053, "target-specific training", False),
    ]):
        y_mid = ypos[i] - 0.5
        ax.text(0.885, y_mid, f"{delta:+.3f}  {note}",
                va="center", ha="right",
                fontsize=7.5 if strong else 7,
                color=ORANGE if strong else MUTE,
                fontweight="bold" if strong else "normal")

    ax.axvline(CHANCE, color=INK, linewidth=1.5, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in DA_LADDER], fontsize=7.5, color=INK2)
    # An empty row below the last bar holds the legend, and the extra room on
    # the left keeps the below-chance value label off the row label.
    ax.set_ylim(-1.35, len(DA_LADDER) - 0.35)
    ax.set_xlim(0.42, 0.89)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(["0.5\nchance", "0.6", "0.7", "0.8"])
    ax.set_xlabel("mean AuROC over patients")
    ax.set_title("a   Where the performance is lost", loc="left",
                 fontweight="bold", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Robin (source)", "CPAP (target)"],
              loc="lower right", fontsize=7, ncol=2)

    # --- B: no method beats the control ----------------------------------
    ax = axes[1]
    ypos = np.arange(len(DA_DELTAS))[::-1]
    # The band is the pipeline's own seed-to-seed spread. Every delta lands
    # inside it, which is the finding; drawing it removes the need to argue
    # from the p-values alone. Paired per-patient CIs need the result pickles
    # (see make_da_figure.py) and are not reproducible from the tables.
    ax.axvspan(-SEED_SPREAD, SEED_SPREAD, color=GRID, zorder=0)
    ax.text(0, len(DA_DELTAS) - 0.35, "seed-to-seed spread of this pipeline",
            ha="center", va="bottom", fontsize=7, color=INK2)

    for yy, (label, delta, pval) in zip(ypos, DA_DELTAS):
        # Nothing reaches significance, so nothing is emphasised: colouring a
        # non-significant point would imply an effect the test does not support.
        ax.plot(delta, yy, "o", color=INK2, markersize=5, zorder=3)
        ax.text(1.03, yy, f"{pval:.2f}", va="center", ha="left", fontsize=7,
                color=MUTE, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.03, len(DA_DELTAS) - 0.35, "p", va="bottom", ha="left",
            fontsize=7, color=MUTE, fontstyle="italic",
            transform=ax.get_yaxis_transform(), clip_on=False)

    ax.axvline(0, color=INK, linewidth=1.5, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in DA_DELTAS], fontsize=7.5, color=INK2)
    ax.set_ylim(-0.7, len(DA_DELTAS) + 0.25)
    ax.set_xlim(-0.09, 0.09)
    ax.set_xticks([-0.08, -0.04, 0.0, 0.04, 0.08])
    ax.set_xticklabels(["-0.08", "-0.04", "0\nno effect", "0.04", "0.08"])
    ax.set_xlabel("AuROC change vs. training on CPAP alone\n"
                  "(paired over 15 infants, 5 seeds, Wilcoxon)")
    ax.set_title("b   No adaptation method beats the control", loc="left",
                 fontweight="bold", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    # --- C: the per-infant rung ------------------------------------------
    loaded = _load_adabn_per_patient()
    ax = fig.add_subplot(gs[1, :])
    if loaded is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{ADABN_PKL} not found — panel c skipped",
                ha="center", va="center", fontsize=8, color=MUTE)
    else:
        ids, ctrl, adapt = loaded
        diff = adapt - ctrl
        order = np.argsort(diff)
        ids = [ids[i] for i in order]
        diff = diff[order]

        # Colour by sign, not by magnitude: the question is only whether
        # adapting to the infant helped that infant, and the answer is split.
        colours = [BLUE if d >= 0 else ORANGE for d in diff]
        ax.bar(np.arange(len(diff)), diff, width=0.62, color=colours)
        ax.axhline(0, color=INK, linewidth=1.5, zorder=3)

        mean = float(np.mean(diff))
        ax.axhline(mean, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(len(diff) - 0.4, mean - 0.004,
                f"mean {mean:+.3f}  (p = 0.36, Wilcoxon)",
                ha="right", va="top", fontsize=7, color=INK2)

        ax.set_xticks(np.arange(len(diff)))
        ax.set_xticklabels(ids, fontsize=7)
        ax.set_xlim(-0.7, len(diff) - 0.3)
        ax.set_xlabel("infant")
        ax.set_ylabel("AuROC change")
        # Both lines belong to the title. Set as a separate `text` at the
        # axes top they landed on each other.
        ax.set_title(
            "c   Adapting to the individual infant does not help either — "
            f"better in {int((diff > 0).sum())} of {len(diff)}, worse in "
            f"{int((diff < 0).sum())}\n"
            "     AdaBN re-estimated on the test infant's own first 33 %, "
            "scored on the rest. One seed.",
            loc="left", fontweight="bold", fontsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)

    _save(fig, out_dir, "domain_adaptation", tight=False)


# =========================================================================
# Fig 4: the normalisation 2x2 and the closing sweeps (B3, B4, B5, B2.7, F3)
# =========================================================================
# Panel a: outputs/norm_ablation/b3_stats_10seeds.txt, block [1], all 15
# infants, count-weighted, with the seed SD of the cell mean over 10 seeds.
NORM_CELLS = [
    ("per-window\n(published)", 0.5557, 0.0075, False),
    ("per-block\n(primary)", 0.5827, 0.0067, True),
    ("both", 0.5513, 0.0060, False),
    ("neither\n(control)", 0.5747, 0.0120, False),
]
# Panel b: every paired contrast from the ablation and the four closing sweeps.
# `spread` is the seed-to-seed spread OF THAT CONTRAST where the analysis
# tabulated it (b3_stats block [2]; Section 4.4 of the thesis SS33.1-33.4).
# None means it was not tabulated -- drawn as a bare point rather than
# borrowing another contrast's interval.
#   (label, delta, spread, p, source tag)
NORM_CONTRASTS = [
    ("per-block minus per-window", +0.0270, 0.0069, 0.252, "B3"),
    ("no normalisation minus per-window", +0.0191, 0.0121, 0.055, "B3"),
    ("no normalisation minus per-block", -0.0080, 0.0145, 0.762, "B3"),
    ("both minus per-window", -0.0044, 0.0041, 0.107, "B3"),
    ("7th channel: effort / causal 120 s baseline", -0.0021, 0.0117, 0.042, "B4"),
    ("HR and SpO2 as two channels each", -0.0073, 0.0131, 0.390, "B2.7"),
    ("causal normalisation, expanding", -0.0100, None, 0.430, "B5"),
    ("causal normalisation, trailing", -0.0080, None, 0.760, "B5"),
    ("equalised per-infant sampling", +0.0010, None, 0.934, "F3"),
]
PRESPEC_BAR = 0.65  # PRE_SPECIFICATION.md SS3, set before the runs


def fig_norm_ablation(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.subplots_adjust(left=0.115, right=0.90, top=0.86, bottom=0.20,
                        wspace=0.85)

    # --- a: the four cells -----------------------------------------------
    ax = axes[0]
    x = np.arange(len(NORM_CELLS))
    for xi, (_lab, val, sd, is_best) in zip(x, NORM_CELLS):
        ax.bar(xi, val - CHANCE, bottom=CHANCE, width=0.62,
               color=ORANGE if is_best else BLUE,
               edgecolor=ORANGE if is_best else BLUE, linewidth=1.4)
        ax.errorbar(xi, val, yerr=sd, fmt="none", ecolor=INK,
                    elinewidth=1.2, capsize=3, zorder=4)
        ax.text(xi, val + sd + 0.004, f"{val:.4f}", ha="center", va="bottom",
                fontsize=7.5, color=INK)

    # The bar the analysis had to clear was fixed before the runs. Drawing it
    # is the difference between "the best cell" and "the best cell, and still
    # short of what was called a result".
    ax.axhline(PRESPEC_BAR, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(len(NORM_CELLS) - 0.5, PRESPEC_BAR - 0.004,
            "pre-registered bar, 0.65 — not met by any cell",
            ha="right", va="top", fontsize=7, color=INK2)

    ax.axhline(CHANCE, color=INK, linewidth=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in NORM_CELLS], fontsize=7.5, color=INK2,
                       linespacing=1.4)
    ax.set_ylim(0.49, 0.70)
    ax.set_yticks([0.5, 0.55, 0.60, 0.65, 0.70])
    ax.set_yticklabels(["0.50\nchance", "0.55", "0.60", "0.65", "0.70"])
    ax.set_ylabel("count-weighted AuROC")
    ax.set_title("a   The normalisation 2x2\n"
                 "     4 cells x 10 seeds, 15 infants, identical windows",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    # --- b: every contrast ------------------------------------------------
    ax = axes[1]
    ypos = np.arange(len(NORM_CONTRASTS))[::-1]
    for yy, (label, delta, spread, pval, tag) in zip(ypos, NORM_CONTRASTS):
        if spread is not None:
            # The interval IS the decision rule: under the study's own
            # two-part test an effect has to clear its own seed spread as
            # well as the Wilcoxon, so plotting the spread shows the verdict.
            ax.errorbar(delta, yy, xerr=spread, fmt="o", color=INK2,
                        ecolor=MUTE, elinewidth=1.5, capsize=3, markersize=5,
                        zorder=3)
        else:
            ax.plot(delta, yy, "o", color=INK2, markersize=5,
                    markerfacecolor="white", zorder=3)
        ax.text(1.03, yy, f"{pval:.3f}", va="center", ha="left", fontsize=7,
                color=MUTE, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.03, len(NORM_CONTRASTS) - 0.35, "p", va="bottom", ha="left",
            fontsize=7, color=MUTE, fontstyle="italic",
            transform=ax.get_yaxis_transform(), clip_on=False)

    ax.axvline(0, color=INK, linewidth=1.5, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c[0] for c in NORM_CONTRASTS], fontsize=7.5,
                       color=INK2)
    ax.set_ylim(-0.8, len(NORM_CONTRASTS) - 0.15)
    ax.set_xlim(-0.045, 0.045)
    ax.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax.set_xticklabels(["-0.04", "-0.02", "0\nno effect", "0.02", "0.04"])
    ax.set_xlabel("AuROC change vs. that contrast's own control\n"
                  "bars are the seed-to-seed spread of the same contrast; "
                  "hollow = spread not tabulated")
    ax.set_title("b   Nothing clears its own noise",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    _save(fig, out_dir, "norm_ablation", tight=False)


# =========================================================================
# Fig 5: the SpO2 baseline idea, measured
# =========================================================================
# outputs/spo2_baseline/ablation_results.txt (panel a, lead 0, horizon 60 s)
# and outputs/spo2_baseline/lead_sweep.txt (panel b). Cross-patient
# leave-one-out throughout.
SPO2_SETS = [
    ("SpO2 baseline drift (B)", BLUE),
    ("desaturation burden (A)", AQUA),
    ("event history", YELLOW),
    ("all features", ORANGE),
]
SPO2_BY_ENDPOINT = {
    # endpoint -> AuROC per feature set, same order as SPO2_SETS
    "a central apnea\nin the next 60 s": [0.526, 0.491, 0.660, 0.663],
    "intermittent hypoxaemia\nin the next 60 s": [0.740, 0.695, 0.631, 0.809],
}
SPO2_LEADS = [0, 60, 120]
SPO2_LEAD_CURVES = [
    ("baseline decline — all features", [0.872, 0.730, 0.697], ORANGE, "-"),
    ("baseline decline — drift only", [0.820, 0.699, 0.677], ORANGE, "--"),
    ("hypoxaemia — all features", [0.812, 0.672, 0.660], BLUE, "-"),
    ("hypoxaemia — drift only", [0.739, 0.647, 0.627], BLUE, "--"),
]


def fig_spo2_baseline(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.19,
                        wspace=0.28)

    # --- a: the dissociation ---------------------------------------------
    ax = axes[0]
    endpoints = list(SPO2_BY_ENDPOINT)
    n = len(SPO2_SETS)
    width = 0.19
    for j, (label, colour) in enumerate(SPO2_SETS):
        vals = [SPO2_BY_ENDPOINT[e][j] for e in endpoints]
        xs = np.arange(len(endpoints)) + (j - (n - 1) / 2) * (width + 0.012)
        ax.bar(xs, [v - CHANCE for v in vals], bottom=CHANCE, width=width,
               color=colour, label=label)
        for xi, v in zip(xs, vals):
            # A below-chance bar grows downward, so its label goes under the
            # bar end rather than on top of the chance rule.
            above = v >= CHANCE
            ax.text(xi, v + (0.007 if above else -0.018), f"{v:.3f}",
                    ha="center", va="bottom" if above else "top",
                    fontsize=6.5, color=INK)

    ax.axhline(CHANCE, color=INK, linewidth=1.5, zorder=3)
    ax.set_xticks(np.arange(len(endpoints)))
    ax.set_xticklabels(endpoints, fontsize=8, color=INK2, linespacing=1.4)
    # 0.45, not 0.47: the one below-chance bar labels underneath itself, and
    # at 0.47 that label collided with the group captions under the axis.
    ax.set_ylim(0.45, 0.93)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9"])
    ax.set_ylabel("AuROC, leave-one-patient-out")
    ax.set_title("a   The baseline drift predicts oxygenation, not apnea",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    # --- b: does it survive real lead time? -------------------------------
    ax = axes[1]
    for label, vals, colour, style in SPO2_LEAD_CURVES:
        ax.plot(SPO2_LEADS, vals, style, marker="o", color=colour,
                markersize=4, label=label)

    ax.axhline(CHANCE, color=INK, linewidth=1.2, zorder=1)
    ax.text(2, 0.507, "chance", fontsize=7, color=INK2, va="bottom")
    # (legend goes lower right: the curves all end high, so that corner is free)
    ax.set_xlim(-6, 128)
    ax.set_ylim(0.47, 0.93)
    ax.set_xticks(SPO2_LEADS)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("gap between the window and the target (s)")
    ax.set_title("b   Most of it is nowcasting: the signal decays\n"
                 "     once a real gap is inserted",
                 loc="left", fontweight="bold", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    _save(fig, out_dir, "spo2_baseline_results", tight=False)


def fig_spo2_simple(out_dir):
    """The SpO2 baseline-drift idea alone, one panel, for talking over.

    Only feature set B, the drift itself, and only the two endpoints it was
    proposed for. The other three feature sets, the lead sweep and the titles
    stay in `spo2_baseline_results`; here the two bars are the whole answer.
    """
    labels = ["a central apnea\nin the next minute",
              "low SpO2\nin the next minute"]
    vals = [SPO2_BY_ENDPOINT[e][0] for e in SPO2_BY_ENDPOINT]  # set B = drift

    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    xs = np.arange(len(vals))
    ax.bar(xs, [v - CHANCE for v in vals], bottom=CHANCE, width=0.5,
           color=BLUE, zorder=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)

    ax.axhline(CHANCE, color=INK, linewidth=1.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5, color=INK2, linespacing=1.4)
    ax.set_xlim(-0.65, len(vals) - 0.35)
    ax.set_ylim(0.47, 0.80)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.5\nchance", "0.6", "0.7", "0.8"])
    ax.set_ylabel("AuROC")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(left=0.155, right=0.98, top=0.965, bottom=0.175)
    _save(fig, out_dir, "spo2_simple", tight=False)


def _save(fig, out_dir, name, tight=True):
    png = os.path.join(out_dir, name + ".png")
    pdf = os.path.join(out_dir, name + ".pdf")
    # `bbox_inches="tight"` re-crops to the drawn content, which throws away a
    # hand-set margin. Figures that place text by figure coordinates save as laid
    # out instead.
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(png, dpi=300, **kw)
    fig.savefig(pdf, **kw)
    plt.close(fig)
    print(f"wrote {png} and {pdf}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="figures/03_results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig_task_map(args.out_dir)
    fig_horizon_curves(args.out_dir)
    fig_horizon_simple(args.out_dir)
    fig_domain_adaptation(args.out_dir)
    fig_norm_ablation(args.out_dir)
    fig_spo2_baseline(args.out_dir)
    fig_spo2_simple(args.out_dir)


if __name__ == "__main__":
    main()
