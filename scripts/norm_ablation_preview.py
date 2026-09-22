"""CHEAP PREVIEW OF THE B2/B3 NORMALISATION ABLATION, BEFORE ANY TRAINING.

The 2x2 of B3 costs a full leave-one-patient-out sweep per cell. This script
answers, in minutes and on real data, the question that decides whether those
sweeps are worth running:

    Under per-block normalisation, does the respiratory effort amplitude of a
    PRE-APNEA window actually differ from that of a CONTROL window?

That is the whole hypothesis. Per-window standardisation forces every window to
zero mean and unit variance, so the amplitude is gone by construction and the
network cannot use it however long it trains. Per-block normalisation keeps it.
If the two classes still overlap completely once it is kept, then the hypothesis
is dead in this cohort and the four sweeps will only confirm it expensively.

The separation is reported as an AuROC of a SINGLE FEATURE (window RMS of the
summed effort channel). Read it as a floor, not as a model result: a real
network sees the waveform, not just its amplitude. But a single-feature AuROC of
~0.50 means the amplitude carries nothing at all.

It also prints the per-block robust statistics, which is the QC pass B2 asks
for: a block whose scale sits far below its neighbours is a loose or
disconnected inductance belt.

Run:
    python scripts/norm_ablation_preview.py --ids 001 002 003
    python scripts/norm_ablation_preview.py            # all 15 (slow, ~10 min)
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import NeoNatal, load_clock_drift, read_data  # noqa: E402

DEFAULT_DATA = "data/brainimmaturity"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None, help="default: all in the config")
    # C4: restrict controls to a band of distances from the nearest event, in
    # SECONDS. Default (None) keeps the published rule -- controls >=3 min away.
    p.add_argument("--control-min", type=float, default=None)
    p.add_argument("--control-max", type=float, default=None)
    return p.parse_args()


def build(pat_id, signal_dict, cfg, control_min=None, control_max=None, **norm):
    return NeoNatal(
        pat_id,
        signal_dict,
        dataset_mode="list",
        signal_types=cfg.signal_types,
        adverse_events=cfg.adverse_events,
        cutter_events=cfg.cutter_events,
        time_window=cfg.time_window,
        lag=cfg.lag,
        away=cfg.away,
        control_min=control_min,
        control_max=control_max,
        **norm,
    )


def effort_rms(ds, channel_index):
    """Per-window RMS of one channel, with the window labels."""
    rms, labels = [], []
    for i in range(len(ds.time_window_df)):
        row = ds.time_window_df.iloc[i]
        rms.append(float(np.sqrt(np.mean(np.square(row["sig"][channel_index])))))
        labels.append(int(row["label"]))
    return np.asarray(rms), np.asarray(labels)


# A per-window standardised channel has mean 0 and sd 1 BY CONSTRUCTION, so its
# RMS is exactly 1 in every window and the feature is constant. Handing that to
# roc_auc_score does not return 0.5 -- it ranks the floating-point residue
# (~5e-16 on this data) and returns a plausible-looking number such as 0.596.
# Measured, not assumed: see the spread printed below. Any feature this flat is
# reported as constant instead of scored.
CONSTANT_TOL = 1e-9


def single_feature_auc(feature, labels):
    """AuROC of one feature, or None if the feature is constant by construction."""
    spread = float(feature.max() - feature.min())
    if spread < CONSTANT_TOL:
        return None, spread
    return float(roc_auc_score(labels, feature)), spread


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    thorax_idx = list(cfg.signal_types).index("Thorax")
    drift = load_clock_drift(cfg.get("clock_drift_file", None))

    print(f"channel under test: {cfg.signal_types[thorax_idx]} "
          f"(summed respiratory effort), index {thorax_idx}")
    print(f"lag={cfg.lag} samples ({cfg.lag / 200:.0f} s lead), "
          f"window={cfg.time_window / 200:.0f} s\n")

    cmin = None if args.control_min is None else int(args.control_min * 200)
    cmax = None if args.control_max is None else int(args.control_max * 200)
    if cmin is None and cmax is None:
        print("controls: published rule (>=%.0f s from any event)"
              % (cfg.away / 200))
    else:
        print("controls: C4 proximity band %s-%s s from the nearest event"
              % (args.control_min, args.control_max))
    print()

    rows = []
    for pat_id in ids:
        signal_dict = read_data(
            pat_id,
            args.data_path,
            annotations_dir=cfg.get("annotations_dir", "annotations"),
            record_duration=cfg.get("record_duration", None),
            clock_drift=drift.get(pat_id),
        )
        per_cell = {}
        for name, norm in (
            ("window", dict(norm_per_window=True, norm_per_block=False)),
            ("block", dict(norm_per_window=False, norm_per_block=True)),
        ):
            ds = build(pat_id, signal_dict, cfg,
                       control_min=cmin, control_max=cmax, **norm)
            if not len(ds.time_window_df):
                per_cell[name] = None
                continue
            rms, labels = effort_rms(ds, thorax_idx)
            per_cell[name] = (rms, labels)

        if per_cell["block"] is None:
            print(f"{pat_id}: no windows\n")
            continue

        rms_b, labels = per_cell["block"]
        rms_w, _ = per_cell["window"]
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        if n_pos < 3 or n_neg < 3:
            print(f"{pat_id}: too few windows ({n_pos} pos / {n_neg} neg) -- skipped\n")
            continue

        auc_b, spread_b = single_feature_auc(rms_b, labels)
        auc_w, spread_w = single_feature_auc(rms_w, labels)
        # The hypothesis is DIRECTIONAL -- effort DECLINES before a central
        # apnea -- so the quantity of interest is the AuROC of falling
        # amplitude, i.e. 1 - AuROC(rising amplitude).
        decline_b = None if auc_b is None else 1.0 - auc_b
        rows.append((pat_id, n_pos, n_neg, auc_w, decline_b,
                     rms_b[labels == 1].mean(), rms_b[labels == 0].mean()))
        shown_w = "constant" if auc_w is None else f"{auc_w:.3f}"
        print(f"{pat_id}: {n_pos:4d} pos / {n_neg:5d} neg | "
              f"per-window {shown_w} (spread {spread_w:.1e}) | "
              f"per-block declining-effort AuROC {decline_b:.3f} | "
              f"mean RMS pre-apnea {rms_b[labels == 1].mean():.3f} "
              f"vs control {rms_b[labels == 0].mean():.3f}\n")

    if not rows:
        print("no patient produced usable windows")
        return

    n_const = sum(1 for r in rows if r[3] is None)
    keep = [r for r in rows if r[4] is not None]
    aucs_b = np.array([r[4] for r in keep])
    n_pos = np.array([r[1] for r in keep])
    print("=" * 78)
    print(f"patients: {len(rows)}")
    print(f"per-window standardised: constant by construction in "
          f"{n_const}/{len(rows)} patients -- NOT SCORED (see CONSTANT_TOL)")
    print(f"per-block, declining-effort AuROC: "
          f"{aucs_b.mean():.3f} (SD {aucs_b.std(ddof=1):.3f}), "
          f"range {aucs_b.min():.3f}-{aucs_b.max():.3f}")
    print(f"  above 0.5 in {int((aucs_b > 0.5).sum())}/{len(aucs_b)} patients")

    # C6. Event rate varies ~38-fold between infants, so a plain mean over
    # infants weights a per-patient AuROC computed on 22 positives exactly as
    # heavily as one computed on 306. Report the count-weighted mean beside it,
    # and test explicitly whether the per-patient score tracks the count: if it
    # does, the plain mean is being carried by its noisiest terms.
    if len(keep) >= 4:
        weighted = float(np.sum(aucs_b * n_pos) / n_pos.sum())
        rho, p_rho = spearmanr(n_pos, aucs_b)
        try:
            _, p_w = wilcoxon(aucs_b - 0.5)
        except ValueError:
            p_w = float("nan")
        median_n = float(np.median(n_pos))
        lo, hi = aucs_b[n_pos <= median_n], aucs_b[n_pos > median_n]
        print(f"  count-weighted mean: {weighted:.3f}   <-- the C6-appropriate summary")
        print(f"  Wilcoxon signed-rank vs 0.5 over {len(aucs_b)} infants: p={p_w:.3f}")
        print(f"  AuROC vs n_positives: Spearman rho={rho:+.3f} (p={p_rho:.3f})")
        print(f"    few positives  (n<={median_n:.0f}, {len(lo):2d} infants): {lo.mean():.3f}")
        print(f"    many positives (n> {median_n:.0f}, {len(hi):2d} infants): {hi.mean():.3f}")
        if rho < -0.5 and p_rho < 0.05:
            print("    WARNING: the per-patient score falls as the positive count")
            print("    rises. The plain mean over infants is then carried by the")
            print("    infants with the fewest positives -- exactly the artifact")
            print("    C6 warns about. Trust the count-weighted mean.")
    print()
    print("READING")
    print("  The per-window column is the control and it is not a number: a")
    print("  standardised window has RMS exactly 1, so the amplitude feature")
    print("  does not exist there. That is precisely B3's point -- the published")
    print("  transform removes this information before the network sees it.")
    print()
    print("  The per-block column is a SINGLE FEATURE (window RMS). Compare it")
    print("  against the ~0.54 the full NAM reaches at this lead. If it is")
    print("  materially above that, per-window standardisation was discarding")
    print("  usable signal and the four sweeps are worth running.")
    print()
    print("  CONFOUND, and it is not small. Control windows sit >=3 min from any")
    print("  event (away=%d), while positives sit 15 s before one. Lower effort" % 36000)
    print("  amplitude before an apnea may therefore be reading 'periodic")
    print("  breathing vs. quiet regular breathing' rather than 'about to have")
    print("  an apnea'. Item C4 -- controls drawn from INSIDE periodic-breathing")
    print("  runs -- is the experiment that separates the two, and this result")
    print("  makes it the necessary next step rather than an optional one.")


if __name__ == "__main__":
    main()
