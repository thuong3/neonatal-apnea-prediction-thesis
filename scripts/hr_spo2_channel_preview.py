"""CHEAP PREVIEW OF B2 BULLET 7, BEFORE ANY TRAINING.

B2's last bullet asks for HR and SpO2 to be carried as TWO channels each:

    "keep the fixed-range channel as primary and ADD a second, per-block
     robust-centred channel for each. The network then sees both the absolute
     level and the deviation from this infant's own baseline on this device,
     which is the cheapest possible attack on between-patient shift."

That takes the network from 6 input channels to 8, so it cannot be folded into
the B3 2x2 without breaking comparability -- it is a separate sweep, ~7 h of GPU
at the pre-specified 10 seeds. This script asks, in minutes and on real data,
whether it is worth them.

WHAT THE SECOND CHANNEL CAN AND CANNOT DO, AND WHY THE ANSWER IS NOT OBVIOUS
---------------------------------------------------------------------------
The fixed-range channel is `normalize_range(x, 50, 240, -1, 1)` for HR and
`normalize_range(x, 60, 100, -1, 1)` for SpO2 -- an AFFINE map, and an
invertible one. It therefore loses no information in any formal sense. The
second channel, `(x - median_b) / (IQR_b / 1.349)`, is ALSO affine, but with
coefficients that differ per block.

The consequence is exact and it is the whole design of this script:

  * WITHIN ONE BLOCK the two representations are related by a strictly
    increasing map, so any rank statistic of them is IDENTICAL. A per-block
    AuROC cannot distinguish them, and any difference reported at that level
    would be a bug. Test [1] checks this empirically rather than trusting it.

  * ACROSS BLOCKS the maps differ, so pooling windows from several blocks is the
    only place the second channel can do anything at all. What it removes there
    is the between-block/between-infant OFFSET -- exactly the "between-patient
    shift" B2 names.

So the question this script answers is narrow and answerable without a GPU:
once windows from all 15 infants and 60 blocks are pooled, does re-centring HR
and SpO2 per block separate pre-apnea windows from controls better than the
fixed-range level does?

If the pooled AuROCs are the same, the second channel has nothing to add on the
one axis it could ever have helped, and B2 bullet 7 can be declined WITH A
NUMBER instead of left open.

WHAT THIS IS NOT. A single-feature rank statistic is a floor, not a model
result: the network sees a 30 s waveform per channel, not a window mean, and it
can combine channels non-additively. Read a null here the way section 27.4 reads
its null -- as evidence the sweep is not worth running, not as proof that no
architecture could use the channel.

Run:
    python scripts/hr_spo2_channel_preview.py --ids 001 002 003
    python scripts/hr_spo2_channel_preview.py             # all 15
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.signal import decimate
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    NeoNatal,
    load_clock_drift,
    read_data,
    robust_stats,
)

DEFAULT_DATA = "data/brainimmaturity"
TARGET_FREQ = 200

# The fixed clinical ranges the published pipeline maps onto [-1, 1]. Used only
# to express the measured between-infant spread as a fraction of the range, the
# way `measure_block_scales.py` reports it.
FIXED_RANGE = {"HR": (50.0, 240.0), "SpO2": (60.0, 100.0)}

# B2 bullet 4's floor, applied to these two channels as well: patient 009's PCO2
# scale is exactly 0.000 in two of four blocks, so a dead physical-unit sensor is
# not hypothetical in this cohort. Same rule as config/experiment/norm_block.yaml
# -- 0.2 x the channel median across blocks, NOT the 5th percentile (section
# 27.2a has the measurement behind that deviation).
FLOOR_FRACTION = 0.2

# A feature that is constant by construction must never be handed to
# roc_auc_score: it ranks floating-point residue and returns a plausible-looking
# number. `norm_ablation_preview.py` was bitten by exactly this and its note is
# worth repeating here rather than re-learning.
CONSTANT_TOL = 1e-9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None, help="default: all in the config")
    return p.parse_args()


def channel_at_1hz(seg, channel, ss=slice(None)):
    """One channel over `ss`, decimated EXACTLY as `feature_extraction` does it.

    HR is plain subsampling at 200; SpO2 is subsampling at 100 then a
    decimate-by-2. Getting this wrong would compute the block statistics on a
    different signal from the one they normalise -- the failure mode
    `check_norm_chains.py` test [3] exists to catch for the oscillatory
    channels.
    """
    if channel == "HR":
        return np.asarray(seg["HR"][ss][::200], dtype=float)
    if channel == "SpO2":
        return np.asarray(decimate(seg["SpO2"][ss][::100], 2), dtype=float)
    raise ValueError(channel)


def artifact_free_mask(seg, cutter_events):
    """Boolean mask at TARGET_FREQ, from the CUTTER events only.

    Never from the adverse events. Excluding annotated apneas would make the
    statistics a function of the LABELS and leak them into held-out blocks --
    B2 bullet 3, the one that would invalidate the whole thing if wrong.
    """
    if not cutter_events:
        return None
    return (
        1 - np.column_stack([seg[event] for event in cutter_events]).max(axis=1)
    ).astype(bool)


def mask_to_length(mask, n_out):
    """Resample a TARGET_FREQ mask onto `n_out` samples (as block_norm_stats)."""
    if mask is None or len(mask) == 0 or n_out <= 0:
        return None
    idx = np.clip((np.arange(n_out) * (len(mask) / n_out)).astype(int), 0, len(mask) - 1)
    return np.asarray(mask)[idx].astype(bool)


def directional_auc(feature, labels):
    """AuROC in the DECLINING direction, or None if the feature is constant.

    Both hypotheses here are directional and both point the same way: HR falls
    (bradycardia) and SpO2 falls (desaturation) around a central apnea. So the
    quantity of interest is the AuROC of a FALLING value, i.e. 1 - AuROC.
    """
    spread = float(np.max(feature) - np.min(feature))
    if spread < CONSTANT_TOL:
        return None
    return 1.0 - float(roc_auc_score(labels, feature))


def collect(pat_id, data_path, cfg, drift):
    """Per-window raw levels + per-block robust statistics for one infant.

    Returns a list of per-block dicts. Nothing is normalised here: the floor
    needs the whole cohort's scales, so scoring happens in a second pass.
    """
    signal_dict = read_data(
        pat_id,
        data_path,
        annotations_dir=cfg.get("annotations_dir", "annotations"),
        record_duration=cfg.get("record_duration", None),
        clock_drift=drift.get(pat_id),
    )
    segments = [signal_dict] if isinstance(signal_dict, dict) else signal_dict

    # The window table only. HR and SpO2 are range-normalised in
    # feature_extraction and are touched by NEITHER normalisation switch, so the
    # cell chosen here cannot affect the answer -- but pick the control cell
    # explicitly rather than relying on that.
    ds = NeoNatal(
        pat_id,
        signal_dict,
        dataset_mode="list",
        signal_types=cfg.signal_types,
        adverse_events=cfg.adverse_events,
        cutter_events=cfg.cutter_events,
        time_window=cfg.time_window,
        lag=cfg.lag,
        away=cfg.away,
        norm_per_window=False,
        norm_per_block=False,
    )
    df = ds.time_window_df
    if df is None or not len(df):
        return []

    out = []
    for si, seg in enumerate(segments):
        rows = df[df["segment_idx"] == si]
        if not len(rows):
            continue
        mask = artifact_free_mask(seg, list(cfg.cutter_events))
        block = {"pat": pat_id, "block": si,
                 "label": rows["label"].to_numpy().astype(int)}
        for channel in ("HR", "SpO2"):
            whole = channel_at_1hz(seg, channel)
            block[f"{channel}_stats"] = robust_stats(
                whole, mask_to_length(mask, len(whole))
            )
            # Two summaries per window: the MEAN is the "absolute level" B2's
            # bullet is written about; the MIN is where the physiology actually
            # lives (a bradycardic dip, a desaturation nadir). Reporting both
            # stops a null on the mean being read as a null on the channel.
            means, mins = [], []
            for sl in rows["slice"]:
                win = channel_at_1hz(seg, channel, sl)
                win = win[np.isfinite(win)]
                if win.size == 0:
                    means.append(np.nan)
                    mins.append(np.nan)
                    continue
                means.append(float(win.mean()))
                mins.append(float(win.min()))
            block[f"{channel}_mean"] = np.asarray(means)
            block[f"{channel}_min"] = np.asarray(mins)
        out.append(block)
    return out


def measure_floors(blocks):
    """B2 bullet 4's floor per channel: 0.2 x the median scale across blocks."""
    floors = {}
    for channel in ("HR", "SpO2"):
        scales = np.array([b[f"{channel}_stats"][1] for b in blocks], dtype=float)
        good = scales[scales > 0.0]
        floors[channel] = FLOOR_FRACTION * float(np.median(good)) if good.size else 0.0
    return floors


def centred(block, channel, stat, floor):
    """The bullet-7 second channel: (x - median_b) / max(IQR_b/1.349, floor)."""
    m, s = block[f"{channel}_stats"]
    s = max(s, floor)
    return (block[f"{channel}_{stat}"] - m) / (s if s > 0.0 else 1.0)


def report_qc(blocks, floors):
    print("\n[0] QC on the physical-unit channels (B2 bullet 4 applied to HR/SpO2)")
    for channel in ("HR", "SpO2"):
        print(f"    {channel}: floor = {floors[channel]:.4g} "
              f"(0.2 x median block scale)")
        for b in blocks:
            _, s = b[f"{channel}_stats"]
            if s <= 0.0:
                print(f"      QC FAILURE  patient {b['pat']} block {b['block']}: "
                      f"scale is EXACTLY ZERO -> floored")
            elif s < floors[channel]:
                print(f"      floored     patient {b['pat']} block {b['block']}: "
                      f"scale {s:.4g} < floor")


def report_spread(blocks):
    """The quantity that decides whether the second channel has anything to say."""
    print("\n[!] BETWEEN-INFANT SPREAD -- what the second channel could remove")
    print(f"    {'channel':<8} {'between-infant sd':>18} {'as % of fixed range':>21}"
          f" {'median block scale':>20}")
    for channel in ("HR", "SpO2"):
        lo, hi = FIXED_RANGE[channel]
        width = hi - lo
        per_infant = {}
        for b in blocks:
            per_infant.setdefault(b["pat"], []).append(b[f"{channel}_stats"][0])
        means = np.array([np.mean(v) for v in per_infant.values()])
        between = float(means.std(ddof=1)) if means.size > 1 else float("nan")
        med_scale = float(np.median([b[f"{channel}_stats"][1] for b in blocks]))
        print(f"    {channel:<8} {between:>18.2f} {between / width:>20.1%}"
              f" {med_scale:>20.3f}")
    print("    A per-block centred channel can only encode what the fixed-range")
    print("    channel's OFFSET hides. Large spread relative to the range = worth")
    print("    building; small = the network already sees essentially this.")


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    drift = load_clock_drift(cfg.get("clock_drift_file", None))

    print("B2 BULLET 7 PREVIEW -- is a second, per-block robust-centred HR/SpO2")
    print("channel worth an 8-input network and a 10-seed sweep?\n")
    print(f"lag={cfg.lag} samples ({cfg.lag / TARGET_FREQ:.0f} s lead), "
          f"window={cfg.time_window / TARGET_FREQ:.0f} s, "
          f"controls >= {cfg.away / TARGET_FREQ:.0f} s from any event")
    print("direction: DECLINING (bradycardia / desaturation), i.e. 1 - AuROC\n")

    blocks = []
    for pat_id in ids:
        print(f"reading {pat_id} ...", flush=True)
        blocks.extend(collect(pat_id, args.data_path, cfg, drift))
    if not blocks:
        print("no usable blocks")
        return

    floors = measure_floors(blocks)
    report_qc(blocks, floors)
    report_spread(blocks)

    n_blocks = len(blocks)
    n_pats = len({b["pat"] for b in blocks})
    print(f"\ncollected {n_blocks} blocks over {n_pats} infants, "
          f"{sum(len(b['label']) for b in blocks)} windows "
          f"({sum(int(b['label'].sum()) for b in blocks)} positive)")

    # -------------------------------------------------------------- test [1]
    print("\n[1] WITHIN A BLOCK THE TWO REPRESENTATIONS MUST RANK IDENTICALLY")
    print("    (an affine map with a positive divisor preserves order, so any")
    print("     difference here is an implementation bug, not a result)")
    worst = 0.0
    for channel in ("HR", "SpO2"):
        for stat in ("mean", "min"):
            for b in blocks:
                y = b["label"]
                if y.sum() < 3 or (1 - y).sum() < 3:
                    continue
                fixed = directional_auc(b[f"{channel}_{stat}"], y)
                cen = directional_auc(centred(b, channel, stat, floors[channel]), y)
                if fixed is None or cen is None:
                    continue
                worst = max(worst, abs(fixed - cen))
    verdict = "OK" if worst < 1e-9 else "MISMATCH"
    print(f"    {verdict} -- max |AuROC(fixed) - AuROC(centred)| over blocks "
          f"= {worst:.2e}")

    # -------------------------------------------------------------- test [2]
    print("\n[2] POOLED ACROSS ALL BLOCKS -- the only place the 2nd channel can help")
    print(f"    {'channel':<8} {'stat':<6} {'fixed-range':>13} {'block-centred':>15}"
          f" {'delta':>9}")
    pooled_rows = []
    for channel in ("HR", "SpO2"):
        for stat in ("mean", "min"):
            y = np.concatenate([b["label"] for b in blocks])
            f_fixed = np.concatenate([b[f"{channel}_{stat}"] for b in blocks])
            f_cen = np.concatenate(
                [centred(b, channel, stat, floors[channel]) for b in blocks]
            )
            keep = np.isfinite(f_fixed) & np.isfinite(f_cen)
            a_fixed = directional_auc(f_fixed[keep], y[keep])
            a_cen = directional_auc(f_cen[keep], y[keep])
            pooled_rows.append((channel, stat, a_fixed, a_cen))
            print(f"    {channel:<8} {stat:<6} {a_fixed:>13.4f} {a_cen:>15.4f}"
                  f" {a_cen - a_fixed:>+9.4f}")

    # -------------------------------------------------------------- test [3]
    print("\n[3] PER-INFANT (pooled over that infant's blocks), the reported metric")
    print(f"    {'channel':<8} {'stat':<6} {'fixed mean':>11} {'centred mean':>13}"
          f" {'cnt-wtd fixed':>14} {'cnt-wtd centred':>16} {'Wilcoxon p':>11}")
    per_infant_rows = []
    for channel in ("HR", "SpO2"):
        for stat in ("mean", "min"):
            fixed_s, cen_s, npos_s = [], [], []
            for pat in sorted({b["pat"] for b in blocks}):
                pb = [b for b in blocks if b["pat"] == pat]
                y = np.concatenate([b["label"] for b in pb])
                if y.sum() < 3 or (1 - y).sum() < 3:
                    continue
                f_fixed = np.concatenate([b[f"{channel}_{stat}"] for b in pb])
                f_cen = np.concatenate(
                    [centred(b, channel, stat, floors[channel]) for b in pb]
                )
                keep = np.isfinite(f_fixed) & np.isfinite(f_cen)
                a_fixed = directional_auc(f_fixed[keep], y[keep])
                a_cen = directional_auc(f_cen[keep], y[keep])
                if a_fixed is None or a_cen is None:
                    continue
                fixed_s.append(a_fixed)
                cen_s.append(a_cen)
                npos_s.append(int(y.sum()))
            fixed_s = np.array(fixed_s)
            cen_s = np.array(cen_s)
            npos_s = np.array(npos_s, dtype=float)
            # Count-weighted is the primary summary per PRE_SPECIFICATION 1.2.
            wf = float((fixed_s * npos_s).sum() / npos_s.sum())
            wc = float((cen_s * npos_s).sum() / npos_s.sum())
            diff = cen_s - fixed_s
            if np.allclose(diff, 0.0):
                p = float("nan")
            else:
                try:
                    _, p = wilcoxon(diff)
                except ValueError:
                    p = float("nan")
            per_infant_rows.append((channel, stat, wf, wc, p, len(fixed_s)))
            print(f"    {channel:<8} {stat:<6} {fixed_s.mean():>11.4f}"
                  f" {cen_s.mean():>13.4f} {wf:>14.4f} {wc:>16.4f} {p:>11.4f}")
    print(f"    (n = {per_infant_rows[0][5]} infants; count-weighted is the "
          "PRE_SPECIFICATION 1.2 primary)")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("READING")
    best_gain = max(c - f for _, _, f, c in pooled_rows)
    best_level = max(max(f, c) for _, _, f, c in pooled_rows)
    print(f"  largest pooled gain from centring : {best_gain:+.4f}")
    print(f"  best AuROC either way             :  {best_level:.4f}")
    print("  Bullet 7 is worth the 8-input sweep only if BOTH are true:")
    print("    (a) centring gains something pooled -- otherwise the second")
    print("        channel is an order-preserving copy of the first, and")
    print("    (b) the channel separates at all -- a better view of a signal")
    print("        that is at chance is still at chance.")


if __name__ == "__main__":
    main()
