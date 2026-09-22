"""B6 + A4: WHAT IS MOVING THE CHEST DURING A SCORED CENTRAL APNEA?

THE PUZZLE THIS ADDRESSES
    Section 3.3 of the thesis found that the effort belts do NOT go
    quiet during the scored central apneas: belt amplitude during an event,
    compared with its own +-20 s surroundings, has a median ratio of ~1.0. That
    has sat in the document as an oddity ("belts noisy / overlaid by the flow
    generator") without a measurement behind it.

    Item B6 of the supervisor's to-do list supplies a candidate explanation.
    Some of the four devices in this crossover deliver MACHINE INFLATIONS at a
    fixed low rate. Those inflations move the chest and abdomen, and they carry
    on during a central apnea, because they do not depend on the infant. If that
    is what the belts are showing, then:

      * the ~1.0 ratio is explained, and is not a labelling error;
      * the scoring rule "both belts below 20% of the preceding breaths" does
        NOT mean the same thing in a machine-triggered arm as in a plain CPAP
        arm -- a validity question for the labels, not just a modelling nuisance;
      * and the device rhythm identifies WHICH block is which device, which is
        most of item A4, recovered from the signals instead of from the records.

WHAT IS MEASURED
    Inside every scored apnea, how fast is the chest actually moving, in cycles
    per minute? Three outcomes, and they mean different things:

        ~0 /min      belts are flat. A true central apnea, cleanly scored.
        ~10-30 /min  far too slow for a neonate (baseline is 53-56/min) and too
                     regular for noise -> machine inflations. B6 confirmed.
        ~50-60 /min  the infant is still breathing at its own rate during an
                     event scored as an apnea -> a labelling problem.

    The same rate is measured in the 30 s BEFORE each event as a within-block
    reference, so "slow" and "fast" are judged against that infant on that
    device rather than against a textbook number.

    Rates come from peak counting on the 5 Hz belt signal, with the peak
    prominence threshold set from the block's own robust scale (the same
    statistic the B2 normalisation uses). Peak counting is used rather than an
    FFT because a scored apnea is ~11 s long, which gives an FFT a frequency
    resolution of only ~0.09 Hz -- too coarse to separate a 10/min machine line
    from DC. Counting peaks does not have that limitation.

Usage:
    python scripts/device_rhythm.py --ids 001 002 003
    python scripts/device_rhythm.py                     # all 15 (slow)
"""
import argparse
import os
import sys

import numpy as np
from omegaconf import OmegaConf
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    DECIMATION_CHAIN,
    _apply_chain,
    _mask_to_length,
    load_clock_drift,
    read_data,
    robust_stats,
)

TARGET_FREQ = 200
BELT_FREQ = 5.0  # after the DECIMATION_CHAIN for Thorax/Abdomen
DEFAULT_DATA = "data/brainimmaturity"
MIN_EVENT_S = 8.0  # need a few cycles to count a rate at all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="config/dataset/neonatal.yaml")
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--prominence", type=float, default=0.5,
                   help="peak prominence, in units of the block's robust scale")
    return p.parse_args()


def event_spans(mask):
    mask = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def cycles_per_min(sig, prominence):
    """Peak rate of a 5 Hz belt segment, in cycles per minute."""
    if len(sig) < int(BELT_FREQ * 4):
        return np.nan
    peaks, _ = find_peaks(sig, prominence=prominence)
    return len(peaks) / (len(sig) / BELT_FREQ) * 60.0


def rhythm_regularity(sig, prominence):
    """Coefficient of variation of the inter-peak intervals, or nan.

    This is what separates a machine from noise, and rate alone does not. A
    ventilator inflating at a set rate produces near-identical intervals (CV
    well below ~0.2). Residual belt wobble during a true apnea produces a few
    scattered peaks whose spacing is arbitrary (CV around 0.5-1.0). Two segments
    can show the same cycles-per-minute and mean completely different things.
    """
    peaks, _ = find_peaks(sig, prominence=prominence)
    if len(peaks) < 4:
        return np.nan
    iv = np.diff(peaks).astype(float)
    if iv.mean() <= 0:
        return np.nan
    return float(iv.std() / iv.mean())


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.dataset)
    ids = args.ids if args.ids else list(cfg.ids)
    drift = load_clock_drift(cfg.get("clock_drift_file", None))

    print("Chest movement rate (cycles/min) inside scored central apneas,")
    print("against the 30 s before each event, per block.\n")
    print(f"{'pat':>4} {'blk':>3} {'n_ev':>5} {'during apnea':>22} "
          f"{'before apnea':>22}  {'verdict':<28}")
    print("-" * 92)

    summary = []
    for pat_id in ids:
        segments = read_data(
            pat_id, args.data_path,
            annotations_dir=cfg.get("annotations_dir", "annotations"),
            record_duration=cfg.get("record_duration", None),
            clock_drift=drift.get(pat_id),
        )
        for si, seg in enumerate(segments):
            # Same decimation the model sees, and the same robust scale.
            belts = {}
            for ch in ("Thorax", "Abdomen"):
                belts[ch] = _apply_chain(seg[ch], DECIMATION_CHAIN[ch])
            artifact_free = (1 - np.asarray(seg["SIGNAL-ARTIFACT"])).astype(bool)
            n_dec = min(len(belts["Thorax"]), len(belts["Abdomen"]))
            mask_dec = _mask_to_length(artifact_free, n_dec)

            scales = {
                ch: robust_stats(belts[ch][:n_dec], mask_dec)[1] for ch in belts
            }
            # Sum the two belts, as the model's effort channel does.
            summed = belts["Thorax"][:n_dec] + belts["Abdomen"][:n_dec]
            scale = max(scales["Thorax"] + scales["Abdomen"], 1e-9)

            during, before, reg = [], [], []
            ratio_used = TARGET_FREQ / BELT_FREQ
            for a, b in event_spans(seg["APNEA-CENTRAL"]):
                if (b - a) / TARGET_FREQ < MIN_EVENT_S:
                    continue
                da, db = int(a / ratio_used), int(b / ratio_used)
                pa = max(0, da - int(30 * BELT_FREQ))
                if db > n_dec or da <= pa:
                    continue
                during.append(cycles_per_min(summed[da:db], args.prominence * scale))
                before.append(cycles_per_min(summed[pa:da], args.prominence * scale))
                reg.append(rhythm_regularity(summed[da:db], args.prominence * scale))

            during = np.asarray([x for x in during if np.isfinite(x)])
            before = np.asarray([x for x in before if np.isfinite(x)])
            reg = np.asarray([x for x in reg if np.isfinite(x)])
            if len(during) < 5:
                continue

            md, mb = float(np.median(during)), float(np.median(before))
            mr = float(np.median(reg)) if len(reg) else np.nan
            # Thresholds, and why they are where they are. Below ~8/min a 10-12 s
            # event contains at most one or two peaks, which is indistinguishable
            # from residual wobble however regular it looks -- so it is called
            # flat regardless of CV. A machine call additionally REQUIRES the
            # intervals to be regular; rate alone does not identify a device.
            if md < 8:
                verdict = "belts flat -> clean apnea"
            elif md > 0.7 * mb:
                verdict = "still breathing -> label?"
            elif np.isfinite(mr) and mr < 0.3:
                verdict = "REGULAR slow rhythm -> machine"
            else:
                verdict = "some movement, irregular"
            summary.append((pat_id, si, len(during), md, mb, mr, verdict))
            reg_s = f"{mr:.2f}" if np.isfinite(mr) else "  - "
            print(f"{pat_id:>4} {si:>3} {len(during):>5} "
                  f"{md:>8.1f} (IQR {np.percentile(during, 25):>4.0f}-"
                  f"{np.percentile(during, 75):>4.0f}) "
                  f"{mb:>8.1f} (IQR {np.percentile(before, 25):>4.0f}-"
                  f"{np.percentile(before, 75):>4.0f})  CV={reg_s}  {verdict:<28}")

    if not summary:
        print("no block had enough scorable events")
        return

    print()
    print("=" * 92)
    md_all = np.array([s[3] for s in summary])
    mb_all = np.array([s[4] for s in summary])
    print(f"blocks measured: {len(summary)}")
    print(f"median rate DURING apnea : {np.median(md_all):.1f} /min "
          f"(range {md_all.min():.1f}-{md_all.max():.1f})")
    print(f"median rate BEFORE apnea : {np.median(mb_all):.1f} /min "
          f"(range {mb_all.min():.1f}-{mb_all.max():.1f})")
    print()
    for tag in ("belts flat -> clean apnea", "REGULAR slow rhythm -> machine",
                "still breathing -> label?", "some movement, irregular"):
        n = sum(1 for s in summary if s[6] == tag)
        if n:
            print(f"  {tag:<30} {n:3d}/{len(summary)} blocks")
    print()
    print("READING")
    print("  A block sitting at 10-30 /min during apneas, well below its own")
    print("  before-event rate, is being driven by something that is not the")
    print("  infant. That is the device rhythm of B6, and it both explains the")
    print("  ~1.0 belt ratio of section 1 and labels the block's device family.")
    print("  A block near its before-event rate means the infant kept breathing")
    print("  through an event scored as a central apnea, which is a question")
    print("  about the labels and should go to the original scorer.")


if __name__ == "__main__":
    main()
