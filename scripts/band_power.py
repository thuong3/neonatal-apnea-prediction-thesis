"""RESPIRATORY BAND POWER PER CHANNEL, OVER THE WHOLE COHORT.

Section 3.2 of the thesis asks whether the CPAP mask pressure carries
any respiratory information at all, since the flow generator holds it near a set
pressure and it could in principle be nearly constant. That check was run once,
on a single five-minute segment of patient 001, and its numbers (6.4 % for the
mask pressure, 9.6 % for the thorax belt, 45.6 % for the plethysmogram) are
quoted in the thesis. One segment of one infant is thin support for a statement
about a channel across fifteen recordings, so this script measures the same
quantity over every infant and every analysis block.

WHAT IS MEASURED
    The fraction of a channel's power that falls in the neonatal breathing band,
    0.5 to 2 Hz, which spans roughly 30 to 120 breaths per minute.

THREE CHOICES THAT AFFECT THE NUMBER, MADE EXPLICIT
  * Native rate, not 200 Hz. `read_data` brings every channel to a common
    200 Hz by sample-and-hold, which is a staircase: it adds imaging above the
    channel's own Nyquist frequency and would inflate the denominator by a
    different amount for each channel, since their native rates differ. Each
    channel is therefore decimated back to its native rate by taking every
    k-th sample, which inverts sample-and-hold exactly.
  * The mean is removed. The mask pressure sits on a set pressure of several
    cmH2O, so with the mean left in, the DC term would dominate every channel's
    total power and the comparison would measure offsets instead of rhythm.
  * Median over windows, then over infants. Five-minute windows are scored
    separately and summarised by their median, so that a few artifact-laden
    stretches cannot move the result.

THE SPLIT BY VENTILATION MODE, AND WHY IT DECIDES THE CLAIM
    Two of every infant's four blocks are synchronised ventilation at about 10
    inflations per minute (`scripts/detect_ventilation_mode.py`, 15/15 infants,
    no ambiguous block). That fundamental is 0.167 Hz, below the band, but a
    non-sinusoidal inflation puts harmonics at 0.33, 0.5, 0.67 Hz and upward,
    and the third harmonic onwards falls inside it. If the mask pressure's
    in-band share came from the machine rather than from the infant, it would be
    markedly higher in the NIPPV blocks than in the CPAP blocks. The two are
    therefore reported separately. Block index is taken to be the order in which
    `read_data` returns the analysis spans, which is the order
    `detect_ventilation_mode.py` numbers them in.

Usage:
    python scripts/band_power.py
    python scripts/band_power.py --data <path to dataset_brainimmaturity>
"""
import argparse
import os
import sys

import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import TARGET_FREQ, read_data  # noqa: E402

DEFAULT_DATA = ("data/"
                "dataset_brainimmaturity")
IDS = [f"{i:03d}" for i in range(1, 16)]

# Native sampling rates of the exported channels, from section 3.1 of the
# thesis. Only the channels that could carry a breathing rhythm are measured.
NATIVE_HZ = {"CPAP": 20, "Thorax": 10, "Abdomen": 10, "PR": 20}

BAND = (0.5, 2.0)      # neonatal breathing band, Hz
WINDOW_S = 300         # five minutes, matching the original check


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--modes", default="outputs/ventilation_mode/modes.csv",
                   help="written by scripts/detect_ventilation_mode.py")
    p.add_argument("--out", default="outputs/band_power/band_power.txt")
    return p.parse_args()


def load_modes(path):
    """{(patient, block index): "NIPPV" | "CPAP"}, or {} if unavailable."""
    if not os.path.exists(path):
        return {}
    modes = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        i_pat, i_blk, i_mode = (header.index("pat"), header.index("block"),
                                header.index("mode"))
        for line in fh:
            f = line.rstrip("\n").split(",")
            if len(f) > max(i_pat, i_blk, i_mode):
                modes[(f[i_pat], int(f[i_blk]))] = f[i_mode]
    return modes


def band_fraction(x, fs):
    """Fraction of the power of `x` that lies in BAND, mean removed."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.std(x) == 0:
        return np.nan
    x = x - x.mean()
    nperseg = min(len(x), fs * 60)
    if nperseg < fs * 10:                     # too short to resolve 0.5 Hz
        return np.nan
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    total = np.trapz(pxx, f)
    if total <= 0:
        return np.nan
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return float(np.trapz(pxx[sel], f[sel]) / total)


def main():
    args = parse_args()
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    modes = load_modes(args.modes)
    per_infant = {ch: [] for ch in NATIVE_HZ}
    by_mode = {m: {ch: [] for ch in NATIVE_HZ} for m in ("NIPPV", "CPAP")}
    n_windows = {ch: 0 for ch in NATIVE_HZ}

    for pid in IDS:
        segments = read_data(pid, args.data)
        per_channel = {ch: [] for ch in NATIVE_HZ}
        per_channel_mode = {m: {ch: [] for ch in NATIVE_HZ}
                            for m in ("NIPPV", "CPAP")}
        for blk, seg in enumerate(segments):
            mode = modes.get((pid, blk))
            for ch, fs in NATIVE_HZ.items():
                if ch not in seg:
                    continue
                step = TARGET_FREQ // fs       # invert the sample-and-hold
                x = np.asarray(seg[ch], dtype=float)[::step]
                w = WINDOW_S * fs
                for start in range(0, len(x) - w + 1, w):
                    v = band_fraction(x[start:start + w], fs)
                    if np.isfinite(v):
                        per_channel[ch].append(v)
                        if mode in per_channel_mode:
                            per_channel_mode[mode][ch].append(v)
        for ch in NATIVE_HZ:
            vals = per_channel[ch]
            n_windows[ch] += len(vals)
            per_infant[ch].append(float(np.median(vals)) if vals else np.nan)
            for m in by_mode:
                v = per_channel_mode[m][ch]
                by_mode[m][ch].append(float(np.median(v)) if v else np.nan)

    out()
    out(f"Power in {BAND[0]}-{BAND[1]} Hz as a fraction of total power, mean "
        f"removed,")
    out(f"median over {WINDOW_S // 60}-minute windows within an infant, then "
        f"over infants.")
    out()
    out(f"{'channel':<10} {'native':>7} {'median':>8} {'IQR':>16} "
        f"{'min':>7} {'max':>7} {'windows':>9}")
    out("-" * 68)
    for ch, fs in NATIVE_HZ.items():
        v = np.asarray(per_infant[ch], dtype=float)
        v = v[np.isfinite(v)]
        q1, q3 = np.percentile(v, [25, 75])
        out(f"{ch:<10} {fs:>6} Hz {np.median(v) * 100:>7.1f}% "
            f"{q1 * 100:>6.1f}-{q3 * 100:<5.1f}% {v.min() * 100:>6.1f}% "
            f"{v.max() * 100:>6.1f}% {n_windows[ch]:>9}")
    if modes:
        out()
        out("Split by ventilation mode, paired within infant. If the mask")
        out("pressure's in-band share came from the machine, NIPPV would exceed")
        out("CPAP; the difference column is NIPPV minus CPAP over infants.")
        out()
        out(f"{'channel':<10} {'NIPPV':>8} {'CPAP':>8} {'difference':>12} "
            f"{'higher in':>12}")
        out("-" * 54)
        for ch in NATIVE_HZ:
            a = np.asarray(by_mode["NIPPV"][ch], dtype=float)
            b = np.asarray(by_mode["CPAP"][ch], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            out(f"{ch:<10} {np.median(a[ok]) * 100:>7.1f}% "
                f"{np.median(b[ok]) * 100:>7.1f}% "
                f"{np.median(d) * 100:>+11.1f}% "
                f"{int((d > 0).sum()):>5}/{int(ok.sum())} infants")
    out()
    out("Per infant (%):")
    out(f"{'id':<5}" + "".join(f"{ch:>10}" for ch in NATIVE_HZ))
    for i, pid in enumerate(IDS):
        row = "".join(f"{per_infant[ch][i] * 100:>10.1f}" for ch in NATIVE_HZ)
        out(f"{pid:<5}{row}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
