"""VERIFICATION GUARD FOR THE B2/B3 NORMALISATION ABLATION.

Two things have to stay true for the 2x2 of B3 to mean anything, and both are
easy to break silently by editing `feature_extraction`:

1. THE BASELINE CELL MUST BE BIT-FOR-BIT THE PUBLISHED PIPELINE.
   `norm_per_window=True, norm_per_block=False` is the cell every previously
   reported CPAP-cohort number was produced under. If refactoring the
   normalisation moved it even in the last decimal, the other three cells could
   no longer be compared against the existing results and the whole ablation
   would have to be re-run from scratch. This script recomputes the ORIGINAL
   expressions inline -- literally the code as it stood before the change -- and
   asserts exact equality against what NeoNatal now produces.

   The trap it guards against is specific. The published `Thorax` channel is a
   NESTED standardisation:

       standardize(standardize(thorax) + standardize(abdomen))

   Wrapping that naively (normalising once, at the end) silently drops the two
   inner steps, and wrapping every channel uniformly silently double-standardises
   the single-channel ones. Neither shows up as an error; both change every
   number the model sees.

2. `DECIMATION_CHAIN` MUST MIRROR `feature_extraction`.
   Block statistics are computed on the whole block via `_apply_chain`, while
   the windows are decimated inside `feature_extraction`. If the two chains
   disagree, the statistics describe a different signal from the one they
   normalise -- a scale error that would look like a modelling result.

   The two are NOT expected to be equal sample-for-sample: decimating a whole
   block and then slicing differs from slicing and then decimating, at the
   filter edges. What must agree is the SAMPLING RATE (output length per input
   length) and the broad amplitude scale.

Run:  python scripts/check_norm_chains.py
"""
import os
import sys

import numpy as np
from scipy.signal import decimate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import (  # noqa: E402
    DECIMATION_CHAIN,
    NeoNatal,
    _apply_chain,
    block_norm_stats,
    normalize_range,
    standardize,
)

TARGET_FREQ = 200
SIGNAL_TYPES = ["CPAP", "Thorax", "HR", "PR", "SpO2", "PCO2"]
ADVERSE = ["APNEA-CENTRAL"]
CUTTER = ["SIGNAL-ARTIFACT"]
ANNOTATION_TYPES = [
    "APNEA-OBSTRUCTIVE", "APNEA-CENTRAL", "APNEA-MIXED",
    "HYPOPNEA-OBSTRUCTIVE", "HYPOPNEA-CENTRAL", "HYPOPNEA-MIXED",
    "DESAT", "ACTIVITY-MOVE", "SIGNAL-ARTIFACT", "SIGNAL-QUALITY-LOW",
]


def make_segment(n_seconds=1800, seed=0):
    """A synthetic block with a plausible respiratory rhythm and two apneas.

    Synthetic on purpose: this guard must run without the clinical data, so it
    can be executed on any machine and in CI.
    """
    rng = np.random.default_rng(seed)
    n = n_seconds * TARGET_FREQ
    t = np.arange(n) / TARGET_FREQ

    # ~55 breaths/min, the cohort baseline, with per-channel gain and drift so
    # that a per-block scale is actually different from a per-window one.
    breath = np.sin(2 * np.pi * (55 / 60) * t)
    # The two belts must NOT be scaled copies of one another. If they are, every
    # per-belt normalisation gives the same shape, their sum is the same shape,
    # and the "both on" cell collapses onto the baseline cell -- making the 2x2
    # look like a 1x3 for reasons that are a property of the test data and not
    # of the method. Real belts differ in gain, drift and phase (thoraco-
    # abdominal asynchrony), so the synthetic ones do too.
    abdo_breath = np.sin(2 * np.pi * (55 / 60) * t - 0.4)
    seg = {
        "CPAP": 5.0 + 1.5 * breath + 0.2 * rng.standard_normal(n),
        "Thorax": 70.0 * breath * (1 + 0.5 * np.sin(2 * np.pi * t / 600))
        + 5 * rng.standard_normal(n),
        "Abdomen": 55.0 * abdo_breath * (1 + 0.4 * np.sin(2 * np.pi * t / 950 + 1.1))
        + 4 * rng.standard_normal(n),
        "PR": 300.0 + 160.0 * np.sin(2 * np.pi * 2.5 * t) + 10 * rng.standard_normal(n),
        "HR": 160.0 + 8 * np.sin(2 * np.pi * t / 300),
        "SpO2": 96.0 + 2 * np.sin(2 * np.pi * t / 200),
        "PCO2": 45.0 + 3 * np.sin(2 * np.pi * t / 400),
    }
    for anno in ANNOTATION_TYPES:
        seg[anno] = np.zeros(n, dtype=int)
    # Two apneas, far apart, so both a positive and control windows exist.
    for onset_s in (600, 1200):
        s = onset_s * TARGET_FREQ
        seg["APNEA-CENTRAL"][s : s + 12 * TARGET_FREQ] = 1
    # One artifact span, to exercise the artifact-free mask.
    seg["SIGNAL-ARTIFACT"][100 * TARGET_FREQ : 130 * TARGET_FREQ] = 1
    return seg


def reference_window(ss, seg):
    """`feature_extraction` EXACTLY as it stood before the B2/B3 change."""
    lower_range, upper_range = -1.0, 1.0
    out = []
    for signal_type in SIGNAL_TYPES:
        if signal_type == "CPAP":
            out.append(standardize(decimate(decimate(seg["CPAP"][ss], 10), 4)))
        elif signal_type == "Thorax":
            out.append(
                standardize(
                    standardize(decimate(seg["Thorax"][ss][::4], 10))
                    + standardize(decimate(seg["Abdomen"][ss][::4], 10))
                )
            )
        elif signal_type == "HR":
            out.append(
                normalize_range(seg["HR"][ss][::200], 50.0, 240.0, lower_range, upper_range)
            )
        elif signal_type == "PR":
            out.append(standardize(decimate(decimate(seg["PR"][ss], 10), 4)))
        elif signal_type == "SpO2":
            out.append(
                normalize_range(
                    decimate(seg["SpO2"][ss][::100], 2), 60.0, 100.0, lower_range, upper_range
                )
            )
        elif signal_type == "PCO2":
            out.append(
                normalize_range(
                    decimate(seg["PCO2"][ss][::100], 2), 30.0, 70.0, lower_range, upper_range
                )
            )
    return out


def build(seg, **kwargs):
    return NeoNatal(
        "test",
        [seg],
        dataset_mode="list",
        signal_types=SIGNAL_TYPES,
        adverse_events=ADVERSE,
        cutter_events=CUTTER,
        time_window=6000,
        lag=3000,
        away=36000,
        **kwargs,
    )


def check_baseline_identical(seg):
    print("[1] baseline cell (window=on, block=off) vs. the original expressions")
    ds = build(seg)
    assert len(ds.time_window_df) > 0, "no windows built -- widen the synthetic segment"
    worst = 0.0
    for i in range(len(ds.time_window_df)):
        row = ds.time_window_df.iloc[i]
        got, want = row["sig"], reference_window(row["slice"], seg)
        assert len(got) == len(want)
        for channel, g, w in zip(SIGNAL_TYPES, got, want):
            assert g.shape == w.shape, f"{channel}: shape {g.shape} != {w.shape}"
            if not np.array_equal(g, w):
                worst = max(worst, float(np.max(np.abs(g - w))))
                raise AssertionError(
                    f"{channel} differs from the published pipeline "
                    f"(max |delta| = {np.max(np.abs(g - w)):.3e}) in window {i}"
                )
    print(f"    OK -- {len(ds.time_window_df)} windows x {len(SIGNAL_TYPES)} channels, "
          f"exactly equal (max |delta| = {worst:.1e})")
    return ds


def check_cells_differ(seg):
    print("[2] the four cells are genuinely different transforms")
    cells = {
        "window=on  block=off": dict(norm_per_window=True, norm_per_block=False),
        "window=off block=on ": dict(norm_per_window=False, norm_per_block=True),
        "window=on  block=on ": dict(norm_per_window=True, norm_per_block=True),
        "window=off block=off": dict(norm_per_window=False, norm_per_block=False),
    }
    sigs = {}
    for name, kw in cells.items():
        ds = build(seg, **kw)
        sigs[name] = ds.time_window_df.iloc[0]["sig"]
        thorax = sigs[name][SIGNAL_TYPES.index("Thorax")]
        print(f"    {name}: Thorax mean={thorax.mean():+.3f} sd={thorax.std():.3f} "
              f"ptp={np.ptp(thorax):.3f}")

    # window=on collapses whatever came before it onto zero mean / unit sd, so
    # the two window=on cells must agree closely -- that is the sanity check
    # that block normalisation is a pure affine map and nothing else.
    a = sigs["window=on  block=off"][SIGNAL_TYPES.index("CPAP")]
    b = sigs["window=on  block=on "][SIGNAL_TYPES.index("CPAP")]
    assert np.allclose(a, b, atol=1e-9), (
        "per-window standardisation should absorb the block affine map on a "
        "single channel; it did not -- the block step is not affine"
    )
    print("    OK -- window=on absorbs the block affine map on single channels")

    # The summed-effort channel is the exception, and the reason the 2x2 is a
    # real 2x2 rather than a 1x3: the two belts are normalised SEPARATELY before
    # summing, so block and window normalisation give them different relative
    # weights. No single affine map relates the two sums, and per-window
    # standardisation therefore cannot absorb the difference.
    ta = sigs["window=on  block=off"][SIGNAL_TYPES.index("Thorax")]
    tb = sigs["window=on  block=on "][SIGNAL_TYPES.index("Thorax")]
    assert not np.allclose(ta, tb, atol=1e-6), (
        "the both-on cell collapsed onto the baseline cell for Thorax -- with "
        "real belts it must not, so either the belt weighting is wrong or the "
        "test signals are scaled copies of each other"
    )
    print(f"    OK -- both-on cell is distinct for summed effort "
          f"(max |delta| = {np.max(np.abs(ta - tb)):.3f})")

    # ...and the block-only cell must NOT be zero-mean/unit-sd per window: that
    # is the entire point of B3, i.e. amplitude information survives.
    blk = sigs["window=off block=on "][SIGNAL_TYPES.index("Thorax")]
    assert abs(blk.std() - 1.0) > 1e-6 or abs(blk.mean()) > 1e-6, (
        "block-only cell looks per-window standardised -- amplitude information "
        "was destroyed anyway, which would make the ablation vacuous"
    )
    print("    OK -- block-only cell retains per-window amplitude information")


def check_chain_matches(seg):
    print("[3] DECIMATION_CHAIN mirrors feature_extraction")
    n = len(seg["CPAP"])
    for channel, chain in DECIMATION_CHAIN.items():
        if channel not in seg:
            continue
        dec = _apply_chain(seg[channel], chain)
        rate = len(dec) / (n / TARGET_FREQ)
        expected = 5.0  # every chain in the table lands on 5 Hz
        assert abs(rate - expected) < 0.05, (
            f"{channel}: chain gives {rate:.3f} Hz, feature_extraction gives {expected} Hz"
        )
        print(f"    {channel:8s} -> {rate:.3f} Hz  (expected {expected})")
    print("    OK -- all chains land on the rate feature_extraction produces")


def check_no_label_leakage(seg):
    print("[4] block statistics do not depend on the labels")
    mask = (1 - seg["SIGNAL-ARTIFACT"]).astype(bool)
    a = block_norm_stats(seg, SIGNAL_TYPES, artifact_free=mask)

    # Move every apnea somewhere else. If the statistics used the labels, they
    # would change; they must not.
    moved = dict(seg)
    moved["APNEA-CENTRAL"] = np.zeros_like(seg["APNEA-CENTRAL"])
    moved["APNEA-CENTRAL"][300 * TARGET_FREQ : 312 * TARGET_FREQ] = 1
    b = block_norm_stats(moved, SIGNAL_TYPES, artifact_free=mask)

    assert a == b, "block statistics changed when the labels moved -- LABEL LEAKAGE"
    print("    OK -- statistics unchanged when apnea annotations move")


def main():
    seg = make_segment()
    check_baseline_identical(seg)
    check_cells_differ(seg)
    check_chain_matches(seg)
    check_no_label_leakage(seg)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
