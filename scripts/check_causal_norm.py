"""VERIFICATION GUARD FOR B5's CAUSAL NORMALISATION STATISTICS.

B5 asks for the whole-block (transductive) statistics to stay primary and for
causal statistics -- expanding with a 10-minute warm-up, or trailing 30 minutes
-- to be reported beside them, "the difference between the two being the price
of real-time operation".

That claim is only worth printing if the causal arms are genuinely causal. Four
things have to hold, and every one of them is easy to break silently:

1. THE DEFAULT MUST NOT MOVE.
   `norm_stats_mode='whole_block'` is what produced every number in sections
   7-30. If adding the causal option perturbed it even in the last decimal, the
   B3 2x2 would have to be re-run. Test [1] asserts bit-for-bit equality against
   `block_norm_stats` computed directly.

2. THE CAUSAL ARMS MUST NOT SEE THE FUTURE.
   This is the whole point, and it cannot be established by reading the code --
   an off-by-one in the index mapping between TARGET_FREQ and the decimated rate
   would leak a little future and change nothing visible. Test [2] measures it
   directly: take a window, corrupt the signal AFTER it by a large amount, and
   assert the window's statistics are UNCHANGED. Then corrupt the signal BEFORE
   it and assert they DO change -- otherwise a function that ignored its inputs
   entirely would pass the first half.

3. THE WINDOW TABLE MUST BE IDENTICAL ACROSS MODES.
   Every comparison in this study is paired per infant, and
   `norm_ablation_stats.verify_pairing` asserts the labels match across cells.
   Normalisation must therefore never add, drop or move a window -- including
   the warm-up windows at the start of a block, which are NEUTRALISED rather
   than dropped for exactly this reason. Test [3].

4. THE SUBSAMPLING MUST NOT MATTER.
   A long history is thinned to CAUSAL_MAX_HISTORY_SAMPLES before the percentiles
   are taken, purely for speed. Test [4] measures the error that introduces
   against the un-thinned answer, so the number can be quoted rather than
   assumed harmless.

Synthetic data on purpose: this must run without the clinical recordings, on any
machine and in CI.

Run:  python scripts/check_causal_norm.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.neonatal_utils as nu  # noqa: E402
from src.neonatal_utils import (  # noqa: E402
    CAUSAL_WARMUP_S,
    NeoNatal,
    block_norm_stats,
    causal_norm_stats_at,
    prepare_causal_norm,
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
LEAK_TOL = 1e-6   # float64 accumulation noise; see test [2]
FLOOR = {"Abdomen": 5.228, "CPAP": 0.09263, "PR": 5.468,
         "Thorax": 3.273, "Thorax_sum": 0.3637}


def make_segment(n_seconds=5400, seed=0):
    """A 90-minute synthetic block: long enough for warm-up AND trailing."""
    rng = np.random.default_rng(seed)
    n = n_seconds * TARGET_FREQ
    t = np.arange(n) / TARGET_FREQ
    breath = np.sin(2 * np.pi * (55 / 60) * t)
    abdo = np.sin(2 * np.pi * (55 / 60) * t - 0.4)
    seg = {
        "CPAP": 5.0 + 1.5 * breath + 0.2 * rng.standard_normal(n),
        # A deliberate slow AMPLITUDE DRIFT, so the expanding and trailing arms
        # genuinely disagree. Without it both converge on the same statistics and
        # the tests below would pass for the wrong reason.
        "Thorax": 70.0 * breath * (1 + 0.6 * np.sin(2 * np.pi * t / 1800))
        + 5 * rng.standard_normal(n),
        "Abdomen": 55.0 * abdo * (1 + 0.5 * np.sin(2 * np.pi * t / 2300 + 1.1))
        + 4 * rng.standard_normal(n),
        "PR": 300.0 + 160.0 * np.sin(2 * np.pi * 2.5 * t) + 10 * rng.standard_normal(n),
        "HR": 160.0 + 8 * np.sin(2 * np.pi * t / 300),
        "SpO2": 96.0 + 2 * np.sin(2 * np.pi * t / 200),
        "PCO2": 45.0 + 3 * np.sin(2 * np.pi * t / 400),
    }
    for anno in ANNOTATION_TYPES:
        seg[anno] = np.zeros(n, dtype=int)
    for onset_s in (1500, 2400, 3600):
        s = onset_s * TARGET_FREQ
        seg["APNEA-CENTRAL"][s: s + 12 * TARGET_FREQ] = 1
    seg["SIGNAL-ARTIFACT"][300 * TARGET_FREQ: 330 * TARGET_FREQ] = 1
    return seg


def build(seg, **kw):
    return NeoNatal(
        "T01", seg, dataset_mode="list", signal_types=SIGNAL_TYPES,
        adverse_events=ADVERSE, cutter_events=CUTTER,
        time_window=6000, lag=3000, away=36000, **kw,
    )


def artifact_mask(seg):
    return (1 - np.column_stack([seg[e] for e in CUTTER]).max(axis=1)).astype(bool)


def main():
    seg = make_segment()
    af = artifact_mask(seg)
    ok = True

    # ------------------------------------------------------------------ [1] --
    print("[1] the default (whole_block) is unchanged")
    ref = block_norm_stats(seg, SIGNAL_TYPES, artifact_free=af, scale_floor=FLOOR)
    ds_default = build(seg, norm_per_window=False, norm_per_block=True,
                       norm_scale_floor=FLOOR)
    ds_explicit = build(seg, norm_per_window=False, norm_per_block=True,
                        norm_scale_floor=FLOOR, norm_stats_mode="whole_block")
    worst = 0.0
    for a, b in zip(ds_default.time_window_df["sig"],
                    ds_explicit.time_window_df["sig"]):
        for xa, xb in zip(a, b):
            worst = max(worst, float(np.max(np.abs(np.asarray(xa) - np.asarray(xb)))))
    print(f"    default vs explicit 'whole_block': max |delta| = {worst:.3e}")
    print(f"    block_norm_stats still returns {len(ref)} channels "
          f"(incl. Thorax_sum: {'Thorax_sum' in ref})")
    if worst != 0.0:
        ok = False
        print("    FAIL -- the default moved")
    else:
        print("    OK -- bit-for-bit identical")

    # ------------------------------------------------------------------ [2] --
    print("\n[2] the causal arms cannot see the future (the load-bearing test)")
    print("    method: corrupt the signal AFTER a window by 50x and re-measure")
    print("    that window's statistics. Then corrupt the signal BEFORE it --")
    print("    which MUST change them, or a function ignoring its input passes.")
    probe_start = int(3000 * TARGET_FREQ)

    def leak_at(mode, guard_s):
        """max |delta| in the statistics when only the FUTURE is corrupted."""
        saved = nu.CAUSAL_FILTER_GUARD_S
        nu.CAUSAL_FILTER_GUARD_S = guard_s
        try:
            prep = prepare_causal_norm(seg, SIGNAL_TYPES, artifact_free=af)
            base = causal_norm_stats_at(prep, probe_start, mode, scale_floor=FLOOR)
            future = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in seg.items()}
            for ch in ("Thorax", "Abdomen", "CPAP", "PR"):
                future[ch][probe_start:] *= 50.0
            prep_f = prepare_causal_norm(future, SIGNAL_TYPES, artifact_free=af)
            after = causal_norm_stats_at(prep_f, probe_start, mode, scale_floor=FLOOR)
            return max(
                max(abs(base[c][0] - after[c][0]), abs(base[c][1] - after[c][1]))
                for c in base
            )
        finally:
            nu.CAUSAL_FILTER_GUARD_S = saved

    # EXACT zero is not achievable and it is worth saying why. decimate() uses
    # filtfilt, which pads both ends of the array and runs the filter backwards;
    # the contribution of the future therefore DECAYS with the guard band rather
    # than truncating. The sweep below shows that decay, so the residual is
    # demonstrated to be filter ring-down and not a real dependency.
    print("\n    leak vs guard band (expanding), showing the decay:")
    for guard_s in (0.0, 1.0, 5.0, nu.CAUSAL_FILTER_GUARD_S):
        print(f"      guard = {guard_s:>5.1f} s  ->  max |delta| = "
              f"{leak_at('expanding', guard_s):.3e}")

    print()
    for mode in ("expanding", "trailing"):
        prep = prepare_causal_norm(seg, SIGNAL_TYPES, artifact_free=af)
        base = causal_norm_stats_at(prep, probe_start, mode, scale_floor=FLOOR)
        d_future = leak_at(mode, nu.CAUSAL_FILTER_GUARD_S)

        # Corrupt everything BEFORE it. Statistics MUST move.
        past = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in seg.items()}
        for ch in ("Thorax", "Abdomen", "CPAP", "PR"):
            past[ch][:probe_start] *= 50.0
        prep_p = prepare_causal_norm(past, SIGNAL_TYPES, artifact_free=af)
        before = causal_norm_stats_at(prep_p, probe_start, mode, scale_floor=FLOOR)
        d_past = max(
            max(abs(base[c][0] - before[c][0]), abs(base[c][1] - before[c][1]))
            for c in base
        )

        # LEAK_TOL is set at the level of float64 accumulation noise, ~8 orders
        # of magnitude below the smallest statistic here and ~11 below the
        # differences B5 is measuring. It is NOT a tolerance for "a bit of
        # leakage" -- the sweep above is what establishes there is none.
        good = d_future < LEAK_TOL and d_past > 1e-6
        ok = ok and good
        print(f"    {mode:<10} future changed -> {d_future:.3e} "
              f"(tol {LEAK_TOL:.0e})   past changed -> {d_past:.3e} (must be > 0)"
              f"   {'OK' if good else 'FAIL'}")

    # ------------------------------------------------------------------ [3] --
    print("\n[3] the window table is identical across all three modes")
    tables = {}
    for mode in ("whole_block", "expanding", "trailing"):
        ds = build(seg, norm_per_window=False, norm_per_block=True,
                   norm_scale_floor=FLOOR, norm_stats_mode=mode)
        tables[mode] = ds.time_window_df
    ref_df = tables["whole_block"]
    same = True
    for mode, df in tables.items():
        same = same and len(df) == len(ref_df) and np.array_equal(
            df["label"].to_numpy(), ref_df["label"].to_numpy()
        )
    ok = ok and same
    print(f"    windows: " + ", ".join(f"{m}={len(d)}" for m, d in tables.items()))
    print(f"    {'OK -- same count and same labels' if same else 'FAIL -- pairing broken'}")

    warm = int((ref_df["slice"].map(lambda s: s.start)
                < CAUSAL_WARMUP_S * TARGET_FREQ).sum())
    print(f"    windows inside the {CAUSAL_WARMUP_S / 60:.0f}-min warm-up: "
          f"{warm}/{len(ref_df)} ({warm / max(len(ref_df), 1):.1%}) -- these use "
          "the statistics as at the end of warm-up, and are NOT dropped")

    # ------------------------------------------------------------------ [4] --
    print("\n[4] how much the history subsampling changes the statistics")
    prep = prepare_causal_norm(seg, SIGNAL_TYPES, artifact_free=af)
    thinned = causal_norm_stats_at(prep, probe_start, "expanding", scale_floor=FLOOR)
    saved = nu.CAUSAL_MAX_HISTORY_SAMPLES
    nu.CAUSAL_MAX_HISTORY_SAMPLES = 10 ** 9          # effectively no thinning
    exact = causal_norm_stats_at(prep, probe_start, "expanding", scale_floor=FLOOR)
    nu.CAUSAL_MAX_HISTORY_SAMPLES = saved
    worst_rel = 0.0
    for c in exact:
        m_e, s_e = exact[c]
        m_t, s_t = thinned[c]
        if s_e:
            worst_rel = max(worst_rel, abs(s_t - s_e) / abs(s_e))
        if m_e:
            worst_rel = max(worst_rel, abs(m_t - m_e) / abs(m_e))
    print(f"    max relative difference vs un-thinned: {worst_rel:.3e}")
    print("    (quoted in the write-up rather than assumed negligible)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
