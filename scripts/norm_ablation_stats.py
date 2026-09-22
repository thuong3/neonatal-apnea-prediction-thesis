"""B3: THE PRE-SPECIFIED ANALYSIS OF THE NORMALISATION 2x2.

B3 asks: demote per-window standardisation to an ablation arm, make per-block
robust normalisation the primary pipeline, run the full 2x2 of
{per-window on/off} x {per-block on/off}, and REPORT ALL FOUR CELLS.

    norm_window   per-window only   the published baseline
    norm_block    per-block only    the proposed primary
    norm_both     both
    norm_none     neither           the control

It also reports ONE OPTIONAL, NON-FACTORIAL cell when its runs are present:

    effort_baseline_channel   B4's causal trailing-baseline channel (7 inputs)

B4 is not a cell of the 2x2 and never can be -- it changes the architecture from
6 input channels to 7, so it is not comparable on normalisation alone. It is
analysed here because it is paired on the SAME WINDOWS as the four cells and its
correct control, norm_none, is one of them: same pipeline, same windows, no
normalisation, six channels instead of seven. It is excluded from the main
effects, which are only defined over the complete factorial.

This script produces the analysis exactly as frozen in PRE_SPECIFICATION.md
BEFORE the runs. Nothing here is chosen after seeing a number.

What the pre-specification requires
-----------------------------------
1.1  The comparison is a DISTRIBUTION OF PAIRED PER-INFANT DIFFERENCES (G4),
     never two averages. Per-patient AuROC is averaged over training seeds
     first, then a Wilcoxon signed-rank runs over the 15 infants.

     A difference is called REAL only if it passes BOTH parts:
       (a) Wilcoxon over the paired per-infant values, alpha = 0.05, and
       (b) it exceeds the seed-to-seed spread of that same contrast.
     Part (b) is what stops a tiny but consistent difference being read as an
     effect when it is smaller than the noise the training procedure injects.

1.2  Primary summary is the COUNT-WEIGHTED mean, sum(auc*n_pos)/sum(n_pos),
     because target counts run 22-306 across infants and the per-patient score
     is anti-correlated with the count (rho = -0.79). The plain mean is printed
     beside it. Both are shown for every cell and contrast.

3.   The pre-specified threshold that would change the conclusion: the
     norm_block cell reaching a count-weighted per-patient AuROC above 0.65,
     surviving 10 seeds and the Wilcoxon. Checked explicitly at the end.

The patient 009 question -- read this before quoting a number
-------------------------------------------------------------
PRE_SPECIFICATION.md 1.2 excludes patient 009 from PER-PATIENT claims (keeping
it in pooled ones), on the grounds that 57% of its marks sit on a dead Abdomen
belt (Section 3.4 of the thesis).

That reason was WITHDRAWN on 25 August 2026, before any cell of this ablation
was run: scripts/check_dead_belt_marks.py showed those marks are real events --
17 of 18 reach a >=3% desaturation against 70% cohort-wide. An unfalsifiable
mark is not a false one.

Because the exclusion is pre-specified but its stated reason no longer holds,
this script refuses to choose for you: it reports EVERY per-patient number BOTH
WAYS, with and without 009. If the two disagree on any contrast, that is
reported loudly. The write-up must state which is primary and why.

Usage
-----
    python scripts/norm_ablation_stats.py --result_path results
    python scripts/norm_ablation_stats.py --result_path ... --seeds 0 1 2 3 4

Missing seeds are skipped with a warning, so this can be run against a sweep
that is still in progress -- but a partial result is NOT the B3 deliverable.
"""
import argparse
import itertools
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.misc_utils import read_pickle_file  # noqa: E402

# The 2x2 itself. All four are REQUIRED: the main-effects section is only
# defined over the complete factorial.
CELLS = ("norm_window", "norm_block", "norm_both", "norm_none")
# Cells that are not part of the 2x2 and are reported only if their runs exist.
# B4's channel changes the network from 6 inputs to 7, so it can never be a cell
# of the factorial -- its runs are not comparable on architecture. It is carried
# here anyway because it is paired on the SAME WINDOWS as every other cell and
# its control (norm_none) is one of them, so the one contrast that matters is a
# valid paired test and belongs in the same analysis.
#
# B5's two causal cells are optional for a different reason: they ARE the
# per-block transform, computed from history instead of the whole block, so they
# are architecturally identical to norm_block and differ only in where the
# median and IQR come from. They are not factorial cells either -- the 2x2
# crosses per-window against per-block, not statistic-source against anything --
# so they are reported beside it rather than inside it.
OPTIONAL_CELLS = (
    "effort_baseline_channel",
    "norm_block_causal_expanding",
    "norm_block_causal_trailing",
    "hr_spo2_dev_channels",
)
ALL_CELLS = CELLS + OPTIONAL_CELLS
LABEL = {
    "norm_window": "per-window only (published baseline)",
    "norm_block": "per-block only (PROPOSED PRIMARY)",
    "norm_both": "both",
    "norm_none": "neither (control)",
    "effort_baseline_channel": "B4: +causal trailing-baseline channel (7 ch)",
    "norm_block_causal_expanding": "B5: causal stats, expanding (10-min warm-up)",
    "norm_block_causal_trailing": "B5: causal stats, trailing 30 min",
    "hr_spo2_dev_channels": "B2.7: +HR/SpO2 deviation channels (8 ch)",
}
# The contrasts that matter, as (a, b) meaning a - b. A contrast whose cells are
# not both present is skipped rather than fatal.
CONTRASTS = (
    ("norm_block", "norm_window", "PRIMARY -- B3's hypothesis"),
    ("norm_none", "norm_window", "does removing per-window help at all?"),
    ("norm_both", "norm_window", "does per-block add anything on top?"),
    ("norm_none", "norm_block", "is per-block better than no normalisation?"),
    # B4. norm_none is the RIGHT control and the only valid one: same pipeline,
    # same windows, same normalisation (none), six channels instead of seven, so
    # the difference is the channel and nothing else. Comparing against
    # norm_window instead would confound the channel with the -0.0252 that
    # per-window standardisation costs.
    ("effort_baseline_channel", "norm_none",
     "B4 -- does the trailing-baseline channel add anything?"),
    # B5. norm_block is the RIGHT control here and norm_none would be wrong: the
    # causal arms ARE per-block normalisation, differing only in whether the
    # median and IQR come from the whole block or from history. Against
    # norm_block the difference is the statistic source and nothing else, which
    # is exactly "the price of real-time operation" B5 asks to be reported.
    ("norm_block_causal_expanding", "norm_block",
     "B5 -- price of real-time operation, expanding"),
    ("norm_block_causal_trailing", "norm_block",
     "B5 -- price of real-time operation, trailing 30 min"),
    ("norm_block_causal_trailing", "norm_block_causal_expanding",
     "B5 -- does a 30-min memory beat an expanding one?"),
    # B2 bullet 7. norm_block is the control: same pipeline, same windows, same
    # normalisation and floor, six channels instead of eight, so the difference
    # is the two added deviation channels and nothing else.
    ("hr_spo2_dev_channels", "norm_block",
     "B2.7 -- do the HR/SpO2 deviation channels add anything?"),
)
PRESPEC_THRESHOLD = 0.65   # PRE_SPECIFICATION.md section 3
EXCLUDE_PRESPEC = "009"    # PRE_SPECIFICATION.md 1.2 -- see the docstring


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result_path", default="results")
    p.add_argument("--experiment", default="cpap_norm_ablation")
    p.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_all(exp_dir, seeds):
    """{cell: {seed: results}}, skipping anything missing."""
    runs, missing = {c: {} for c in ALL_CELLS}, []
    for cell, seed in itertools.product(ALL_CELLS, seeds):
        tag = f"{cell}_s{seed}"
        if not os.path.exists(os.path.join(exp_dir, f"{tag}_results.pkl")):
            # An absent OPTIONAL cell is not "missing" -- it simply was not run.
            if cell in CELLS:
                missing.append(tag)
            continue
        runs[cell][seed] = read_pickle_file(f"{tag}_results.pkl", exp_dir)
    return runs, missing


def verify_pairing(runs, ids, cells):
    """Every cell must see IDENTICAL windows and labels.

    Normalisation changes the signal values, never the window table -- so if the
    labels differ between cells the comparison is not paired and no signed-rank
    test over infants is valid. This is the one assumption that would silently
    invalidate everything below, so it is asserted rather than assumed.
    """
    ref_cell, ref_seed = next((c, s) for c in cells for s in runs[c])
    ref = runs[ref_cell][ref_seed]
    for cell in cells:
        for seed, res in runs[cell].items():
            assert sorted(res) == ids, f"{cell}_s{seed}: patient set differs"
            for pid in ids:
                assert np.array_equal(res[pid]["ys"], ref[pid]["ys"]), (
                    f"{cell}_s{seed} patient {pid}: labels differ from "
                    f"{ref_cell}_s{ref_seed} -- the cells are NOT paired"
                )


def per_patient(res, ids):
    return np.array([roc_auc_score(res[p]["ys"], res[p]["scores"]) for p in ids])


def pooled(res, ids):
    ys = np.concatenate([res[p]["ys"] for p in ids])
    sc = np.concatenate([res[p]["scores"] for p in ids])
    return roc_auc_score(ys, sc)


def cw(auc, w):
    """Count-weighted mean -- PRE_SPECIFICATION 1.2's primary summary."""
    return float(np.sum(auc * w) / np.sum(w))


def summarise(auc, w):
    return cw(auc, w), float(np.mean(auc))


def report_contrast(a, b, why, seed_auc, ids, w, alpha, tag):
    """One paired contrast under the two-part rule of PRE_SPECIFICATION 1.1."""
    # Seed-average per patient FIRST, then test over infants (Vetter's protocol).
    mean_a = np.mean([seed_auc[a][s] for s in sorted(seed_auc[a])], axis=0)
    mean_b = np.mean([seed_auc[b][s] for s in sorted(seed_auc[b])], axis=0)
    d = mean_a - mean_b

    # Part (b): the seed-to-seed spread of this same contrast.
    shared = sorted(set(seed_auc[a]) & set(seed_auc[b]))
    per_seed = [cw(seed_auc[a][s] - seed_auc[b][s], w) for s in shared]
    spread = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")

    d_cw, d_mean = cw(d, w), float(np.mean(d))
    try:
        stat, p = wilcoxon(d)
    except ValueError:      # all-zero differences
        stat, p = float("nan"), 1.0

    passes_w = p < alpha
    passes_s = np.isfinite(spread) and abs(d_cw) > spread
    verdict = ("REAL" if (passes_w and passes_s)
               else "not called" + ("" if passes_w else " (Wilcoxon)")
               + ("" if passes_s else " (within seed spread)"))

    print(f"  {a} - {b}   [{tag}]   {why}")
    print(f"    count-weighted delta {d_cw:+.4f} | plain mean {d_mean:+.4f}")
    print(f"    Wilcoxon over {len(d)} infants: W={stat:.1f}, p={p:.4f} "
          f"({'passes' if passes_w else 'fails'} alpha={alpha})")
    print(f"    seed-to-seed spread of this contrast (n={len(per_seed)}): "
          f"{spread:.4f}  -> |delta| {'>' if passes_s else '<='} spread")
    print(f"    positive in {int((d > 0).sum())}/{len(d)} infants")
    print(f"    VERDICT: {verdict}")
    return dict(contrast=f"{a}-{b}", d_cw=d_cw, p=p, spread=spread,
                verdict=verdict, real=(passes_w and passes_s))


def main():
    args = parse_args()
    exp_dir = os.path.join(args.result_path, args.experiment)
    print("B3 -- normalisation 2x2, pre-specified analysis\n")
    print(f"results: {exp_dir}")

    runs, missing = load_all(exp_dir, args.seeds)
    have = {c: sorted(runs[c]) for c in ALL_CELLS}
    # The 2x2 is required; optional cells appear only when they have been run.
    cells = list(CELLS) + [c for c in OPTIONAL_CELLS if have[c]]
    for c in cells:
        print(f"  {c:<24} seeds {have[c] if have[c] else 'NONE'}")
    for c in OPTIONAL_CELLS:
        if not have[c]:
            print(f"  {c:<24} not run -- skipped")
    if missing:
        print(f"  ! {len(missing)} run(s) missing: {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    if not all(have[c] for c in CELLS):
        sys.exit("Not every cell has at least one seed -- cannot compare.")
    n_seeds = min(len(have[c]) for c in cells)
    if n_seeds < len(args.seeds):
        print(f"\n  *** PARTIAL: {n_seeds} complete seed(s) of {len(args.seeds)}.")
        print("  *** This is NOT the B3 deliverable. Re-run when the sweep ends.")

    ids = sorted(next(iter(runs["norm_window"].values())))
    verify_pairing(runs, ids, cells)
    print(f"\npairing verified: all cells see identical windows and labels "
          f"({len(ids)} infants)")

    n_pos = np.array([int(np.sum(next(iter(runs['norm_window'].values()))[p]["ys"]))
                      for p in ids], dtype=float)
    seed_auc = {c: {s: per_patient(runs[c][s], ids) for s in runs[c]} for c in cells}

    keep = {"all 15 infants": np.ones(len(ids), bool)}
    if EXCLUDE_PRESPEC in ids:
        m = np.array([p != EXCLUDE_PRESPEC for p in ids])
        keep[f"excluding {EXCLUDE_PRESPEC} (pre-spec 1.2)"] = m

    # ---------------------------------------------------------------- cells --
    print("\n" + "=" * 78)
    print("[1] ALL CELLS  (B3: 'report all four cells'; optional cells follow)")
    for lbl, mask in keep.items():
        print(f"\n  -- {lbl} --")
        print(f"  {'cell':<12} {'count-weighted':>15} {'plain mean':>12} "
              f"{'pooled':>8} {'seed SD of mean':>17}")
        for c in cells:
            avg = np.mean([seed_auc[c][s] for s in sorted(seed_auc[c])], axis=0)
            w, m = summarise(avg[mask], n_pos[mask])
            pool = float(np.mean([pooled(runs[c][s], ids) for s in sorted(runs[c])]))
            per_seed_mean = [cw(seed_auc[c][s][mask], n_pos[mask]) for s in sorted(seed_auc[c])]
            sd = float(np.std(per_seed_mean, ddof=1)) if len(per_seed_mean) > 1 else float("nan")
            print(f"  {c:<12} {w:>15.4f} {m:>12.4f} {pool:>8.4f} {sd:>17.4f}"
                  f"   {LABEL[c]}")

    # ------------------------------------------------------------ contrasts --
    print("\n" + "=" * 78)
    print("[2] PAIRED PER-INFANT CONTRASTS  (pre-spec 1.1: never two averages)")
    results = {}
    for lbl, mask in keep.items():
        print(f"\n  -- {lbl} --")
        sa = {c: {s: v[mask] for s, v in seed_auc[c].items()} for c in cells}
        results[lbl] = [report_contrast(a, b, why, sa, ids, n_pos[mask],
                                        args.alpha, lbl)
                        for a, b, why in CONTRASTS
                        if a in cells and b in cells]

    # ------------------------------------------------------- 2x2 main effects --
    print("\n" + "=" * 78)
    print("[3] 2x2 MAIN EFFECTS  (averaged over the other factor)")
    for lbl, mask in keep.items():
        avg = {c: np.mean([seed_auc[c][s] for s in sorted(seed_auc[c])], axis=0)[mask]
               for c in CELLS}   # the 2x2 only -- B4 is not a factorial cell
        w = n_pos[mask]
        win = (avg["norm_window"] + avg["norm_both"]) / 2 - \
              (avg["norm_none"] + avg["norm_block"]) / 2
        blk = (avg["norm_block"] + avg["norm_both"]) / 2 - \
              (avg["norm_none"] + avg["norm_window"]) / 2
        inter = (avg["norm_both"] + avg["norm_none"]) - \
                (avg["norm_window"] + avg["norm_block"])
        print(f"\n  -- {lbl} --")
        for name, v in (("per-window standardisation ON", win),
                        ("per-block normalisation ON", blk),
                        ("interaction", inter)):
            try:
                _, p = wilcoxon(v)
            except ValueError:
                p = 1.0
            print(f"    {name:<32} {cw(v, w):+.4f} (count-weighted), p={p:.4f}")

    # ------------------------------------------------------------ threshold --
    print("\n" + "=" * 78)
    print("[4] THE PRE-SPECIFIED THRESHOLD  (PRE_SPECIFICATION.md section 3)")
    print(f"    'the norm_block cell reaching a count-weighted per-patient AuROC")
    print(f"     above {PRESPEC_THRESHOLD}, surviving 10 seeds and the Wilcoxon'")
    for lbl, mask in keep.items():
        avg = np.mean([seed_auc["norm_block"][s] for s in sorted(seed_auc["norm_block"])],
                      axis=0)
        got = cw(avg[mask], n_pos[mask])
        met = got > PRESPEC_THRESHOLD
        print(f"    {lbl:<34} norm_block = {got:.4f}  -> "
              f"{'MET' if met else 'NOT met'}")

    # --------------------------------------------------------------- C6 check --
    print("\n" + "=" * 78)
    print("[5] C6 SANITY: does per-patient score still track the target count?")
    avg = np.mean([seed_auc["norm_block"][s] for s in sorted(seed_auc["norm_block"])],
                  axis=0)
    rho, p_rho = spearmanr(n_pos, avg)
    print(f"    norm_block: Spearman rho={rho:+.3f} (p={p_rho:.4f}) vs n_positives")
    print(f"    (pre-spec 1.2 measured -0.79/-0.83; this is why the count-weighted")
    print(f"     mean is primary and the plain mean is only reported beside it)")

    # ------------------------------------------------------------ disagreement --
    if len(results) > 1:
        a, b = list(results)
        diff = [x["contrast"] for x, y in zip(results[a], results[b])
                if x["real"] != y["real"]]
        print("\n" + "=" * 78)
        print("[6] DOES EXCLUDING 009 CHANGE ANY VERDICT?")
        if diff:
            print(f"    !! YES -- verdicts differ for: {', '.join(diff)}")
            print("    The write-up MUST state which set is primary and why.")
        else:
            print("    No. Every contrast reaches the same verdict either way,")
            print("    so the 009 question does not affect B3's conclusion.")


if __name__ == "__main__":
    main()
