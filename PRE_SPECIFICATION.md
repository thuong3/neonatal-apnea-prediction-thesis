# Pre-specification for the remaining runs

Item **H1** of the re-analysis to-do list: *"Pre-specify and freeze the plan.
Record which of the section D diagnostics decided the direction taken. The
honest set of possible conclusions includes 'the four remaining modalities do
not support 15-second prediction of central apnea in preterm infants at 28–34
weeks corrected gestational age', and pre-specification is what protects that
result from looking like a failure."*

This file is written **before** the runs it describes. Its whole value is the
date: it is what makes "we decided in advance" checkable rather than asserted.

**Written 23 August 2026.** Nothing below has been run at the time of writing
except where explicitly marked *(already measured)*.

---

## 1. The decisions being frozen

### 1.1 Per-infant contribution to training (F3) — **decided: equal**

The published sampler balances the two classes inside each infant but not the
infants against each other. Measured on this cohort (`scripts/check_sampler_equal.py`):

| | published (arm A) | equal (arm B) |
|---|---|---|
| largest infant's share of an epoch | **18.3%** (patient 015) | 6.7% |
| smallest infant's share | **1.3%** (patient 012) | 6.7% |
| spread | **14×** | 1× |
| top 3 infants | **45% of all training data** | 20% |

**Primary: arm B, equal contribution** (`+experiment=sampler_equal`).

**Reason, and it is not a performance reason.** The reported metric is a mean
over infants, in which every infant counts once. Arm A optimises a different
weighting from the one being reported; arm B makes training and evaluation
agree. This argument would hold even if arm B scored worse.

**Both arms will be run and both reported**, as a distribution of paired
per-infant differences (G4), never as two averages. If arm A scores higher, that
is reported as such — the pre-specification exists so that outcome can be
reported honestly, not so it can be avoided.

**Threshold for calling a difference real.** With n = 15 infants, differences of
0.02–0.03 AuROC are within seed-to-seed variation. A difference will only be
called real if it survives a Wilcoxon signed-rank over the 15 paired per-infant
values at α = 0.05 *and* exceeds the spread across the 5 training seeds. This is
selecting on noise: the mechanism by which a tuned selection criterion
acquires an optimistic bias (Cawley & Talbot), and the one the original paper
cites.

### 1.2 Summary statistic over infants (C6) — **decided: count-weighted**

Target-window counts run 22–306 across the infants (14-fold), and the
per-patient score is strongly anti-correlated with that count — Spearman
ρ = −0.79 under the published control rule and ρ = −0.83 under proximity-matched
controls, both p ≤ 0.001 *(already measured, §28.4)*. A plain mean over infants
is therefore carried by its noisiest terms.

**Primary summary: count-weighted mean.** The plain mean is reported beside it
in every table. On the measured data the two differ by 0.039, which is larger
than most of the effects being tested — so this choice is not cosmetic.

**Patient 009 is excluded from per-patient claims**, with the reason stated:
57% of its marks sit on a dead Abdomen belt (Section 3.4 of the thesis), and
the B2 QC log independently flags block 2 as having a zero robust scale on that
channel *(already measured, §29.1)*. It is retained in pooled analyses.

### 1.3 Normalisation (B2/B3) — **primary: per-block, all four cells reported**

`+experiment=norm_block` is the primary pipeline; `norm_window` (the published
transform), `norm_both` and `norm_none` are all run and all four cells reported.

The baseline cell is verified bit-for-bit identical to the published pipeline
(`scripts/check_norm_chains.py`, max |Δ| = 0.0) *(already measured)*, without
which the cells could not be compared to any earlier result.

### 1.4 Control definition (C4) — **primary: published; proximity-matched reported beside it**

C4 as literally specified (controls from inside periodic-breathing runs) is not
constructible here: 0 windows fit inside such a run and they cover 0.4% of the
record *(already measured, §28.1)*. The substitute draws controls 60–180 s from
the nearest event.

**Primary: the published rule** (≥180 s), for comparability with §7–§30.
**Reported beside it: proximity-matched**, as the test of whether the contrast is
"an apnea is imminent" or merely "this infant is in a busy phase".

### 1.5 Training repetitions (F4) — **decided: 10 seeds**

Vetter et al. repeat training ten times per infant before their Wilcoxon tests.
The runner scripts currently default to 5. **The reported runs will use 10**
(`SEEDS="0 1 2 3 4 5 6 7 8 9"`), and the seed-to-seed spread will be reported,
because §1.1 uses it as the threshold for calling a difference real.

### 1.6 Data exclusions (added 25 August 2026) — **decided: exclude nothing**

Four data-quality problems were found while preparing the B3 runs. **Each was
checked against the results, and none is acted on.** Recording the checks rather
than a rule, because every case resolved the same way:

| problem | check | decision |
|---|---|---|
| 009 block 2: Abdomen belt IQR exactly 0 while swinging −4867…+2336 | the scale floor (§1.7) puts its `Thorax_sum` divisor back in range, 3.904 → 2.435 against a cohort median of 1.818 | **keep** |
| 009: 69% of its marks scored on a dead belt (§6.3a) | those marks desaturate *more* than the cohort — 17 of 18 reach ≥3%, vs 70% cohort-wide (`scripts/check_dead_belt_marks.py`) | **keep**; §6.3a's exclusion advice withdrawn |
| 009 + 012: PCO₂ flat across all four blocks | their folds are unremarkable — 009 is 2nd of 15 (0.633), 012 is at the median. The flat value (≈57 mmHg) is where four live-sensor infants also sit | **keep**, all 6 channels |
| 002 block 3: PCO₂ median 3.5 mmHg → ≈ −2.3, outside [−1, 1] | 002 scores 0.563, above the cohort mean | **keep**, recorded as a limitation |

**Why this is not cheating, and why the date matters.** Every criterion above is
a statement about the *input* (a sensor was disconnected), never about the
outcome, and each was evaluated *before* the sweeps were run. The exclusion set
is empty, so there is no selection to argue about at all. `SIGNAL-ARTIFACT`
remains the only exclusion in the pipeline, unchanged from the published work.

An earlier draft of this section proposed a general "dead-channel rule". It is
**not adopted**: every case it would have governed turned out not to need it.

### 1.7 The per-block scale floor (B2 bullet 4) — **decided: 0.2 × channel median**

Measured over all 60 blocks with `scripts/measure_block_scales.py`, set in
`config/experiment/norm_{block,both}.yaml`:

```yaml
norm_scale_floor: {Abdomen: 5.228, CPAP: 0.09263, PR: 5.468,
                   Thorax: 3.273, Thorax_sum: 0.3637}
```

**Deviation from B2, deliberate.** B2 suggests "e.g. the 5th percentile". On this
cohort p5 would clip **15 blocks to fix 1** — only Abdomen has a genuine failure,
the other channels are a smooth continuum. `0.2 × median` clips exactly the one
dead block and is more stable under leave-one-infant-out (3–6% drift vs 16–19%).

**Computed over all 60 blocks rather than per fold, and this is inert rather than
transductive:** Abdomen's lowest *live* scale is 6.851, so any floor in
(0, 6.851) gives identical output, and the leave-one-out range of the rule
(4.9–5.5) sits entirely inside that gap. Every fold clips the same block.

---

---

## 2. What the diagnostics have already decided

Per H1, the direction taken and what decided it. All of the following were
measured before this file was written:

| diagnostic | result | consequence |
|---|---|---|
| **C3** horizon 0 | 0.581, not ~0.85 | the limit is not the horizon |
| **D6** event-history baseline | 0.677 vs 0.54 physiological | the physiological channels add little |
| **D7** subject identifiability | 7.1× chance | strong subject nuisance is present |
| **D3** within-patient ceiling | gap **+0.0036**, 95% CI −0.008 to +0.016, p = 0.482 *(complete, 57/60 folds)* | subject nuisance is **not** what limits prediction |
| **G2** operating point | 1.4% sensitivity at 1 false alarm/h | not clinically usable |
| **G3** permutation tests | 5/15 infants above chance (18/19 in Vetter) | the effect is real in a minority, absent in most |
| **B2/B3 + C4** effort amplitude | 0.508 → 0.494 count-weighted | the amplitude precursor is absent |

**The direction these point in.** D3 is the decisive one: giving the model three
device blocks of the *test infant* improves its score by 0.004. There is no
within-patient ceiling to reach, so section E of the to-do list — domain
adaptation and personalisation — could not have worked, which is consistent with
the transfer result of Section 4.5 of the thesis, and with the personalisation runs of Section 4.4 finding
that personalisation is null or harmful.

**The conclusion this pre-specifies as acceptable.** If the remaining runs
confirm the above, the reported result is:

> The four physiological modalities available in this cohort do not support
> 15-second-ahead prediction of scored central apnea in preterm infants at
> 28–34 weeks corrected gestational age. Detection works (0.87). An
> event-history baseline reaching ~0.68 at a 1-minute horizon outperforms every
> physiological model tried, which locates the usable signal in the clustering
> of events rather than in a physiological precursor.

This is recorded now so that it reads as a finding rather than as a failure to
reach a target.

---

## 3. What would change the conclusion

Stated in advance, so that a positive result is not dismissed after the fact:

* ~~the **norm_block** cell reaching a count-weighted per-patient AuROC above
  **0.65**, surviving 10 seeds and the Wilcoxon in §1.1~~ — **RESOLVED 26 Aug
  2026: threshold not met.** 4 cells × 10 seeds run; `norm_block` reached
  **0.5827** count-weighted and its contrast against the baseline gives
  Wilcoxon **p = 0.25** (positive in 10/15 infants). Failed on both criteria.
  What the 2×2 *does* show is that per-window standardisation costs
  −0.0252 (p = 0.026) while per-block normalisation adds +0.0018 (p = 0.80) —
  B3's mechanism holds, B3's proposed fix does not. Recorded rather than
  deleted, for the same reason as D3 below. See Section 4.4 of the thesis
  §27.6;
* ~~**D3** completing with a gap above **+0.05** and p < 0.05 over the 60 folds,
  which would reopen section E~~ — **RESOLVED 24 Aug 2026: threshold not met.**
  The gap is +0.0036 with a 95% CI of −0.008 to +0.016, so the pre-specified
  +0.05 is excluded by the data rather than merely unreached. Section E stays
  closed. Recorded here rather than deleted, because the value of a
  pre-specified threshold is that it can be seen not to have moved;
* the **periodic-breathing undercount** (§28.1) turning out to be large under a
  signal-based pause detector, which would mean C1's premise was right after all
  and the window construction does need revisiting.

---

## 4. Runs to execute

```bash
# F3, both arms, at the primary normalisation
DATA_PATH=... RESULT_PATH=... SEEDS="0 1 2 3 4 5 6 7 8 9" \
  CELLS="sampler_equal sampler_natural" bash scripts/run_norm_ablation.sh

# B2/B3, all four cells, at the primary sampler
DATA_PATH=... RESULT_PATH=... SEEDS="0 1 2 3 4 5 6 7 8 9" \
  bash scripts/run_norm_ablation.sh

# D3, to completion
python scripts/leave_one_block_out.py --out outputs/d3/lobo_all15.json
```

### 4.1 Measured cost, and where to run it *(measured 25 August 2026)*

Timed on the development laptop (**CPU-only torch, 4 cores, no CUDA**):

| | measured |
|---|---|
| dataset build, 1 infant | 21.8 s → **5.5 min** for all 15, per run |
| training, per fold | ≈ 12 s per *training* infant (linear: 24 s at 2, 60 s at 5) |
| **one cell, one seed, 15 infants** | **≈ 48 min** (15 folds × 14 infants × 12 s + build) |

**Superseded by the measured GPU figure** *(26 August 2026)*. The laptop
extrapolation above was **wrong by a factor of five**. On the VM's RTX A6000 the
B2/B3 grid — 4 cells × 10 seeds, 40 runs — completed in **6 h 37 min**, i.e.
**≈ 9.9 min per cell per seed**, of which ~5 min is the CPU-bound dataset build
that the GPU does not accelerate. Kept rather than replaced, because the size of
the error is itself worth knowing before anyone extrapolates from a laptop again.

| plan | runs | laptop (est.) | **VM (measured)** |
|---|---|---|---|
| B2/B3, 4 cells × 10 seeds | 40 | ≈ 32 h | **6 h 37 min** ✅ done |
| F3, 2 arms × 10 seeds | 20 | ≈ 16 h | ≈ 3 h 20 min (expected) |

**Run these on the GPU VM, not the laptop.** `run_norm_ablation.sh` is already
written for it (`DATA_PATH=...`), and 48 h of a 4-core CPU is the
wrong instrument. A single-seed run on the laptop is *not* a useful substitute:
§1.5 requires ten seeds, so a one-seed number could not be reported anyway.

**Validated end-to-end before committing VM time** *(25 August 2026)*: all four
cells run to completion on a 2-infant subset, and the new scale floor fires in
the real training path —
`block_norm_stats [patient 009 block 2]: channel Abdomen has a ZERO robust scale
... Floored to 5.228.`

**Operational note.** `save_pickle` refuses to overwrite an existing result
(`FileExistsError`), which is the right behaviour but means an interrupted run
cannot simply be restarted with the same `meta.tag` — clear or rename the partial
output first. Worth knowing before a 32-hour job.
