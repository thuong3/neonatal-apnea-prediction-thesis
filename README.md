# Neonatal apnea prediction under transfer to a preterm CPAP cohort

Code for the bachelor thesis *Neonatal Apnea Prediction* (Ha Thuong Tran,
University of Tübingen, 2026).

This is a fork of
[mackelab/neonatal_apnea_prediction](https://github.com/mackelab/neonatal_apnea_prediction),
the implementation published with Vetter et al. (2024). That study predicted
apnea and hypopnea 15 seconds before onset in 19 infants with Robin sequence,
at a per-patient AUROC of 0.80. The thesis applies the same pipeline to 15
preterm infants on nasal respiratory support, in whom only central apneas were
scored and no airflow channel was recorded.

## What the thesis found

| task | per-patient AUROC |
|---|---|
| Positive control: the published pipeline on its own cohort | 0.789 (published: 0.80) |
| Detection, window contains the event | 0.870 |
| **Prediction, window ends 15 s before onset** | **0.554** |
| Whether the coming minute contains an apnea | 0.677 |
| Whether mean SpO₂ falls below 90 % in the coming minute | 0.770 |

Prediction stays between 0.52 and 0.58 across more than ten approaches, and six
domain adaptation methods recover none of the gap. The positive control matters:
run unchanged on the cohort it was published for, this implementation reaches
0.789, so the null describes the data rather than a reimplementation error.

## Data

Neither dataset is in this repository.

**Robin sequence** (source cohort, used for the positive control) was published
with Vetter et al. (2024) at [Zenodo](https://zenodo.org/record/7711137). The
scored analysis window of that public release does not overlap the paired signal
for most of the nineteen infants; the per-patient clinical export was used
instead.

**Brainimmaturity** (target cohort) was collected for the randomised crossover
trial of Pantalitschka et al. (2009) and scored by Sievers (2008). These are
identifiable physiological recordings and are not deposited publicly. They
remain with the Department of Neonatology at Tübingen University Hospital, from
which access may be requested.

## Installation

```
pip install -e .
```

Python 3.11, PyTorch, Hydra. The full dependency set is in `pyproject.toml`.

## Reproducing a result

### 1. Point the code at your data

Every path defaults to a location relative to the repository root:
`data/brainimmaturity`, `data/robin_sequence`, `results`. Either place your
copies there, or override per run. The defaults live in `config/meta/default.yaml`
and `config/cohort/*.yaml`.

### 2. Run the experiment

Training is driven by Hydra from `src/`:

```bash
cd src
python train_neonatal.py \
    dataset=neonatal +experiment=norm_block \
    meta.experiment=my_run meta.tag=norm_block_s0 meta.seed=0 \
    meta.data_path=../data/brainimmaturity meta.result_path=../results
```

The `run_*.sh` scripts wrap this for the sweeps the thesis reports — each one
documents its cells and seeds in its header:

| script | what it sweeps | thesis |
|---|---|---|
| `neonatal_script.sh` | the original pipeline on Robin | §4.1 |
| `run_capacity.sh` | published capacity vs. wider and longer | §4.3 |
| `run_nested_hp.sh` | nested hyperparameter selection | §4.3 |
| `run_norm_ablation.sh` | the 2×2 normalisation design | §4.4 |
| `run_np_ablation_seeds.sh` | removing the nasal pressure channel | §4.5 |
| `run_domain_adaptation.sh` | the transfer ladder, six variants | §4.5 |
| `run_section19_vm.sh` | Robin restricted to central apnea | §4.5 |
| `run_spo2_baseline.sh` | the oxygenation endpoint | §4.6 |
| `run_clock_drift_experiment.sh` | corrected vs. uncorrected annotations | §3.3 |

### 3. Build the figures

```bash
python scripts/make_thesis_figures.py --out_dir figures
```

Other figures have their own builders: `make_flow_vs_mask_figure.py` (§2.1),
`make_alignment_check_figure.py` (§3.3), `make_effort_channel_figure.py` (§4.4),
`make_hp_tuning_figure.py` (§4.3), `make_da_figure.py` (§4.5),
`make_spo2_baseline_figure.py` (§4.6).

Prefer `make_da_figure.py` over the version inside `make_thesis_figures.py` when
the result pickles are reachable — the latter redraws the same panel from
transcribed constants.

## Which script produced which claim

| thesis | scripts |
|---|---|
| §2.1 channels, scoring, periodic breathing | `band_power.py`, `device_rhythm.py`, `desat_criterion_check.py`, `pb_and_gap_stats.py` |
| §3.1 cohort and blocks | `detect_ventilation_mode.py`, `count_windows.py`, `make_gap_histogram.py` |
| §3.3 annotation clock drift | `measure_clock_drift.py`, `clock_drift_stats.py`, `check_intercepts.py`, `check_timestamp_granularity.py` |
| §3.4 data quality | `check_dead_belt_marks.py` |
| §3.6 normalisation | `measure_block_scales.py`, `check_norm_chains.py` |
| §4.2 detection and desaturation coupling | `apnea_desat_coupling.py`, `coupling_like_for_like.py` |
| §4.3 single-event prediction | `capacity_stats.py`, `permutation_test.py`, `operating_points.py` |
| §4.4 alternatives and ablations | `effort_baseline_prediction.py`, `literature_features_prediction.py`, `precursor_gbm.py`, `norm_ablation_stats.py`, `subject_identifiability.py`, `leave_one_block_out.py`, `patient_adaptation.py`, `personalised_prediction.py`, `np_ablation_stats.py` |
| §4.5 domain adaptation | `da_summary.py` |
| §4.6 apnea-prone state and oxygenation | `clustering_prediction.py`, `hawkes_prediction.py`, `spo2_baseline_prediction.py`, `spo2_operating_points.py` |
| §4.7 leakage | `leakage_personalisation.py` |
| §4.8 causal normalisation | `causal_norm_preview.py`, `check_causal_norm.py` |
| appendix variants | `hr_spo2_channel_preview.py`, `check_sampler_equal.py`, `norm_ablation_preview.py`, `check_dropout_identity.py` |

Every script carries a docstring stating what it measures and which thesis
section reports it. The `check_*.py` scripts are guards rather than
measurements: they assert a property the thesis claims, such as the baseline
normalisation cell being bit-for-bit identical to the published pipeline.

## Layout

| path | contents |
|---|---|
| `src/` | the model, the training loops, shared utilities |
| `config/` | Hydra configs for every experiment, cohort and normalisation variant |
| `scripts/` | one script per measurement, plus the `run_*.sh` drivers |
| `notebooks/` | the three figure notebooks inherited from the original repository |
| `PRE_SPECIFICATION.md` | the analysis decisions fixed in writing before the final runs |

`config/clock_drift.yaml` holds the per-recording clock-drift coefficients.

## Citation

```
@thesis{tran2026neonatal,
  title  = {Neonatal Apnea Prediction},
  author = {Tran, Ha Thuong},
  school = {University of T\"ubingen},
  type   = {Bachelor thesis},
  year   = {2026}
}
```

```
@article{vetter2024neonatal,
  title={Neonatal apnea and hypopnea prediction in infants with Robin sequence with neural additive models for time series},
  author={Vetter, Julius and Lim, Kathleen and Dijkstra, Tjeerd MH and Dargaville, Peter A and Kohlbacher, Oliver and Macke, Jakob H and Poets, Christian F},
  journal={PLOS Digital Health},
  volume={3},
  number={12},
  pages={e0000678},
  year={2024},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

## Licence

MIT, inherited from the original repository. The modifications and the analysis
code written for the thesis are released under the same licence.
