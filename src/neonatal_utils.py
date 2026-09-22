import logging
import os
from datetime import datetime
from fractions import Fraction
from random import shuffle

import numpy as np
import pandas as pd
import scipy
import torch
import yaml

from pyedflib import highlevel
from scipy.signal import decimate, resample_poly
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLOCK_DRIFT_FILE = os.path.join(_REPO_ROOT, "config", "clock_drift.yaml")


# BA adaptation (annotation clock drift)
# ---------------------------------------------------------------------------
# In the CPAP export the annotation clock and the signal clock run at different
# speeds. An event stamped at t seconds after the recording start is actually
# found in the sample arrays at t*(1 + slope) -- about 0 s off at the start of a
# night and ~+10 s off by its end, at a cohort median slope of ~158 ppm
# (~13.7 s per 24 h). Annotation timestamps come from the recorder's real-time
# clock, sample indices are counted on its sampling crystal, and the two do not
# agree.
#
# This was previously treated as a fixed +6 s offset applied when DRAWING
# figures only (see the annotation-offset constant used by the figure scripts).
# That was wrong twice over: no constant can follow a ramp, and correcting the
# picture but not the labels means the model was trained on one alignment and
# inspected under another.
#
# Correcting it HERE fixes both at once, plus the cutter masks (a displaced
# SIGNAL-ARTIFACT span lets artefacts leak into windows) and every standalone
# analysis script that reads these masks.
#
# See scripts/measure_clock_drift.py for the measurement, and
# config/clock_drift.yaml for the per-recording coefficients. The default of
# 0.0 ppm is the identity, so any run that does not opt in reproduces exactly.
def load_clock_drift(path=DEFAULT_CLOCK_DRIFT_FILE):
    """{patient_id: {slope_ppm, intercept_s}} from a config; {} if `path` is None.

    `None` is the correct value for any cohort the drift was not measured on --
    notably the Robin-sequence set, whose patient ids OVERLAP with the CPAP
    cohort's, so a drift table keyed by id alone would silently be applied to
    the wrong recordings.
    """
    if path is None:
        return {}
    # Hydra chdir's into its run directory, so a config-relative path such as
    # "config/clock_drift.yaml" must be resolved against the repo, not the cwd.
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"clock-drift config {path} not found. Generate it with "
            "`python scripts/measure_clock_drift.py`, or pass None to run "
            "uncorrected."
        )
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return {
        k: {"slope_ppm": float(v.get("slope_ppm", 0.0)),
            "intercept_s": float(v.get("intercept_s", 0.0))}
        for k, v in (cfg.get("patients") or {}).items()
    }


def _clock_map(clock_drift):
    """(scale, shift) such that annotation time t -> signal time t*scale + shift.

    The SLOPE is the measured quantity and the one that matters: it removes a
    time-varying error that reaches ~10 s by the end of a night, and after
    applying it the residual lag is flat to +-0.5 s across a whole recording.

    The INTERCEPT is deliberately 0 by default. Fitted freely, the anchors
    disagree about it in opposite directions -- DESAT lands at ~-1.9 s, CPAP
    modulation at ~+1.4 s -- which is the signature of each anchor's own bias in
    locating an event, not of a real offset at t = 0 (where the recorder's
    real-time clock and its sample counter start together and so cannot
    disagree). Zero is the midpoint of that bracket, and the residual is under
    the ~2 s with which a scorer places a mark by hand.

    It is exposed rather than hard-coded because only a direct comparison
    against RemLogic can settle the remaining couple of seconds; set
    `intercept_s` per patient in config/clock_drift.yaml if that comparison
    shows a consistent constant.
    """
    if not clock_drift:
        return 1.0, 0.0
    return (1.0 + float(clock_drift.get("slope_ppm", 0.0)) * 1e-6,
            float(clock_drift.get("intercept_s", 0.0)))


# ---------------------------------------------------------------------------
# BA adaptation (CPAP / Embla data set)
# ---------------------------------------------------------------------------
# Map raw EDF channel labels to the internal signal names used by the model.
# Unlike the original code the sampling frequency is NOT hard-coded here; it is
# read from the EDF header (see `_header_freq`). Every channel is therefore
# up-sampled (sample-and-hold) to the common TARGET_FREQ regardless of its
# native rate, so recordings with different sampling rates work unchanged.
#
# "CPAP Pressure" is its OWN channel ("CPAP"), not an alias for "NP" (Nasal
# Pressure): the CPAP recordings contain no true nasal-airflow channel, and
# mask pressure is a physically different quantity (dominated by the flow
# generator's set pressure, only weakly modulated by the infant's own
# breathing -- see Section 3.2 of the thesis). Labelling it "CPAP" keeps the
# model/interpretability output honest and lets an ablation (config/
# experiment/no_np.yaml) isolate whether this channel contributes anything.
TARGET_FREQ = 200  # common rate (Hz) every channel is brought to

SIGNAL_LABEL_MAP = {
    "EKG": "EKG",
    "Pleth Radical": "PR",
    "Pulswelle": "PR",  # CPAP export label for the pulse plethysmogram
    "Nasal Pressure": "NP",
    "CPAP Pressure": "CPAP",  # own channel; NOT a surrogate for NP
    "Thorax": "Thorax",
    "Brust": "Thorax",  # CPAP export label
    "Abdomen": "Abdomen",
    "Bauch": "Abdomen",  # CPAP export label
    "Sentec PCO2": "PCO2",
    "tcpCO2": "PCO2",  # CPAP export label
    "SpO2 Radical": "SpO2",
    "SpO2": "SpO2",
    "Heart Rate B2B": "HR",
    "Heart rate B2B": "HR",  # One file contains "rate" instead of "Rate"
    "Herzfrequenz_CU": "HR",  # CPAP export label
    # BA: gold-standard respiratory effort (esophageal pressure catheter).
    # The "oe" is exported as "?" because EDF labels are ASCII-only.
    "?sophagusdruck": "ESO",
    "Oesophagusdruck": "ESO",
    "tcpO2": "tcpO2",
}


def _header_freq(sig_header, freq_scale=1.0):
    """Native sampling rate (Hz) of a channel, taken from the EDF header.

    `freq_scale` corrects a mis-declared data-record duration (see
    `_declared_record_duration`); it is 1.0 for a well-formed file.
    """
    freq = sig_header.get("sample_frequency", sig_header.get("sample_rate"))
    return int(round(freq * freq_scale))


# BA adaptation (cross-cohort / domain adaptation)
# ---------------------------------------------------------------------------
# pyedflib reports a channel's rate as (samples per data record) / (declared
# data-record duration). That is only the true rate if the header's declared
# record duration is correct -- and in the published Robin-sequence export it
# is NOT: every one of the 19 EDFs declares a 1 s record while the records are
# really 10 s long. The evidence is the annotations, which consistently run
# ~10x past the declared end of file (e.g. patient 001: declared 4867 s, last
# annotation at 48645 s -> ratio 9.99; the ratio is 8.7-10.0 for all 19, the
# low ones simply stop being scored before end-of-file). Dividing out the
# factor 10 recovers exactly the rates the original code hard-coded (EKG 200,
# NP 200, Thorax/Abdomen 50, PR 100, PCO2/SpO2 2, HR 1 Hz), which confirms the
# correction. The CPAP export declares its 10 s records correctly and is
# therefore read unchanged (freq_scale = 1.0).
#
# Left uncorrected the failure is SILENT, not loud: the channels get
# downsampled by 10, the recording collapses to a tenth of its true length,
# and the annotation indices -- which are computed from real wall-clock
# timestamps -- then land past the end of the mask arrays, where numpy slice
# assignment is a no-op. The result would be a cohort with almost no positive
# labels rather than an exception. `_assert_annotations_fit` below turns that
# silent corruption into an error.
def _declared_record_duration(edf_path):
    """Data-record duration (s) declared in the raw EDF header (offset 244)."""
    with open(edf_path, "rb") as fh:
        fh.seek(244)
        return float(fh.read(8).decode("ascii").strip())


def _assert_annotations_fit(pat_id, annotations, signal_start, n_samples, declared_rd,
                            clock_scale=1.0):
    """Fail loudly if the annotations do not fit inside the loaded signals.

    Annotation timestamps are absolute wall-clock times, so they line up with
    the sample arrays only if every channel's rate was read correctly. If they
    run past the end of the recording, the rates are wrong. This must be an
    error rather than a warning: writing to an out-of-range numpy slice is a
    silent no-op, so the run would otherwise continue with a near-empty label
    mask and report a meaningless AUC.
    """
    if annotations.empty:
        return
    last = max(
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        for ts in annotations["timestamp"]
    )
    # Same clock correction the indices get, so this check tests the sampling
    # rates (what it is for) and not the drift (which is handled separately).
    span = (last - signal_start) * clock_scale
    duration = n_samples / TARGET_FREQ
    # 60 s of slack absorbs an event that is scored right at the recording end.
    if span > duration + 60.0:
        raise ValueError(
            f"Patient {pat_id}: annotations span {span:.0f}s but the loaded "
            f"signals are only {duration:.0f}s long (factor {span / duration:.2f}). "
            "The EDF's declared data-record duration is almost certainly wrong, "
            "so every channel was read at the wrong rate. Pass the true record "
            "duration via the dataset config's `record_duration` field "
            f"(the header declares {declared_rd:g}s, the mismatch implies "
            f"{declared_rd * span / duration:.0f}s)."
        )


# Read in the raw signals of one recording. Returns a *list* of signal
# segments, one per ANALYSIS-START/ANALYSIS-STOP block in the annotations.
def read_data(pat_id, file_path, annotations_dir="annotations", record_duration=None,
              clock_drift=None):
    """Read one recording.

    `annotations_dir`   name of the annotation sub-folder (the Robin-sequence
                        cohort ships them in `annotations_original/`).
    `record_duration`   true data-record duration in seconds. Pass it only to
                        override a mis-declared EDF header (the Robin export
                        needs `record_duration: 10`); `None` trusts the file.
    `clock_drift`       {"slope_ppm": ..., "intercept_s": ...} for this
                        recording, as produced by `load_clock_drift`, or None.
                        An event stamped t seconds after the recording start is
                        placed at t*(1 + slope_ppm*1e-6) + intercept_s seconds
                        into the arrays. None, the default, is the identity, so
                        any caller that does not opt in is unaffected.
    """
    print(f"Reading ID {pat_id}.")

    # Two layouts exist for this data. The CPAP export and the cleaned
    # Robin export split into signals/ and annotations/ sub-folders; the
    # original upstream release (and `VM Dateien/data_manual`) keeps both in one
    # FLAT folder. Try the sub-folder first, fall back to flat, so a reference
    # run against the published release does not need the files rearranged.
    edf_path = os.path.join(file_path, "signals", f"signals{pat_id}.edf")
    if not os.path.exists(edf_path):
        edf_path = os.path.join(file_path, f"signals{pat_id}.edf")
    signals, signal_headers, header = highlevel.read_edf(edf_path)

    # See the _declared_record_duration note above.
    if record_duration is None:
        freq_scale = 1.0
    else:
        freq_scale = _declared_record_duration(edf_path) / float(record_duration)

    all_signals = {}

    for sig_header, sig in zip(signal_headers, signals):
        label = sig_header["label"]
        if label in SIGNAL_LABEL_MAP:
            key = SIGNAL_LABEL_MAP[label]
            freq = _header_freq(sig_header, freq_scale)
            if freq <= TARGET_FREQ:
                # Original path: sample-and-hold upsampling. Unchanged so
                # every previously reported CPAP-cohort result (all channels
                # there are <=200 Hz) stays exactly reproducible.
                assert TARGET_FREQ % freq == 0, (
                    f"{label}: native rate {freq} Hz does not divide {TARGET_FREQ} Hz"
                )
                all_signals[key] = np.repeat(sig, TARGET_FREQ // freq)
            else:
                # Higher-rate source (e.g. the original Robin-sequence/Zenodo
                # export, EKG/NP at 2000 Hz): downsample with an
                # anti-aliasing filter. freq/TARGET_FREQ need not be an
                # integer (Thorax/Abdomen are 500 Hz -> 200 Hz, a 2/5 ratio),
                # so use polyphase resampling rather than plain decimate.
                frac = Fraction(TARGET_FREQ, freq).limit_denominator(1000)
                all_signals[key] = resample_poly(
                    np.asarray(sig, dtype=float), frac.numerator, frac.denominator
                )
        else:
            # Log skipped signal types.
            print(f"Skipped {label}")
            continue

    if all_signals:
        # Fractional resampling can leave channels a sample or two apart in
        # length; the rest of this function assumes a single shared length.
        min_len = min(len(v) for v in all_signals.values())
        all_signals = {k: v[:min_len] for k, v in all_signals.items()}

    # Same two-layout fallback as the EDF above.
    anno_path = os.path.join(file_path, annotations_dir, f"annotations{pat_id}.txt")
    if not os.path.exists(anno_path):
        anno_path = os.path.join(file_path, f"annotations{pat_id}.txt")
    annotations = pd.read_csv(anno_path, sep="\t")

    annotation_types = [
        "APNEA-OBSTRUCTIVE",
        "APNEA-CENTRAL",
        "APNEA-MIXED",
        "HYPOPNEA-OBSTRUCTIVE",
        "HYPOPNEA-CENTRAL",
        "HYPOPNEA-MIXED",
        "DESAT",
        "ACTIVITY-MOVE",
        "SIGNAL-ARTIFACT",
        "SIGNAL-QUALITY-LOW",
    ]
    # Annotation masks live at TARGET_FREQ, matching the up-sampled signals.
    # Do NOT derive the length from signals[0]: its native rate may differ
    # from TARGET_FREQ, which would misalign every annotation index.
    ref_len = len(next(iter(all_signals.values())))
    all_signals.update(
        {anno_type: np.zeros(ref_len, dtype=int) for anno_type in annotation_types}
    )

    SIGNAL_START = header["startdate"].timestamp()
    # Annotation clock -> signal clock. A pure scaling, with no constant term:
    # the recorder's real-time clock and its sample counter start together, so
    # the two cannot disagree at t = 0 (measured and confirmed -- see
    # scripts/measure_clock_drift.py). Event DURATIONS are on the same clock and
    # are therefore scaled by the same factor, which falls out of scaling both
    # ends of the span below.
    clock_scale, clock_shift = _clock_map(clock_drift)
    if clock_scale != 1.0 or clock_shift:
        end_shift = (clock_scale - 1) * ref_len / TARGET_FREQ + clock_shift
        log.info(
            f"clock drift: {(clock_scale - 1) * 1e6:+.1f} ppm, "
            f"{clock_shift:+.2f} s -> annotations near the end of this "
            f"recording move {end_shift:+.1f} s"
        )
    _assert_annotations_fit(
        pat_id,
        annotations,
        SIGNAL_START,
        ref_len,
        _declared_record_duration(edf_path),
        clock_scale=clock_scale,
    )
    # A recording may contain several ANALYSIS-START / ANALYSIS-STOP blocks.
    # Collect every analysed span; each is returned as a separate segment so
    # that no time window and no event distance crosses a block boundary.
    analysis_spans = []
    current_start = None
    for row in annotations.itertuples(index=False):
        event_type = row.type
        start = datetime.strptime(row.timestamp, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        dur = row.duration
        ind_start = int(np.round(
            ((start - SIGNAL_START) * clock_scale + clock_shift) * TARGET_FREQ))
        ind_end = int(np.round(
            ((start + dur - SIGNAL_START) * clock_scale + clock_shift) * TARGET_FREQ))
        if event_type == "ANALYSIS-START":
            current_start = ind_start
        elif event_type == "ANALYSIS-STOP":
            if current_start is not None:
                analysis_spans.append((current_start, ind_end))
                current_start = None
        elif event_type in annotation_types:
            all_signals[event_type][ind_start : ind_end + 1] = 1
        else:
            raise ValueError(f"Unknown annotation type: {event_type}")

    # Fallback: no ANALYSIS block -> keep the whole recording as one segment.
    if not analysis_spans:
        analysis_spans = [(0, ref_len)]

    # Each segment carries its own absolute wall-clock start time (recording
    # start + the segment's sample offset). This lets downstream code turn a
    # window's sample slice back into a real timestamp (e.g. to look up the
    # raw EDF/annotation data for a specific prediction example).
    segments = []
    for s, e in analysis_spans:
        seg = {k: v[s:e] for k, v in all_signals.items()}
        # NOTE: this is a SIGNAL-clock timestamp -- sample index / TARGET_FREQ
        # after the recording start. Once a drift correction is applied it is no
        # longer the same as the annotation-clock time a scorer reads in
        # RemLogic; the two diverge by up to ~10 s over a night. The recording
        # start and the scale factor travel with the segment so that downstream
        # code can convert back for display (`annotation_clock_ts`) instead of
        # having to re-read the config.
        seg["_SEGMENT_START_TS"] = SIGNAL_START + s / TARGET_FREQ
        seg["_RECORDING_START_TS"] = SIGNAL_START
        seg["_CLOCK_SCALE"] = clock_scale
        seg["_CLOCK_SHIFT"] = clock_shift
        segments.append(seg)
    return segments


def annotation_clock_ts(signal_ts, recording_start_ts, clock_scale, clock_shift=0.0):
    """Signal-clock timestamp -> the wall clock a scorer sees in RemLogic.

    The inverse of the mapping applied in `read_data`. Use it for anything a
    human will look up in the scoring software; use the signal clock for
    anything that indexes the arrays.
    """
    return recording_start_ts + (signal_ts - recording_start_ts - clock_shift) / clock_scale


def sum_from_left(arr):
    res = np.zeros_like(arr)
    sumi = 0
    for idx, i in enumerate(arr):
        if i == 0:
            sumi = 0
        if i == 1:
            res[idx] = sumi
            sumi += 1
    return res


def sum_from_right(arr):
    temp = sum_from_left(arr[::-1])
    return temp[::-1]


def create_slices(cutter):
    temp = np.concatenate([np.zeros(1), cutter, np.zeros(1)])

    diff = temp[1:] - temp[:-1]
    start_pos_arr = np.where(diff == 1)[0]
    end_pos_arr = np.where(diff == -1)[0]

    return [slice(start, end) for start, end in zip(start_pos_arr, end_pos_arr)]


def standardize(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    return (arr - arr_mean) / (arr_std if (arr_std > 0.0) else 1.0)


def normalize_range(signal, lower_source, upper_source, lower_target, upper_target):
    assert lower_source < upper_source
    assert lower_target < upper_target

    return (
        (upper_target - lower_target)
        * ((signal - lower_source) / (upper_source - lower_source))
    ) + lower_target


# BA adaptation (B2/B3): per-block robust normalisation
# ---------------------------------------------------------------------------
# The published pipeline standardises every signal on a 30 s TIME-WINDOW basis
# (`standardize`, applied inside `feature_extraction`). Under that transform two
# windows with identical waveform SHAPE but a tenfold difference in AMPLITUDE
# are numerically identical.
#
# For the obstructive events of the Robin-sequence cohort that costs nothing:
# the discriminative pattern there is waveform irregularity, which survives it.
# For central apnea the leading candidate precursor is a progressive DECLINE IN
# RESPIRATORY EFFORT AMPLITUDE relative to the infant's own baseline -- which
# per-window standardisation destroys by construction, and which the scoring
# rule ("below 20% of the mean amplitude of the preceding breaths") is literally
# built on.
#
# The alternative is to normalise once per BLOCK (one infant, one device, i.e.
# one ANALYSIS-START/STOP segment here) with a median and an IQR-based scale:
#
#     m = median(x),  s = (Q75(x) - Q25(x)) / 1.349,  xn = (x - m) / max(s, floor)
#
# The 1.349 divisor makes `s` a consistent estimate of sigma under Gaussianity,
# so the normalised signals land on the same numerical scale as the z-scored
# inputs the published network was tuned on and its hyperparameters transfer
# without rescaling. After the transform one unit is approximately one typical
# breath amplitude for that infant on that device -- the same scale the human
# scorer used, which makes the feature space commensurate with the label space.
#
# Both switches are exposed so the full 2x2 can be run and reported:
#   norm_per_window=True,  norm_per_block=False  -> the published baseline
#   norm_per_window=False, norm_per_block=True   -> the proposed primary
#   norm_per_window=True,  norm_per_block=True   -> both
#   norm_per_window=False, norm_per_block=False  -> neither (control)
# The defaults are the FIRST cell, so every previously reported result
# reproduces bit-for-bit without touching a config.
# ---------------------------------------------------------------------------

# Per-channel decimation chain, mirroring `feature_extraction` exactly. Block
# statistics must be computed on the FILTERED, DECIMATED signal, not on raw
# samples, so that filter transients and out-of-band noise do not inflate the
# IQR. `scripts/check_norm_chains.py` is the guard that keeps this table honest
# if `feature_extraction` is ever edited.
DECIMATION_CHAIN = {
    "NP": (("decimate", 10), ("decimate", 4)),
    "CPAP": (("decimate", 10), ("decimate", 4)),
    "PR": (("decimate", 10), ("decimate", 4)),
    "Thorax": (("select", 4), ("decimate", 10)),
    "Abdomen": (("select", 4), ("decimate", 10)),
}

# Channels that get per-block robust normalisation: the arbitrary-unit
# oscillatory signals. HR, SpO2 and PCO2 are deliberately NOT in this list --
# they are in physical units and are range-normalised over fixed clinical ranges
# (HR 50-240 bpm, SpO2 60-100%, PCO2 30-70 mmHg), where the absolute level is
# meaningful and must be preserved.
BLOCK_NORM_CHANNELS = tuple(DECIMATION_CHAIN)

# ---------------------------------------------------------------------------
# B2 bullet 7: HR and SpO2 carried as TWO channels each.
#
# "Heart rate and SpO2: keep both scales. These are in physical units and are
# range-normalised to [-1, 1] over fixed ranges. The absolute level is
# clinically meaningful -- baseline heart rate 163-167 bpm and baseline SpO2
# 96-97% in this cohort -- so keep the fixed-range channel as primary and ADD a
# second, per-block robust-centred channel for each. The network then sees both
# the absolute level and the deviation from this infant's own baseline on this
# device, which is the cheapest possible attack on between-patient shift."
#
# The fixed-range channels "HR" and "SpO2" are UNCHANGED. These are the added
# ones, and they take the network from 6 inputs to 8 -- which is why this can
# never be a cell of the B3 2x2 and needs its own sweep and its own network
# config (`nam_b7`).
#
# WHY THESE ARE NOT SIMPLY ADDED TO DECIMATION_CHAIN. `BLOCK_NORM_CHANNELS` is
# derived from it, so putting HR and SpO2 there would silently start
# block-normalising them in EVERY existing cell -- changing norm_block,
# norm_both and every number section 27.6 reports. They are kept in a separate
# table for that reason and nothing else.
#
# The chains mirror `feature_extraction`'s handling of the fixed-range channels
# exactly: HR is plain subsampling to 1 Hz, SpO2 is subsampling then a
# decimate-by-2. `check_hr_spo2_channels.py` test [2] asserts the two agree.
DEV_CHANNELS = {"HR_dev": "HR", "SpO2_dev": "SpO2"}
DEV_DECIMATION_CHAIN = {
    "HR": (("select", 200),),
    "SpO2": (("select", 100), ("decimate", 2)),
}


def _apply_chain(arr, chain):
    """Apply a `DECIMATION_CHAIN` entry to a 1-D array."""
    out = np.asarray(arr, dtype=float)
    for kind, factor in chain:
        out = out[::factor] if kind == "select" else decimate(out, factor)
    return out


def robust_stats(arr, mask=None):
    """Median and IQR-based scale of `arr`, optionally over `mask` only.

    Returns (median, scale) with scale = IQR / 1.349, a consistent estimate of
    sigma under Gaussianity.
    """
    x = np.asarray(arr, dtype=float)
    if mask is not None:
        x = x[mask]
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    q25, q75 = np.percentile(x, [25, 75])
    return float(np.median(x)), float(q75 - q25) / 1.349


def _mask_to_length(mask, n_out):
    """Resample a boolean mask defined at TARGET_FREQ onto `n_out` samples."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if len(mask) == 0 or n_out <= 0:
        return None
    idx = np.clip((np.arange(n_out) * (len(mask) / n_out)).astype(int), 0, len(mask) - 1)
    return mask[idx].astype(bool)


def apply_robust(arr, stats, floor=None):
    """(x - m) / max(s, floor). A zero scale falls back to 1.0 (never /0)."""
    m, s = stats
    if floor is not None:
        s = max(s, float(floor))
    return (np.asarray(arr, dtype=float) - m) / (s if s > 0.0 else 1.0)


def block_norm_stats(seg, signal_types, artifact_free=None, scale_floor=None,
                     pat_id=None, segment_idx=None):
    """Per-block median/scale for every channel that gets robust normalisation.

    `seg`            one ANALYSIS-START/STOP segment as returned by `read_data`.
    `artifact_free`  boolean mask at TARGET_FREQ. This must be built from the
                     CUTTER events only (SIGNAL-ARTIFACT), never from the
                     adverse events: excluding annotated apneas would make the
                     normalisation a function of the LABELS and would leak label
                     information into held-out blocks. Events occupy ~0.4-1.1%
                     of the recording anyway, so they move a median and an IQR
                     by a negligible amount.
    `scale_floor`    optional {channel: value} lower bound on the scale, to stop
                     a dead or disconnected belt producing a near-zero divisor
                     that amplifies its own noise to full scale.

    Returns {channel: (median, scale)}, plus the synthetic key "Thorax_sum" (the
    summed respiratory effort) when Thorax is requested: the Vetter recipe is to
    normalise chest and abdomen SEPARATELY, sum pointwise, then normalise the
    sum -- so the sum needs its own block statistics.
    """
    needed = set()
    for signal_type in signal_types:
        if signal_type == "Thorax":
            # The "Thorax" model channel is the SUM of both belts.
            needed.update(("Thorax", "Abdomen"))
        elif signal_type in BLOCK_NORM_CHANNELS:
            needed.add(signal_type)

    def _floor(channel):
        if scale_floor is None:
            return None
        return scale_floor.get(channel) if hasattr(scale_floor, "get") else scale_floor

    decimated, stats = {}, {}
    for channel in sorted(needed):
        dec = _apply_chain(seg[channel], DECIMATION_CHAIN[channel])
        decimated[channel] = dec
        stats[channel] = robust_stats(dec, _mask_to_length(artifact_free, len(dec)))

    if "Thorax" in signal_types:
        thorax, abdomen = decimated["Thorax"], decimated["Abdomen"]
        n = min(len(thorax), len(abdomen))
        summed = (
            apply_robust(thorax[:n], stats["Thorax"], _floor("Thorax"))
            + apply_robust(abdomen[:n], stats["Abdomen"], _floor("Abdomen"))
        )
        stats["Thorax_sum"] = robust_stats(summed, _mask_to_length(artifact_free, n))

    # B2 bullet 7: statistics for the ADDED deviation channels. Computed on the
    # same decimation the fixed-range channel uses, so the pair describes one
    # signal at two scales rather than two different signals. Only computed when
    # the channel is actually requested, so no existing cell is touched.
    for dev, base in DEV_CHANNELS.items():
        if dev in signal_types:
            dec = _apply_chain(seg[base], DEV_DECIMATION_CHAIN[base])
            stats[dev] = robust_stats(dec, _mask_to_length(artifact_free, len(dec)))

    # QC log. A block whose scale sits far below the rest is the signature of a
    # loose or disconnected inductance belt; log every scale so a floor can be
    # chosen from the training blocks rather than guessed.
    where = f"patient {pat_id} block {segment_idx}"
    log.info(
        f"block_norm_stats [{where}]: "
        + ", ".join(f"{c}: m={m:.4g} s={s:.4g}" for c, (m, s) in sorted(stats.items()))
    )
    # B2 asks for any block that HITS THE FLOOR to be logged as a QC failure,
    # not only one whose scale is exactly zero. The distinction is load-bearing:
    # a belt that is railed for most of a block but still spikes has an IQR of
    # exactly 0 AND a peak-to-peak of thousands (patient 009 block 2, Abdomen),
    # while a merely loose belt has a small but non-zero scale. Both need the
    # floor; only the first was previously logged.
    for channel, (_, s) in stats.items():
        floor = _floor(channel)
        if s <= 0.0:
            detail = (
                f"Floored to {floor:.4g}."
                if floor
                else (
                    "NO FLOOR IS SET, so apply_robust falls back to dividing by "
                    "1.0 and this channel enters in RAW units while its siblings "
                    "are on unit scale. For summed effort that lets a dead belt "
                    "dominate Thorax_sum. Set dataset.norm_scale_floor -- run "
                    "scripts/measure_block_scales.py to choose it."
                )
            )
            log.warning(
                f"block_norm_stats [{where}]: channel {channel} has a ZERO robust "
                f"scale (flat over the artifact-free mask). {detail} "
                "This block is a QC failure for that channel."
            )
        elif floor is not None and s < floor:
            log.warning(
                f"block_norm_stats [{where}]: channel {channel} scale {s:.4g} is "
                f"below the floor {floor:.4g} and was clipped to it -- the "
                "loose/disconnected-sensor signature. QC failure for that channel."
            )
    return stats


# ---------------------------------------------------------------------------
# B5: CAUSAL normalisation statistics, beside the transductive ones.
#
# B5 asks: "Report both transductive and causal normalisation statistics.
# Whole-block median and IQR are transductive: no label leakage, but they
# presume the entire 6-hour block is in hand. Report (i) whole-block statistics
# as primary, matching an 'adapt once per recording' deployment, and (ii) causal
# statistics from an expanding window with a 10-minute warm-up, or a trailing
# 30-minute window, matching real-time use. The difference between the two is
# the price of real-time operation and deserves a sentence in the paper."
#
# (i) is `block_norm_stats` above and stays the primary. (ii) is here. Both are
# selected with `dataset.norm_stats_mode`:
#
#     whole_block   median/IQR over the entire block          (default, primary)
#     expanding     over [block start, t), 10-minute warm-up   (real-time)
#     trailing      over [t - 30 min, t)                       (real-time)
#
# The default is `whole_block`, so every previously reported result reproduces
# without touching a config.
#
# WHAT IS AND IS NOT CAUSAL HERE.
# The statistics are taken from history that ENDS AT THE WINDOW'S START, so a
# window never contributes to the scale it is divided by. Using history up to
# the window's END would also be defensible for real-time use (at decision time
# you do have those 30 s), but then a window partly sets its own normalisation,
# and a window containing an unusual event would flatten itself. Ending at the
# start is the conservative choice and it keeps the transform a constant applied
# to the window, exactly as in the whole-block case.
#
# THE WARM-UP INVOLVES A BOUNDED, DOCUMENTED CONCESSION. A window in the first
# 10 minutes of a block has less than 10 minutes of history. Real-time hardware
# would simply not score those windows -- but DROPPING them would change the
# window set and break the pairing that every comparison in this study depends
# on (`norm_ablation_stats.verify_pairing` asserts it). So a window earlier than
# the warm-up is normalised with the statistics AS AT the end of the warm-up,
# i.e. it may use up to 10 minutes of data that postdates it. This is a
# look-ahead of at most CAUSAL_WARMUP_S, it affects only the first 10 minutes of
# each block, and `scripts/causal_norm_preview.py` reports exactly how many
# windows it touches so the number can be quoted rather than hand-waved.
#
# COST. Recomputing a median and an IQR per window over a growing history is the
# expensive part. Two things keep it cheap: the statistics are computed once per
# WINDOW (not per sample), and a long history is subsampled to at most
# CAUSAL_MAX_HISTORY_SAMPLES before the percentiles are taken. Quantiles from a
# systematic subsample of a smooth physiological signal are accurate to far
# better than the differences being measured here; `causal_norm_preview.py`
# test [3] measures that error rather than assuming it.
# ---------------------------------------------------------------------------
CAUSAL_WARMUP_S = 600.0            # 10 min, as B5 specifies
CAUSAL_TRAILING_S = 1800.0         # 30 min, as B5 specifies
CAUSAL_MAX_HISTORY_SAMPLES = 30000 # subsample cap for the percentile estimate
NORM_STATS_MODES = ("whole_block", "expanding", "trailing")

# THE ANTI-ALIAS FILTER IS THE REASON THIS CONSTANT EXISTS, and it is the one
# trap in the whole of B5. `_apply_chain` decimates with `scipy.signal.decimate`,
# whose default `zero_phase=True` applies the filter with `filtfilt` -- FORWARD
# AND BACKWARD. The backward pass propagates information from later samples into
# earlier ones, so a block decimated in one pass has every output sample
# contaminated by its future. Slicing "history only" out of that array is
# therefore NOT causal, however carefully the indices are computed.
#
# Measured, not reasoned about: with no guard band, corrupting the signal AFTER a
# window changed that window's statistics by 1.8e-2 (`check_causal_norm.py`
# test [2] fails loudly if this regresses).
#
# Two fixes were rejected. Decimating each window's history separately is exact
# but re-filters a growing slice per window -- minutes per block, hours per
# sweep. Switching to `zero_phase=False` makes the chain causal but changes the
# filter, so the statistics would then describe a DIFFERENT signal from the one
# `feature_extraction` normalises -- precisely the mismatch
# `check_norm_chains.py` test [3] exists to prevent.
#
# What is used instead: a GUARD BAND. The history stops this many seconds before
# the window starts, which is far longer than the filter's impulse response, so
# no retained sample carries information from the window or from anything after
# it. The cost is that the statistics ignore the last few seconds of history --
# irrelevant against a 10-minute warm-up or a 30-minute trailing window.
CAUSAL_FILTER_GUARD_S = 10.0


def prepare_causal_norm(seg, signal_types, artifact_free=None):
    """Decimate a block ONCE, so per-window causal statistics are cheap.

    Returns a dict consumed by `causal_norm_stats_at`. Mirrors the channel set
    and the decimation chain of `block_norm_stats` exactly -- the two must agree
    or the causal and transductive arms would not be measuring the same signal.
    """
    needed = set()
    for signal_type in signal_types:
        if signal_type == "Thorax":
            needed.update(("Thorax", "Abdomen"))
        elif signal_type in BLOCK_NORM_CHANNELS:
            needed.add(signal_type)

    n_full = len(seg[next(iter(needed))]) if needed else 0
    decimated, masks = {}, {}
    for channel in sorted(needed):
        dec = _apply_chain(seg[channel], DECIMATION_CHAIN[channel])
        decimated[channel] = dec
        masks[channel] = _mask_to_length(artifact_free, len(dec))
    return {
        "decimated": decimated,
        "masks": masks,
        "n_full": n_full,
        "wants_sum": "Thorax" in signal_types,
    }


def _history_slice(n_dec, n_full, window_start, mode):
    """Index range of the history a window at `window_start` may legally use.

    `window_start` is in TARGET_FREQ samples; the return is in DECIMATED ones.
    """
    if n_full <= 0 or n_dec <= 0:
        return 0, 0
    rate = n_dec / n_full                       # decimated samples per raw sample
    # The guard band: stop short of the window so the zero-phase anti-alias
    # filter cannot have carried the window's own samples backwards into the
    # history. See CAUSAL_FILTER_GUARD_S -- this line is what makes the arm
    # causal, and check_causal_norm.py test [2] fails without it.
    hi = int(window_start * rate) - int(CAUSAL_FILTER_GUARD_S * TARGET_FREQ * rate)
    warm = int(CAUSAL_WARMUP_S * TARGET_FREQ * rate)
    # The warm-up: never normalise on less than CAUSAL_WARMUP_S of history.
    hi = max(hi, warm)
    hi = min(hi, n_dec)
    if mode == "trailing":
        lo = max(0, hi - int(CAUSAL_TRAILING_S * TARGET_FREQ * rate))
    else:                                        # expanding
        lo = 0
    return lo, hi


def _subsample(x, mask):
    """Apply `mask`, then thin to at most CAUSAL_MAX_HISTORY_SAMPLES points."""
    if mask is not None:
        x = x[mask]
    x = x[np.isfinite(x)]
    if x.size > CAUSAL_MAX_HISTORY_SAMPLES:
        x = x[:: int(np.ceil(x.size / CAUSAL_MAX_HISTORY_SAMPLES))]
    return x


def causal_norm_stats_at(prep, window_start, mode, scale_floor=None):
    """`block_norm_stats`-shaped statistics computed from HISTORY ONLY.

    Same return type as `block_norm_stats` -- {channel: (median, scale)} plus
    "Thorax_sum" -- so `feature_extraction` consumes it unchanged.
    """
    if mode not in ("expanding", "trailing"):
        raise ValueError(f"causal mode must be expanding|trailing, got {mode}")

    def _floor(channel):
        if scale_floor is None:
            return None
        return scale_floor.get(channel) if hasattr(scale_floor, "get") else scale_floor

    decimated, masks, n_full = prep["decimated"], prep["masks"], prep["n_full"]
    stats, hist = {}, {}
    for channel, dec in decimated.items():
        lo, hi = _history_slice(len(dec), n_full, window_start, mode)
        m = masks[channel]
        window_mask = None if m is None else m[lo:hi]
        hist[channel] = (lo, hi)
        stats[channel] = robust_stats(*_stats_args(dec[lo:hi], window_mask))

    if prep["wants_sum"]:
        # The summed effort needs its own statistics, and they must come from
        # the SAME history: normalise each belt by its causal stats over that
        # history, sum, then take the sum's median/IQR over it. Anything else
        # would mix a causal numerator with a transductive scale.
        lo_t, hi_t = hist["Thorax"]
        lo_a, hi_a = hist["Abdomen"]
        th = apply_robust(decimated["Thorax"][lo_t:hi_t], stats["Thorax"],
                          _floor("Thorax"))
        ab = apply_robust(decimated["Abdomen"][lo_a:hi_a], stats["Abdomen"],
                          _floor("Abdomen"))
        n = min(len(th), len(ab))
        m = masks["Thorax"]
        summed_mask = None if m is None else m[lo_t:lo_t + n]
        stats["Thorax_sum"] = robust_stats(*_stats_args(th[:n] + ab[:n], summed_mask))
    return stats


def _stats_args(x, mask):
    """(values, None) after masking and thinning -- keeps robust_stats honest.

    `robust_stats` takes an optional mask; here the mask and the subsampling are
    applied first so the percentile is taken on exactly the retained points.
    An empty history yields a single zero, which `robust_stats` maps to (0, 0)
    and `apply_robust` then treats as scale 1.0 -- the neutral fallback.
    """
    x = _subsample(np.asarray(x, dtype=float), mask)
    return (x if x.size else np.zeros(1)), None


# ---------------------------------------------------------------------------
# B4: an explicit CAUSAL TRAILING-BASELINE channel.
#
# B4 asks for the envelope e(t) of the SUMMED respiratory effort, supplied as
#
#       e(t) / median_{tau in [t-120s, t-15s]} e(tau)
#
# "This encodes the scoring rule directly rather than hoping the network
# rediscovers it, and it is causal by construction."
#
# WHY THIS IS NOT A REPEAT OF section 24. That script tested the same IDEA at a
# 10-30 min baseline, and the section 27.4 preview tested it against the whole
# ~6 h block. Both came back at chance. B4's window is 120 s, and the scoring
# rule it encodes says "below 20% of the amplitude of the PRECEDING BREATHS" --
# seconds to a minute, not half an hour. The short baseline is untested.
#
# WHY IT CAN HELP AT ALL, given everything else was null: the model's input
# window is 30 s, so it structurally CANNOT see further back than that. This
# channel reaches 120 s back and hands the network context from outside the
# window. It is not a rescaling of information the model already has.
EFFORT_ENV_WIN_S = 3.0      # RMS envelope, ~2-3 neonatal breath cycles
EFFORT_BASE_FROM_S = 120.0  # trailing baseline starts here...
EFFORT_BASE_TO_S = 15.0     # ...and ends here, 15 s back from t
EFFORT_MIN_HISTORY_S = 30.0 # below this the ratio is set to neutral, never dropped
EFFORT_LOG_CLIP = 3.0       # log2 ratio clipped to +-3, i.e. an 8x swing


def effort_baseline_ratio(seg, artifact_free=None, scale_floor=None):
    """B4's channel, at TARGET_FREQ, ready to slice like any other signal.

    Returns log2(e(t) / trailing median), clipped to +-EFFORT_LOG_CLIP and
    divided by it, so the channel lands in [-1, 1] alongside the range-normalised
    ones. That is a MONOTONE transform of what B4 writes, chosen only so the
    channel sits on the same numerical scale as its neighbours; the criterion the
    rule states maps to a fixed value on it (20% of baseline -> -0.774, and
    Adams' 50% -> -0.333), so nothing about the hypothesis is lost.

    THREE DECISIONS, each of which would otherwise be a silent trap:

    * BOUNDARY WINDOWS ARE NEUTRALISED, NEVER DROPPED. The first 120 s of a
      block has no full baseline. Dropping those windows would change the window
      set and break comparability with every other cell and experiment, which are
      all paired on identical windows. Instead the median is taken over whatever
      history exists down to EFFORT_MIN_HISTORY_S, and below that the ratio is
      set to 1 (log 0 = neutral, "no information"). The window table is therefore
      bit-for-bit the same as without this channel.
    * A DEAD BELT IS NEUTRALISED TOO. Patient 009 block 2's Abdomen belt is flat,
      so its envelope is ~0 and the ratio would explode. Where the trailing
      median falls below a floor derived from the block itself, the ratio is set
      to neutral rather than allowed to produce a spike the network would read as
      a precursor.
    * THE BELT WEIGHTING IS TRANSDUCTIVE, THE BASELINE IS NOT. The two belts are
      put on a common scale with robust statistics over the whole segment before
      summing (the same recipe as block_norm_stats), because an inductance belt
      has an arbitrary per-belt gain. Only the RELATIVE weighting of the two
      belts uses that; the ratio itself cancels any overall scale, and the
      trailing median is strictly backward-looking. Item B5 is where the
      transductive/causal question is settled properly.
    """
    fs = TARGET_FREQ
    n = min(len(seg["Thorax"]), len(seg["Abdomen"]))

    # Put the belts on a common scale before summing -- see docstring.
    belts = []
    for channel in ("Thorax", "Abdomen"):
        x = np.asarray(seg[channel][:n], dtype=float)
        dec = _apply_chain(x, DECIMATION_CHAIN[channel])
        stats = robust_stats(dec, _mask_to_length(artifact_free, len(dec)))
        floor = None
        if scale_floor is not None:
            floor = (scale_floor.get(channel) if hasattr(scale_floor, "get")
                     else scale_floor)
        belts.append(apply_robust(x, stats, floor))
    summed = belts[0] + belts[1]

    # RMS envelope over ~2-3 breaths, computed at TARGET_FREQ then handled at
    # 5 Hz, which is the rate the effort channels reach the network at.
    step = fs // 5
    env_win = max(1, int(EFFORT_ENV_WIN_S * fs))
    sq = pd.Series(summed ** 2).rolling(env_win, min_periods=1, center=True).mean()
    env = np.sqrt(sq.to_numpy())[::step]          # -> 5 Hz

    # Causal trailing median over [t-120s, t-15s]: a window of (120-15) s that
    # ENDS 15 s before t. The 15 s gap is what makes the feature sensitive to a
    # recent decline -- without it, a fall would be absorbed into its own baseline.
    span = int((EFFORT_BASE_FROM_S - EFFORT_BASE_TO_S) * 5)
    gap = int(EFFORT_BASE_TO_S * 5)
    min_per = max(1, int((EFFORT_MIN_HISTORY_S) * 5))
    s = pd.Series(env)
    base = s.rolling(span, min_periods=min_per).median().shift(gap)

    # Neutralise where the baseline is missing or the belts are dead.
    live = np.isfinite(env) & (env > 0)
    dead_floor = 0.05 * float(np.median(env[live])) if live.any() else 0.0
    b = base.to_numpy()
    ok = np.isfinite(b) & (b > dead_floor)
    ratio = np.ones_like(env, dtype=float)
    np.divide(env, b, out=ratio, where=ok)
    ratio[~ok] = 1.0
    ratio[ratio <= 0] = 1.0

    out = np.clip(np.log2(ratio), -EFFORT_LOG_CLIP, EFFORT_LOG_CLIP) / EFFORT_LOG_CLIP
    # Back to TARGET_FREQ so the caller can slice it exactly like any channel.
    return np.repeat(out, step)[:n]


def create_time_windows(anti_adverse_events, cutter, window_size, away, lag,
                        control_min=None, control_max=None):
    """Cut one segment into labelled windows.

    BA adaptation (C4): `control_min` / `control_max` bound how far a CONTROL
    window may sit from the nearest scored event, in samples. Both None (the
    default) reproduces the published rule exactly -- controls at least `away`
    from any event, in either direction.

    Why the option exists. Under the published rule every control is >=3 min
    from any event, so controls are long quiet stretches while positives sit
    15 s before an apnea. The classifier can then separate them by recognising
    "this infant is in an apnea-rich phase" instead of "an apnea is about to
    start", and the reported AuROC does not distinguish the two. Restricting
    controls to a band CLOSE to events (e.g. 60-180 s) removes that shortcut:
    both classes then come from the same apnea-rich stretches and only the
    imminence differs.

    `control_min` must exceed `lag + window_size`, otherwise the control band
    overlaps the region a positive window occupies (a positive spans `lag` to
    `lag + window_size` samples before the onset) and the same samples would be
    handed to the model under both labels.
    """
    # `lag` controls where the POSITIVE window sits relative to the event onset:
    #   lag > 0  -> PREDICTION: the window ends `lag` samples BEFORE the event,
    #               so the model never sees the event itself (early warning).
    #   lag <= 0 -> DETECTION:  the window ends |lag| samples AFTER the onset,
    #               i.e. it overlaps/contains the event. Used to establish the
    #               detectability upper bound (BA methodology, step "detection").
    assert away >= window_size + max(lag, 0)

    n = len(anti_adverse_events)
    increasing = sum_from_left(anti_adverse_events)
    decreasing = sum_from_right(anti_adverse_events)
    slice_ls = create_slices(cutter)

    slices_and_labels = []
    for ss in slice_ls[:-1]:  # Throw away in last slice.
        decr_min = np.min(decreasing[ss])
        decr_max = np.max(decreasing[ss]) + 1
        incr_min = np.min(increasing[ss])
        incr_max = np.max(increasing[ss]) + 1
        if lag > 0:
            # Prediction: positive window strictly before the event.
            if (decr_min <= lag) and (decr_max >= lag + window_size):
                start_pos = lag - decr_min
                slices_and_labels.append(
                    {
                        "label": 1,
                        "slice": slice(
                            ss.stop - start_pos - window_size, ss.stop - start_pos
                        ),
                    }
                )
        else:
            # Detection: positive window overlaps the event (ends |lag| after
            # onset). Requires the slice to abut an adverse event (decr_min == 0)
            # and the window to stay inside the signal.
            if decr_min <= 1:
                start_pos = lag - decr_min
                win_start = ss.stop - start_pos - window_size
                win_end = ss.stop - start_pos
                if win_start >= ss.start and win_end <= n:
                    slices_and_labels.append(
                        {"label": 1, "slice": slice(win_start, win_end)}
                    )
        if control_min is None and control_max is None:
            # Published rule, untouched.
            if (decr_max >= away + window_size) and (incr_max >= away + window_size):
                start_pos = ss.start + max(away - incr_min, 0)
                end_pos = ss.stop - max(away - decr_min, 0)
                slices_and_labels.extend(
                    [
                        {
                            "label": 0,
                            "slice": slice(
                                end_pos - (window_size * (i + 1)),
                                end_pos - (window_size * i),
                            ),
                        }
                        for i in range((end_pos - start_pos) // window_size)
                    ]
                )
        else:
            # C4: tile the slice and keep only windows whose distance to the
            # NEAREST event (either side) falls inside the requested band.
            lo = 0 if control_min is None else control_min
            hi = np.inf if control_max is None else control_max
            n_win = (ss.stop - ss.start) // window_size
            for i in range(n_win):
                w_start = ss.start + i * window_size
                w_stop = w_start + window_size
                # `increasing` counts non-event samples since the previous event
                # ended; `decreasing` counts them until the next event starts.
                dist_prev = increasing[w_start]
                dist_next = decreasing[w_stop - 1]
                dist = min(dist_prev, dist_next)
                if lo <= dist <= hi:
                    slices_and_labels.append(
                        {"label": 0, "slice": slice(w_start, w_stop)}
                    )
    return slices_and_labels


class NeoNatal(Dataset):
    def __init__(
        self,
        pat_id,
        signal_dict,
        dataset_mode,
        signal_types,
        adverse_events,
        cutter_events,
        time_window,
        lag,
        away,
        positive_cap=None,
        control_min=None,
        control_max=None,
        norm_per_window=True,
        norm_per_block=False,
        norm_scale_floor=None,
        norm_stats_mode="whole_block",
    ):
        super().__init__()
        self.pat_id = pat_id  # for logging
        self.dataset_mode = dataset_mode
        self.signal_types = signal_types
        self.adverse_events = adverse_events
        self.cutter_events = cutter_events
        self.time_window = time_window
        self.lag = lag
        self.away = away
        # BA adaptation (B2/B3): normalisation switches. The defaults are the
        # published behaviour (per-window standardisation only), so any caller
        # that does not opt in reproduces exactly.
        # BA adaptation (C4). Both None reproduces the published control rule.
        self.control_min = control_min
        self.control_max = control_max
        if control_min is not None and control_min <= lag + time_window:
            raise ValueError(
                f"control_min ({control_min}) must exceed lag + time_window "
                f"({lag + time_window}): a control any closer to an event "
                "overlaps the samples a positive window is cut from."
            )
        self.norm_per_window = norm_per_window
        self.norm_per_block = norm_per_block
        self.norm_scale_floor = norm_scale_floor
        # BA adaptation (B5): where the per-block statistics come from.
        # 'whole_block' is transductive and is the pre-specified PRIMARY; the
        # other two are causal and match real-time use. Default reproduces.
        if norm_stats_mode not in NORM_STATS_MODES:
            raise ValueError(
                f"norm_stats_mode must be one of {NORM_STATS_MODES}, "
                f"got {norm_stats_mode!r}"
            )
        if norm_stats_mode != "whole_block" and not norm_per_block:
            # Failing loudly beats silently ignoring the option: a causal mode
            # with per-block normalisation switched off does nothing at all, and
            # a config that says 'causal' while running the transductive arm
            # would put a wrong sentence in the paper.
            raise ValueError(
                f"norm_stats_mode={norm_stats_mode!r} has no effect unless "
                "norm_per_block is True -- B5's causal statistics ARE the "
                "per-block statistics, computed from history instead of the "
                "whole block."
            )
        self.norm_stats_mode = norm_stats_mode
        # BA adaptation (B2 bullet 7). The deviation channels ARE the per-block
        # transform applied to HR/SpO2, so without per-block statistics they
        # would enter in raw bpm and percent -- two orders of magnitude off the
        # scale of every other channel, and silently so. Fail loudly instead.
        dev_requested = [s for s in signal_types if s in DEV_CHANNELS]
        if dev_requested and not norm_per_block:
            raise ValueError(
                f"{dev_requested} require norm_per_block=True: these channels "
                "are the per-block robust-centred HR/SpO2, and without block "
                "statistics they would enter in raw physical units."
            )
        if dev_requested and norm_stats_mode != "whole_block":
            # Not a limitation worth building around: bullet 7 is a B2 question
            # and B5's causal machinery does not carry the dev channels.
            raise ValueError(
                f"{dev_requested} are only implemented for "
                "norm_stats_mode='whole_block' (B2 bullet 7 is independent of "
                f"B5); got {norm_stats_mode!r}."
            )
        if norm_per_block and dataset_mode == "features":
            # feature_engineering() has its own transforms and was not part of
            # the B3 ablation; failing loudly beats silently normalising only
            # half the pipeline.
            raise ValueError(
                "norm_per_block is implemented for dataset_mode 'list'/'stacked' "
                "(feature_extraction), not for 'features' (feature_engineering)."
            )
        # read_data returns a list of signal segments (one per ANALYSIS
        # block). A plain dict (e.g. loaded from a pickle) is wrapped so the
        # original single-segment behaviour is preserved.
        segments = [signal_dict] if isinstance(signal_dict, dict) else signal_dict

        # Build the time-window table independently for every segment and
        # concatenate. Because each segment is processed on its own arrays,
        # neither a time window nor an event distance ever crosses a segment
        # boundary (i.e. an ANALYSIS-STOP / ANALYSIS-START gap).
        segment_dfs = []
        for si, seg in enumerate(segments):
            # Adverse events on which distances are calculated.
            anti_adverse = 1 - np.column_stack(
                [seg[event] for event in self.adverse_events]
            ).max(axis=1)

            # Adverse events plus events that must not be part of the windows.
            cutter = 1 - np.column_stack(
                [seg[event] for event in (self.adverse_events + self.cutter_events)]
            ).max(axis=1)

            seg_df = pd.DataFrame(
                create_time_windows(
                    anti_adverse_events=anti_adverse,
                    cutter=cutter,
                    window_size=self.time_window,
                    away=self.away,
                    lag=self.lag,
                    control_min=self.control_min,
                    control_max=self.control_max,
                )
            )
            if seg_df.empty:
                continue

            # Remember which segment (and, if known, which wall-clock time)
            # each window came from, so a window can later be traced back to
            # a real timestamp in the original recording (e.g. to pull the
            # raw signal for a specific true/false-positive example).
            seg_df["segment_idx"] = si
            seg_start_ts = seg.get("_SEGMENT_START_TS")
            if seg_start_ts is not None:
                seg_df["abs_window_start"] = (
                    seg_start_ts + seg_df["slice"].map(lambda sl: sl.start) / TARGET_FREQ
                )
                seg_df["abs_window_end"] = (
                    seg_start_ts + seg_df["slice"].map(lambda sl: sl.stop) / TARGET_FREQ
                )

            # BA adaptation (B2): per-block robust statistics for this
            # segment. Computed ONCE per block, before any window is cut, over
            # the artifact-free samples only. The mask deliberately uses the
            # CUTTER events alone and never the adverse events -- see
            # block_norm_stats for why that distinction is load-bearing.
            block_stats = None
            causal_prep = None
            if self.norm_per_block:
                artifact_free = None
                if self.cutter_events:
                    artifact_free = (
                        1
                        - np.column_stack(
                            [seg[event] for event in self.cutter_events]
                        ).max(axis=1)
                    ).astype(bool)
                if self.norm_stats_mode == "whole_block":
                    block_stats = block_norm_stats(
                        seg,
                        self.signal_types,
                        artifact_free=artifact_free,
                        scale_floor=self.norm_scale_floor,
                        pat_id=self.pat_id,
                        segment_idx=si,
                    )
                else:
                    # BA adaptation (B5): causal statistics. The block is
                    # decimated ONCE here; each window then takes its median and
                    # IQR from its own history in `causal_norm_stats_at`. The
                    # whole-block statistics are still computed and logged, so
                    # the QC pass (dead belts, floored scales) happens in the
                    # causal arms too rather than being silently skipped.
                    block_norm_stats(
                        seg,
                        self.signal_types,
                        artifact_free=artifact_free,
                        scale_floor=self.norm_scale_floor,
                        pat_id=self.pat_id,
                        segment_idx=si,
                    )
                    causal_prep = prepare_causal_norm(
                        seg, self.signal_types, artifact_free=artifact_free
                    )

            # BA adaptation (B4): the causal trailing-baseline channel. Built
            # ONCE per block -- a rolling median over 105 s at 5 Hz is far too
            # expensive to recompute per window -- and written into `seg` so
            # feature_extraction can slice it exactly like any other signal.
            if "EffortRatio" in self.signal_types:
                af = None
                if self.cutter_events:
                    af = (
                        1
                        - np.column_stack(
                            [seg[event] for event in self.cutter_events]
                        ).max(axis=1)
                    ).astype(bool)
                seg["EffortRatio"] = effort_baseline_ratio(
                    seg, artifact_free=af, scale_floor=self.norm_scale_floor
                )

            # Process (extract) the signal windows using this segment's data.
            if dataset_mode == "list" or dataset_mode == "stacked":
                # B5: in a causal mode the statistics differ PER WINDOW, so they
                # are resolved here rather than once per block. feature_extraction
                # is unchanged -- it receives the same {channel: (median, scale)}
                # dict either way.
                def _stats_for(row):
                    if causal_prep is None:
                        return block_stats
                    return causal_norm_stats_at(
                        causal_prep,
                        row["slice"].start,
                        self.norm_stats_mode,
                        scale_floor=self.norm_scale_floor,
                    )

                seg_df["sig"] = seg_df.apply(
                    lambda row: self.feature_extraction(
                        row["slice"], seg, block_stats=_stats_for(row)
                    ),
                    axis=1,
                )
            elif dataset_mode == "features":
                seg_df["sig"] = seg_df.apply(
                    lambda row: self.feature_engineering(row["slice"], seg), axis=1
                )
            else:
                raise ValueError
            segment_dfs.append(seg_df)

        self.time_window_df = (
            pd.concat(segment_dfs, ignore_index=True)
            if segment_dfs
            else pd.DataFrame(columns=["label", "slice", "sig"])
        )

        # BA control experiment: optionally thin the positive class to a fixed
        # count. Restricting the Robin cohort to central apnea only drops the
        # positives from ~1700 to ~272, so a lower AuROC could equally well be
        # caused by the event type OR by data scarcity. Capping the positives
        # of the *full* task to the same count holds the event type fixed and
        # varies only the amount of data, which separates the two.
        # `positive_cap` is either an int or a {patient_id: int} mapping;
        # `None` (the default) leaves every previous run untouched.
        cap = self._resolve_positive_cap(positive_cap)
        if cap is not None and len(self.time_window_df):
            pos_idx = self.time_window_df.index[self.time_window_df["label"] == 1]
            if len(pos_idx) > cap:
                # Uses the global numpy seed, so each training seed draws a
                # different subset and the result is not an artefact of one
                # lucky or unlucky sample.
                keep = np.random.choice(np.asarray(pos_idx), size=cap, replace=False)
                drop = pos_idx.difference(pd.Index(keep))
                self.time_window_df = self.time_window_df.drop(drop).reset_index(
                    drop=True
                )
                log.info(f"positive_cap: kept {cap} of {len(pos_idx)} positive windows")

        log.info(f"Number of time windows: {len(self.time_window_df)}")
        if len(self.time_window_df):
            log.info(
                f"Imbalance: {self.time_window_df['label'].sum() / len(self.time_window_df)}"
            )

    def _scale_floor_for(self, channel):
        """Lower bound on this channel's robust scale, or None."""
        floor = self.norm_scale_floor
        if floor is None:
            return None
        return floor.get(channel) if hasattr(floor, "get") else floor

    def _resolve_positive_cap(self, positive_cap):
        """Cap for this patient: an int, a {pat_id: int} mapping, or None."""
        if positive_cap is None:
            return None
        if isinstance(positive_cap, int):
            return positive_cap
        # OmegaConf DictConfig or plain dict, keyed by patient id.
        if self.pat_id not in positive_cap:
            raise KeyError(
                f"positive_cap has no entry for patient {self.pat_id}; it must "
                "cover every id in dataset.ids (or be a single int)."
            )
        return int(positive_cap[self.pat_id])

    # Get undersampled elements for training.
    def __getitem__(self, index):
        if self.dataset_mode == "stacked":
            return self._get_stacked(index)
        elif self.dataset_mode == "list":
            return self._get_list(index)
        elif self.dataset_mode == "features":
            return self._get_features(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.time_window_df)

    # Needs to be implemented for BalancedSamlping
    def _get_labels(self):
        return self.time_window_df["label"].to_numpy()

    # Stacked mode requires that all signals have the same length.
    # Or that only one channel is used.
    def _get_stacked(self, index):
        index_row = self.time_window_df.iloc[index]
        sig_list = index_row["sig"]
        if len(sig_list) > 1:
            max_len = max([len(sig) for sig in sig_list])
            sig_list = [np.repeat(sig, max_len // len(sig)) for sig in sig_list]
        x_s = torch.from_numpy(np.row_stack(sig_list).astype(np.float32))
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_list(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = [
            torch.from_numpy(ind.astype(np.float32)).unsqueeze(0)
            for ind in index_row["sig"]
        ]
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_features(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = torch.from_numpy(index_row["sig"].astype(np.float32))
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def feature_extraction(self, ss, signal_dict, block_stats=None):
        lower_range = -1.0
        upper_range = 1.0

        # BA adaptation (B2/B3). Two independent normalisation stages:
        #   block  -- (x - median_b) / (IQR_b / 1.349), statistics from the whole
        #             block, computed once in __init__ (block_stats)
        #   window -- the published per-30 s-window z-score (`standardize`)
        # `_osc` is the single-channel case; `_belt` is the inner per-belt step
        # of the summed respiratory effort, which the published recipe
        # standardises SEPARATELY before summing. Keeping the inner and outer
        # steps switchable apart is what makes the block-off/window-on cell
        # reproduce the published pipeline bit-for-bit.
        def _belt(dec, channel):
            if block_stats is not None:
                return apply_robust(
                    dec, block_stats[channel], self._scale_floor_for(channel)
                )
            return standardize(dec) if self.norm_per_window else dec

        def _osc(dec, channel):
            if block_stats is not None:
                dec = apply_robust(
                    dec, block_stats[channel], self._scale_floor_for(channel)
                )
            return standardize(dec) if self.norm_per_window else dec

        processed_windows = []
        for signal_type in self.signal_types:
            processed_window = None

            # 5 Hz
            if signal_type == "NP":
                np_decimate_one = 10
                np_decimate_two = 4
                processed_window = _osc(
                    decimate(
                        decimate(signal_dict["NP"][ss], np_decimate_one),
                        np_decimate_two,
                    ),
                    "NP",
                )
            # 5 Hz -- CPAP mask pressure, own channel (see SIGNAL_LABEL_MAP note)
            elif signal_type == "CPAP":
                cpap_decimate_one = 10
                cpap_decimate_two = 4
                processed_window = _osc(
                    decimate(
                        decimate(signal_dict["CPAP"][ss], cpap_decimate_one),
                        cpap_decimate_two,
                    ),
                    "CPAP",
                )
            # 5Hz
            elif signal_type == "Thorax":
                thorax_select = 4
                thorax_decimate = 10
                # Summed respiratory effort: normalise each belt, sum pointwise,
                # then normalise the sum. Under per-block statistics the sum
                # needs its OWN block median/IQR ("Thorax_sum"), because adding
                # two normalised belts does not leave the result on unit scale.
                thorax_dec = decimate(
                    signal_dict["Thorax"][ss][::thorax_select],
                    thorax_decimate,
                )
                abdomen_dec = decimate(
                    signal_dict["Abdomen"][ss][::thorax_select],
                    thorax_decimate,
                )
                summed = _belt(thorax_dec, "Thorax") + _belt(abdomen_dec, "Abdomen")
                if block_stats is not None:
                    summed = apply_robust(
                        summed,
                        block_stats["Thorax_sum"],
                        self._scale_floor_for("Thorax_sum"),
                    )
                processed_window = (
                    standardize(summed) if self.norm_per_window else summed
                )
            # 5 Hz -- B4's causal trailing-baseline channel.
            #
            # DELIBERATELY NOT NORMALISED, and this is the whole point. The
            # channel is already a scale-free RATIO. Per-window standardisation
            # subtracts the window mean and divides by the window SD, which would
            # erase exactly the information it exists to carry -- the same
            # mechanism section 27.6 measured at -0.025 for the effort channels.
            # Applying it here would guarantee a null and invite the wrong
            # conclusion. Plain subsampling, no filter: the ratio is already
            # smooth (an RMS envelope over a rolling median), so decimate's
            # anti-alias stage would buy nothing.
            elif signal_type == "EffortRatio":
                effort_ratio_select = 40   # 200 Hz -> 5 Hz, matching the belts
                processed_window = signal_dict["EffortRatio"][ss][::effort_ratio_select]
            # 1Hz
            elif signal_type == "HR":
                hr_select = 200
                hr_lower = 50.0
                hr_upper = 240.0
                processed_window = normalize_range(
                    signal_dict["HR"][ss][::hr_select],  # normal mode
                    hr_lower,
                    hr_upper,
                    lower_range,
                    upper_range,
                )
            # 5Hz
            elif signal_type == "PR":
                pr_decimate_one = 10
                pr_decimate_two = 4
                processed_window = _osc(
                    decimate(
                        decimate(signal_dict["PR"][ss], pr_decimate_one),
                        pr_decimate_two,
                    ),
                    "PR",
                )
            # 1Hz
            elif signal_type == "SpO2":
                spo2_select = 100
                spo2_decimate = 2
                spo2_lower = 60.0
                spo2_upper = 100.0
                processed_window = normalize_range(
                    decimate(
                        signal_dict["SpO2"][ss][::spo2_select],
                        spo2_decimate,
                    ),
                    spo2_lower,
                    spo2_upper,
                    lower_range,
                    upper_range,
                )
            # B2 bullet 7: the ADDED per-block robust-centred channels. The
            # fixed-range HR/SpO2 channels above are untouched and remain
            # primary; these carry the deviation from this infant's own baseline
            # on this device, which the fixed-range map cannot express because
            # its offset is the same for every infant.
            #
            # NOT range-normalised and NOT clipped: after the robust transform
            # the channel is already on unit scale (1.0 ~ one IQR/1.349 of that
            # infant's own HR), which is the scale the network's other
            # block-normalised channels arrive on.
            elif signal_type in DEV_CHANNELS:
                base = DEV_CHANNELS[signal_type]
                dec = _apply_chain(
                    signal_dict[base][ss], DEV_DECIMATION_CHAIN[base]
                )
                processed_window = _osc(dec, signal_type)
            # 1Hz
            elif signal_type == "PCO2":
                pco2_select = 100
                pco2_decimate = 2
                pco2_lower = 30.0
                pco2_upper = 70.0
                processed_window = normalize_range(
                    decimate(
                        signal_dict["PCO2"][ss][::pco2_select],
                        pco2_decimate,
                    ),
                    pco2_lower,
                    pco2_upper,
                    lower_range,
                    upper_range,
                )
            else:
                raise ValueError

            processed_windows.append(processed_window)

        return processed_windows

    def feature_engineering(self, ss, signal_dict):
        lower_range = -1.0
        upper_range = 1.0

        processed_windows = []
        for signal_type in self.signal_types:
            processed_window = None

            if signal_type == "NP":
                np_decimate_one = 10
                np_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(signal_dict["NP"][ss], np_decimate_one),
                        np_decimate_two,
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "CPAP":
                cpap_decimate_one = 10
                cpap_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(signal_dict["CPAP"][ss], cpap_decimate_one),
                        cpap_decimate_two,
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "Thorax":
                thorax_select = 4
                thorax_decimate = 10
                processed_window = standardize(
                    standardize(
                        decimate(
                            signal_dict["Thorax"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                    + standardize(
                        decimate(
                            signal_dict["Abdomen"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "PR":
                pr_decimate_one = 10
                pr_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(signal_dict["PR"][ss], pr_decimate_one),
                        pr_decimate_two,
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "HR":
                hr_select = 200
                hr_lower = 50.0
                hr_upper = 240.0
                processed_window = normalize_range(
                    signal_dict["HR"][ss][::hr_select],
                    hr_lower,
                    hr_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            elif signal_type == "SpO2":
                spo2_select = 100
                spo2_decimate = 2
                spo2_lower = 60.0
                spo2_upper = 100.0
                processed_window = normalize_range(
                    decimate(
                        signal_dict["SpO2"][ss][::spo2_select],
                        spo2_decimate,
                    ),
                    spo2_lower,
                    spo2_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            elif signal_type == "PCO2":
                pco2_select = 100
                pco2_decimate = 2
                pco2_lower = 30.0
                pco2_upper = 70.0
                processed_window = normalize_range(
                    decimate(
                        signal_dict["PCO2"][ss][::pco2_select],
                        pco2_decimate,
                    ),
                    pco2_lower,
                    pco2_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            else:
                raise ValueError
            processed_windows.extend(features)

        return np.array(processed_windows)


# NOTE: C1==C2 is not garantueed. Only in expectation.
class BalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, replacement=False):
        labels = dataset._get_labels()

        classes, counts = np.unique(labels, return_counts=True)
        weights = np.zeros_like(labels, dtype=float)
        for val, nums in zip(classes, counts):
            weights[labels == val] = 1.0 / nums
        num_samples = int(min(counts) * len(counts))

        super().__init__(
            weights=weights, num_samples=num_samples, replacement=replacement
        )

    def __iter__(self):
        return super().__iter__()

    def __len__(self):
        return super().__len__()


class MultiDatasetBalancedSampler(Sampler):
    """Class-balanced sampling within each dataset of a ConcatDataset.

    BA adaptation (F3): `equal_per_dataset` additionally equalises how much each
    INFANT contributes to an epoch.

    The published behaviour balances the two classes inside each infant but not
    the infants against each other: an infant contributes 2*min(n_pos, n_neg)
    windows per epoch, which is driven by its apnea count. Measured on this
    cohort that runs from 44 windows (patient 012) to 612 (patient 015) -- a
    14-fold spread in which three infants supply 45% of all training data.

    That matters because the reported metric is a MEAN OVER INFANTS, in which
    every infant counts once. Training therefore optimises a different weighting
    from the one being reported. With `equal_per_dataset=True` each infant
    contributes the same number of windows, still class-balanced internally, so
    the two agree.

    Note this does not permanently discard data the way a fixed positive cap
    does: a different random subset is drawn every epoch, so over a training run
    the model still sees the busy infants' full window set.
    """

    def __init__(self, concat_dataset, replacement=False, equal_per_dataset=False,
                 target_per_dataset=None):
        self.concat_dataset = concat_dataset
        self.replacement = replacement
        self.equal_per_dataset = equal_per_dataset
        # Windows PER CLASS each dataset contributes when equalising. None ->
        # the median of min(n_pos, n_neg) across datasets, which neither starves
        # the model (as the minimum would) nor heavily oversamples the small
        # infants (as the maximum would).
        self.target_per_dataset = target_per_dataset

    def _per_class_target(self):
        if self.target_per_dataset is not None:
            return int(self.target_per_dataset)
        sizes = []
        for dat in self.concat_dataset.datasets:
            labels = np.asarray(dat._get_labels())
            _, counts = np.unique(labels, return_counts=True)
            if len(counts) >= 2:
                sizes.append(int(counts.min()))
        return int(np.median(sizes)) if sizes else 0

    def _equal_indices(self):
        n_per_class = self._per_class_target()
        offset = 0
        all_indices = []
        for dat in self.concat_dataset.datasets:
            labels = np.asarray(dat._get_labels())
            for cls in np.unique(labels):
                idx = np.where(labels == cls)[0]
                if len(idx) == 0:
                    continue
                # Sample without replacement where the class is large enough,
                # with replacement where it is not, so every infant reaches the
                # same count either way.
                replace = len(idx) < n_per_class
                pick = np.random.choice(idx, size=n_per_class, replace=replace)
                all_indices.extend((pick + offset).tolist())
            offset += len(dat)
        shuffle(all_indices)
        return all_indices

    def __iter__(self):
        if self.equal_per_dataset:
            yield from iter(self._equal_indices())
            return
        offset = 0
        all_indices = []
        for dat in self.concat_dataset.datasets:
            indices = list(BalancedSampler(dat, replacement=self.replacement))
            offset_indicies = [idx + offset for idx in indices]
            all_indices.extend(offset_indicies)
            offset += len(dat)
        shuffle(all_indices)

        yield from iter(all_indices)

    def __len__(self):
        if self.equal_per_dataset:
            n_classes = 0
            for dat in self.concat_dataset.datasets:
                n_classes += len(np.unique(np.asarray(dat._get_labels())))
            return self._per_class_target() * n_classes
        length = 0
        for dat in self.concat_dataset.datasets:
            length += len(BalancedSampler(dat, replacement=self.replacement))
        return length


def skewness(signal):
    return scipy.stats.skew(signal)


def kurtosis(signal):
    return scipy.stats.kurtosis(signal)


def calc_fft(signal, fs):
    fmag = np.abs(np.fft.rfft(signal))
    f = np.fft.rfftfreq(len(signal), d=1 / fs)

    return f.copy(), fmag.copy()


def spectral_centroid(signal, fs):
    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


def spectral_spread(signal, fs):
    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5


def spectral_skewness(signal, fs):
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


def spectral_kurtosis(signal, fs):
    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)
