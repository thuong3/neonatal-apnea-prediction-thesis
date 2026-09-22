"""Cross-cohort domain adaptation: Robin sequence (source) -> CPAP (target).

Runs a ladder of transfer settings, all evaluated under the SAME
leave-one-patient-out split over the target cohort so that every number in the
final table is directly comparable:

    target_only    fresh model, target training patients only     (control)
    zero_shot      source-trained model applied unchanged         (size of shift)
    pooled         fresh model, source + target training data
    finetune_full  source-pretrained, all weights updated
    finetune_head  source-pretrained, conv layers frozen (linear probe)
    adabn          source-pretrained, only BatchNorm statistics re-estimated
    coral          source-pretrained, fine-tuned with CORAL feature alignment

The two cohorts share exactly five channels (Thorax, HR, PR, SpO2, PCO2);
neither respiratory-pressure channel is common to both, so both are dropped.
See Section 4.5 of the thesis for the rationale and the reading of the results.

Usage:
    python src/train_domain_adaptation.py meta.experiment=da meta.tag=da_01 meta.seed=0
"""

import copy
import logging
import os
import pickle
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.misc_utils import tensor_to_array
from src.neonatal_utils import (
    BalancedSampler,
    MultiDatasetBalancedSampler,
    NeoNatal,
    load_clock_drift,
    read_data,
)
from src.train_neonatal import init_classifier, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def init_cohort_dataset(pat_id, cohort_cfg, cfg):
    """Build one patient's windowed dataset from the given cohort.

    The windowing parameters come from the shared `dataset` block (identical
    for both cohorts, so the task definition does not change between domains),
    while the event vocabulary and the file layout come from the cohort.
    """
    # Per cohort, and only where it was measured: `clock_drift_file` is null in
    # config/cohort/robin.yaml because the skew was measured on the CPAP export
    # alone, and the two cohorts' patient ids overlap.
    drift = load_clock_drift(cohort_cfg.get("clock_drift_file", None))
    signal_dict = read_data(
        pat_id,
        cohort_cfg.data_path,
        annotations_dir=cohort_cfg.annotations_dir,
        record_duration=cohort_cfg.record_duration,
        clock_drift=drift.get(pat_id),
    )
    return NeoNatal(
        pat_id,
        signal_dict,
        dataset_mode=cfg.dataset.dataset_mode,
        signal_types=cfg.dataset.signal_types,
        adverse_events=cohort_cfg.adverse_events,
        cutter_events=cohort_cfg.cutter_events,
        time_window=cfg.dataset.time_window,
        lag=cfg.dataset.lag,
        away=cfg.dataset.away,
        # BA adaptation (B2/B3): normalisation ablation. Absent from every
        # dataset config except the four norm_* experiments, so the defaults
        # reproduce the published pipeline and every earlier run is unaffected.
        norm_per_window=cfg.dataset.get("norm_per_window", True),
        norm_per_block=cfg.dataset.get("norm_per_block", False),
        norm_scale_floor=cfg.dataset.get("norm_scale_floor", None),
        # BA adaptation (B5): whole_block (transductive, the primary) vs
        # expanding/trailing (causal, real-time). Default reproduces.
        norm_stats_mode=cfg.dataset.get("norm_stats_mode", "whole_block"),
    )


def build_cohort(cohort_cfg, cfg):
    datasets = {}
    for pat_id in cohort_cfg.ids:
        log.info(f"[{cohort_cfg.name}] processing id {pat_id}")
        dataset = init_cohort_dataset(pat_id, cohort_cfg, cfg)
        labels = dataset._get_labels()
        n_pos = int(np.sum(labels))
        # A patient with only one class contributes no usable gradient signal
        # and cannot be scored with AUC; drop it explicitly rather than
        # letting it fail later inside the sampler or roc_auc_score.
        if n_pos == 0 or n_pos == len(labels):
            log.warning(
                f"[{cohort_cfg.name}] id {pat_id}: {n_pos}/{len(labels)} positive "
                "-> single-class, excluded"
            )
            continue
        log.info(
            f"[{cohort_cfg.name}] id {pat_id}: {len(dataset)} windows, "
            f"{n_pos} positive ({100 * n_pos / len(labels):.1f}%)"
        )
        datasets[pat_id] = dataset
    return datasets


def to_device(xs):
    if isinstance(xs, list):
        return [sig.to(device) for sig in xs]
    return xs.to(device)


def make_loader(dataset, batch_size, balanced=True):
    """Class-balanced loader. Picks the right sampler for concat vs single."""
    if not balanced:
        return DataLoader(dataset, batch_size, shuffle=True)
    if isinstance(dataset, ConcatDataset):
        sampler = MultiDatasetBalancedSampler(dataset, replacement=False)
    else:
        sampler = BalancedSampler(dataset, replacement=False)
    return DataLoader(dataset, batch_size, sampler=sampler, pin_memory=True)


def _cycle(loader):
    """Endlessly repeat a loader (for drawing source batches during CORAL)."""
    while True:
        yield from loader


# --------------------------------------------------------------------------
# Domain-adaptation pieces
# --------------------------------------------------------------------------
def nam_features(classifier, x_ls):
    """Shared representation of a NAM: each module's pooled conv features.

    This is the layer just below the per-channel linear read-out, i.e. the
    deepest point at which the model still holds a distributed representation
    rather than a single logit -- the natural place to align two domains.
    Shape: (batch, sum(hidden_channels)).
    """
    return torch.cat(
        [
            torch.mean(module.only_conv(x), dim=2)
            for module, x in zip(classifier.module_list, x_ls)
        ],
        dim=1,
    )


def coral_loss(source_feats, target_feats):
    """CORAL: squared Frobenius distance between feature covariances.

    Sun & Saenko (2016), "Deep CORAL: Correlation Alignment for Deep Domain
    Adaptation". Aligns the second-order statistics of the two domains'
    representations; the 1/(4 d^2) normalisation is theirs.
    """

    def cov(feats):
        n = feats.shape[0]
        centred = feats - feats.mean(dim=0, keepdim=True)
        return centred.t() @ centred / max(n - 1, 1)

    d = source_feats.shape[1]
    return ((cov(source_feats) - cov(target_feats)) ** 2).sum() / (4 * d * d)


def head_parameters(classifier):
    """The per-channel linear read-out plus the additive combination weights.

    Freezing everything else turns fine-tuning into a linear probe on the
    transferred features: it asks whether the source representation is useful
    for the target at all, separately from whether it can be reshaped.
    """
    params = list(classifier.multiplier_list) + [classifier.bias]
    for module in classifier.module_list:
        params += list(module.lin_comb.parameters())
    return params


def freeze_feature_extractor(classifier):
    for module in classifier.module_list:
        for part in (module.conv_pool, module.final_conv):
            for param in part.parameters():
                param.requires_grad_(False)


def set_train_mode(classifier, freeze_features):
    """train() everywhere, except a frozen extractor which stays in eval().

    Keeping the frozen convolutions in eval() matters: otherwise their
    BatchNorm running statistics would keep drifting towards the target data
    even though their weights are fixed, which would quietly turn a linear
    probe into a partial adaptation and confound the comparison with `adabn`.
    """
    classifier.train()
    if freeze_features:
        for module in classifier.module_list:
            module.conv_pool.eval()
            module.final_conv.eval()


def adapt_batchnorm(classifier, dataset, batch_size, balanced=True):
    """AdaBN: re-estimate BatchNorm statistics on the target, weights fixed.

    Li et al. (2017), "Revisiting Batch Normalization for Practical Domain
    Adaptation". The cheapest possible domain adaptation -- it changes no
    learned parameter, only the per-channel means/variances that normalisation
    divides by, which are exactly the domain-specific part of a BN network.
    Setting momentum to None makes BatchNorm accumulate a cumulative average,
    i.e. the exact statistics of the data it sees rather than an EMA.

    `balanced=False` is required for the single-patient calibration period used
    by `adabn_patient`: a balanced sampler needs the labels, which would make an
    otherwise label-free method supervised, and it cannot be built at all if the
    calibration window happens to contain no event. The cohort-level `adabn`
    keeps the balanced default so its published numbers stay reproducible.
    """
    for module in classifier.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.reset_running_stats()
            module.momentum = None

    classifier.train()
    loader = make_loader(dataset, batch_size, balanced=balanced)
    with torch.no_grad():
        for xs, _ys in loader:
            classifier(to_device(xs))
    classifier.eval()
    return classifier


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def fit(
    classifier,
    train_dataset,
    cfg,
    epochs,
    lr,
    trainable=None,
    freeze_features=False,
    coral_source=None,
    coral_lambda=0.0,
    tag="",
):
    params = trainable if trainable is not None else list(classifier.parameters())
    optimizer = optim.AdamW(
        params, lr=lr, weight_decay=cfg.optimizer.weight_decay
    )
    criterion = nn.BCELoss()
    loader = make_loader(train_dataset, cfg.optimizer.train_batch_size)

    source_iter = None
    if coral_source is not None and coral_lambda > 0.0:
        source_iter = _cycle(
            make_loader(coral_source, cfg.optimizer.train_batch_size)
        )

    for epoch in range(epochs):
        set_train_mode(classifier, freeze_features)
        agg_loss, agg_coral, n_seen = 0.0, 0.0, 0
        epoch_ys, epoch_scores = [], []

        for xs, ys in loader:
            xs = to_device(xs)
            ys = ys.to(device)

            y_hats = classifier(xs)
            loss = criterion(y_hats, ys)

            if source_iter is not None:
                source_xs, _ = next(source_iter)
                # Separate forward passes: mixing the domains inside one batch
                # would let BatchNorm average them together and hide the very
                # shift CORAL is meant to penalise.
                penalty = coral_loss(
                    nam_features(classifier, to_device(source_xs)),
                    nam_features(classifier, xs),
                )
                loss = loss + coral_lambda * penalty
                agg_coral += penalty.item() * ys.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_seen += ys.shape[0]
            agg_loss += loss.item() * ys.shape[0]
            epoch_ys.append(tensor_to_array(ys).flatten())
            epoch_scores.append(tensor_to_array(y_hats).flatten())

        train_auc = _safe_auc(
            np.concatenate(epoch_ys), np.concatenate(epoch_scores)
        )
        # Scientific notation, and the weighted contribution next to it: the
        # raw CORAL value is ~1e-4 here, so a plain %.4f would print 0.0000 and
        # hide a mis-scaled coral_lambda (see config/da/default.yaml).
        mean_coral = agg_coral / max(n_seen, 1)
        coral_msg = (
            f" coral={mean_coral:.3e} (x lambda = {coral_lambda * mean_coral:.4f})"
            if source_iter
            else ""
        )
        log.info(
            f"    {tag} epoch {epoch + 1}/{epochs}: "
            f"loss={agg_loss / max(n_seen, 1):.4f} train_auc={train_auc:.3f}"
            f"{coral_msg}"
        )

    return classifier


def _safe_auc(labels, scores):
    """AUC that returns nan instead of raising when a fold has one class."""
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


def predict(classifier, dataset, batch_size=4096):
    classifier.eval()
    all_ys, all_scores = [], []
    with torch.no_grad():
        for xs, ys in DataLoader(dataset, batch_size):
            y_hats = classifier(to_device(xs))
            all_ys.append(ys.numpy().flatten())
            all_scores.append(tensor_to_array(y_hats).flatten())
    return np.concatenate(all_ys), np.concatenate(all_scores)


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------
def pretrain_on_source(cfg, source_datasets):
    """Train once on the whole source cohort; reused by every target fold.

    No leakage: the source cohort contains none of the target patients, so the
    same pre-trained model is legitimate for all leave-one-out folds.
    """
    log.info(f"Pre-training on {len(source_datasets)} source patients")
    classifier = init_classifier(cfg).to(device)
    train_dataset = ConcatDataset(list(source_datasets.values()))
    fit(
        classifier,
        train_dataset,
        cfg,
        epochs=cfg.da.pretrain_epochs,
        lr=cfg.optimizer.learning_rate,
        tag="[pretrain]",
    )
    return classifier


def chronological_split(dataset, calib_frac):
    """Split ONE patient's windows into an early calibration part and a later
    evaluation part, in recording order.

    Order is taken from (segment_idx, slice.start) rather than from the DataFrame
    index, so it does not depend on how `time_window_df` happened to be built.
    Splitting chronologically rather than at random is what makes the per-patient
    calibration honest: at random, the calibration set would contain windows from
    *after* the evaluation windows, i.e. the model would be adapted on its own
    future.
    """
    df = dataset.time_window_df
    order = np.lexsort((df["slice"].map(lambda sl: sl.start).to_numpy(),
                        df["segment_idx"].to_numpy()))
    n_calib = int(round(len(order) * calib_frac))
    return Subset(dataset, order[:n_calib].tolist()), Subset(
        dataset, order[n_calib:].tolist()
    )


def run_variant(
    variant, cfg, pretrained_state, source_datasets, train_datasets, test_dataset
):
    """Produce (labels, scores) on the held-out target patient for one variant."""
    batch_size = cfg.optimizer.train_batch_size
    pooled_train = ConcatDataset(train_datasets)

    # The two per-patient variants score only the LATE part of the held-out
    # recording, because the early part was spent on calibration. Their numbers
    # are therefore NOT comparable with the rest of the ladder, which scores the
    # whole patient -- which is exactly why `zero_shot_late` exists: it is the
    # same unadapted model on the same late windows, so the pair isolates the
    # effect of the per-patient adaptation and nothing else.
    if variant in ("adabn_patient", "zero_shot_late"):
        calib_set, eval_set = chronological_split(
            test_dataset, cfg.da.patient_calib_frac
        )
        classifier = init_classifier(cfg).to(device)
        classifier.load_state_dict(pretrained_state)
        if variant == "adabn_patient" and len(calib_set):
            # Label-free: only this infant's own early windows, no events used.
            adapt_batchnorm(classifier, calib_set, batch_size, balanced=False)
        return predict(classifier, eval_set)

    if variant == "target_only":
        classifier = init_classifier(cfg).to(device)
        fit(
            classifier,
            pooled_train,
            cfg,
            epochs=cfg.optimizer.epochs,
            lr=cfg.optimizer.learning_rate,
            tag="[target_only]",
        )

    elif variant == "zero_shot":
        classifier = init_classifier(cfg).to(device)
        classifier.load_state_dict(pretrained_state)

    elif variant == "pooled":
        classifier = init_classifier(cfg).to(device)
        source_subset = list(source_datasets.values())
        if cfg.da.pooled_max_source_patients is not None:
            source_subset = source_subset[: cfg.da.pooled_max_source_patients]
        fit(
            classifier,
            ConcatDataset(source_subset + list(train_datasets)),
            cfg,
            epochs=cfg.optimizer.epochs,
            lr=cfg.optimizer.learning_rate,
            tag="[pooled]",
        )

    elif variant == "adabn":
        classifier = init_classifier(cfg).to(device)
        classifier.load_state_dict(pretrained_state)
        adapt_batchnorm(classifier, pooled_train, batch_size)

    elif variant in ("finetune_full", "finetune_head", "coral"):
        classifier = init_classifier(cfg).to(device)
        classifier.load_state_dict(pretrained_state)
        freeze = variant == "finetune_head"
        if freeze:
            freeze_feature_extractor(classifier)
        fit(
            classifier,
            pooled_train,
            cfg,
            epochs=cfg.da.finetune_epochs,
            # A linear probe over frozen features needs a far larger LR than a
            # full fine-tune; see config/da/default.yaml.
            lr=cfg.da.finetune_head_lr if freeze else cfg.da.finetune_lr,
            trainable=head_parameters(classifier) if freeze else None,
            freeze_features=freeze,
            coral_source=(
                ConcatDataset(list(source_datasets.values()))
                if variant == "coral"
                else None
            ),
            coral_lambda=cfg.da.coral_lambda if variant == "coral" else 0.0,
            tag=f"[{variant}]",
        )

    else:
        raise ValueError(f"Unknown domain-adaptation variant: {variant}")

    return predict(classifier, test_dataset)


def source_internal_loo(cfg, source_datasets):
    """Leave-one-patient-out WITHIN the source cohort.

    This is a diagnostic, not a rung of the ladder, and it is the precondition
    for reading the ladder at all: it measures how well the model predicts in
    the cohort where prediction is known to work, but restricted to the SHARED
    feature space (5 channels, apnea-only) that transfer is limited to.

    The published 0.78/0.80 was obtained with 6 channels including real nasal
    pressure and with hypopnea included, so it does not answer this. If this
    number is itself near chance, the shared channels carry little signal even
    in the good cohort, and a near-chance `zero_shot` would say nothing about
    domain shift. Together with the target-side numbers it separates two costs
    that would otherwise be confounded:

        paper           6ch, apnea+hypopnea, Robin
        source_internal 5ch, apnea-only,     Robin  <- cost of shared space
        zero_shot       5ch, apnea-only,     CPAP   <- cost of domain shift
        target_only     5ch, apnea-only,     CPAP

    NOTE: these scores are computed on SOURCE patients. They belong in a
    separate table from the ladder, which is scored on target patients.
    """
    results = {}
    ids = list(source_datasets.keys())
    log.info(f"=== source-internal leave-one-out over {len(ids)} patients ===")

    for pat_id in ids:
        train_dataset = ConcatDataset(
            [source_datasets[jd] for jd in ids if jd != pat_id]
        )
        classifier = init_classifier(cfg).to(device)
        fit(
            classifier,
            train_dataset,
            cfg,
            epochs=cfg.optimizer.epochs,
            lr=cfg.optimizer.learning_rate,
            tag=f"[source_internal {pat_id}]",
        )
        labels, scores = predict(classifier, source_datasets[pat_id])
        results[pat_id] = {"ys": labels, "scores": scores}
        log.info(
            f"  source patient {pat_id} [source_internal]: "
            f"test_auc={_safe_auc(labels, scores):.3f}"
        )

    return results


def leave_one_out_domain_adaptation(cfg):
    source_datasets = build_cohort(cfg.source, cfg)

    source_internal = None
    if cfg.da.run_source_internal:
        source_internal = source_internal_loo(cfg, source_datasets)

    variants = list(cfg.da.variants)
    if not variants:
        # Diagnostic-only run: never touch the target cohort, so this stays
        # cheap enough to run before committing to the full experiment.
        log.info("No ladder variants requested; skipping the target cohort.")
        return {}, source_internal

    target_datasets = build_cohort(cfg.target, cfg)
    log.info(
        f"Cohorts ready: {len(source_datasets)} source / "
        f"{len(target_datasets)} target patients"
    )

    needs_pretrain = any(
        v
        in (
            "zero_shot",
            "finetune_full",
            "finetune_head",
            "adabn",
            "coral",
            "adabn_patient",
            "zero_shot_late",
        )
        for v in variants
    )
    pretrained_state = None
    if needs_pretrain:
        pretrained = pretrain_on_source(cfg, source_datasets)
        # Keep a CPU copy so each fold starts from identical weights.
        pretrained_state = copy.deepcopy(pretrained.state_dict())

    results = {variant: {} for variant in variants}
    target_ids = list(target_datasets.keys())

    for pat_id in target_ids:
        log.info(f"=== target fold: held-out patient {pat_id} ===")
        start = time.time()
        train_datasets = [target_datasets[jd] for jd in target_ids if jd != pat_id]
        test_dataset = target_datasets[pat_id]

        for variant in variants:
            labels, scores = run_variant(
                variant,
                cfg,
                pretrained_state,
                source_datasets,
                train_datasets,
                test_dataset,
            )
            results[variant][pat_id] = {"ys": labels, "scores": scores}
            log.info(
                f"  patient {pat_id} [{variant}]: "
                f"test_auc={_safe_auc(labels, scores):.3f}"
            )

        log.info(f"  fold took {time.time() - start:.0f}s")

    return results, source_internal


def summarise(results):
    """Per-patient mean AUC (the paper's metric) plus the pooled AUC."""
    summary = {}
    for variant, per_patient in results.items():
        aucs = [_safe_auc(d["ys"], d["scores"]) for d in per_patient.values()]
        all_ys = np.concatenate([d["ys"] for d in per_patient.values()])
        all_scores = np.concatenate([d["scores"] for d in per_patient.values()])
        summary[variant] = {
            "mean_auc": float(np.nanmean(aucs)),
            "std_auc": float(np.nanstd(aucs)),
            "pooled_auc": _safe_auc(all_ys, all_scores),
            "pooled_ap": average_precision_score(all_ys, all_scores),
            "per_patient_auc": aucs,
        }
    return summary


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="domain_adaptation_config",
)
def main(cfg):
    log.info("Cuda is available: " + str(torch.cuda.is_available()))
    log.info(OmegaConf.to_yaml(cfg))

    assert cfg.meta.experiment is not None, "set meta.experiment=<name>"
    assert cfg.meta.tag is not None, "set meta.tag=<name>"
    # np.random.randint(2**32) overflows on Windows, so a seed is mandatory
    # here rather than optional (see Section 3.2 of the thesis).
    assert cfg.meta.seed is not None, "set meta.seed=<int> for reproducibility"
    set_seed(cfg.meta.seed)

    results, source_internal = leave_one_out_domain_adaptation(cfg)
    summary = summarise(results)
    source_summary = (
        summarise({"source_internal": source_internal}) if source_internal else {}
    )

    def _table(header, stats_by_name):
        log.info(header)
        log.info(
            f"{'variant':<16} {'mean AUC':>10} {'SD':>7} {'pooled':>8} {'AP':>7}"
        )
        for name, stats in stats_by_name.items():
            log.info(
                f"{name:<16} {stats['mean_auc']:>10.3f} {stats['std_auc']:>7.3f} "
                f"{stats['pooled_auc']:>8.3f} {stats['pooled_ap']:>7.3f}"
            )

    # Two separate tables on purpose: the diagnostic is scored on SOURCE
    # patients and the ladder on TARGET patients, so the rows are not
    # comparable and must not be printed as one table.
    if source_summary:
        _table(
            "=== DIAGNOSTIC (source cohort, leave-one-patient-out) ===",
            source_summary,
        )
    if summary:
        _table(
            "=== SUMMARY (target cohort, leave-one-patient-out) ===", summary
        )

    os.makedirs(cfg.meta.result_path, exist_ok=True)
    out_path = os.path.join(cfg.meta.result_path, f"{cfg.meta.tag}_da_results.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(
            {
                "results": results,
                "summary": summary,
                "source_internal": source_internal,
                "source_internal_summary": source_summary,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            fh,
        )
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
