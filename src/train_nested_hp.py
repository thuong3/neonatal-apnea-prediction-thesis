"""F2: nested hyperparameter selection over the CPAP cohort.

WHAT NESTED MEANS HERE, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------
For each held-out infant, the hyperparameters are chosen using ONLY the other
14, by holding `hp.n_val` of them out as an inner validation set. The held-out
infant is never seen by the selection. Tuning against the outer fold instead is
the failure this study has already met once: Section 4.4 of the thesis
reports random cross-validation giving 0.67 where the correct temporal split
gives 0.55. With 15 infants and a statistic that moves 0.022 between identical
runs (section 33.6), selecting on the test fold would produce a better number that
means nothing, and it would be very hard to see afterwards.

WHAT THE OUTPUT IS AN ESTIMATE OF
---------------------------------
The outer per-patient AuROCs estimate the performance of THE TUNED PIPELINE --
"fit this model, choosing hyperparameters by inner validation" -- not the
performance of whichever configuration won most often. Reporting the best cell's
own score would be selection bias, the same trap `make_patient_adaptation_figure`
documents for its argmax cell. Quote the outer mean; report the winner table as a
description of how stable selection was, never as a result.

WHAT TO WATCH, NOT JUST THE HEADLINE
------------------------------------
The reason to run this is the TRAIN AuROC, not the test AuROC. The thesis claims
the model cannot fit the CPAP training windows (train 0.655 against Robin's
0.859). Two outcomes, both useful:

  * train stays low under every configuration  -> the claim is measured rather
    than assumed, and section 22 gets stronger.
  * train rises well above 0.655 while test stays near 0.55 -> the null holds but
    its EXPLANATION changes from "underfits, so the information is absent" to
    "fits, and none of it generalises". Section 4.3, the figure
    `underfitting_and_central`, and section 5.1 would need rewriting. Budget for
    this before starting.

Both numbers are written per fold and per configuration, so the answer is in the
JSON either way.

Run (on the GPU VM -- see the header note in scripts/run_nested_hp.sh):

    PYTHONPATH=. python src/train_nested_hp.py dataset=neonatal \\
        meta.experiment=cpap_hp meta.tag=nested_seed0 meta.seed=0 \\
        meta.data_path=/path/to/dataset_brainimmaturity \\
        meta.result_path=/path/to/results

Cheap first pass:  '+hp.configs=[base,wide_long,drop25]'
"""
import copy
import json
import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import roc_auc_score

from src.train_neonatal import (
    get_full_performance_and_logits,
    init_classifier,
    init_neonatal_dataset,
    set_seed,
    train_model_and_evaluate_performance,
)
from torch.utils.data import ConcatDataset

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HP_GRID = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/hp_grid.yaml"
)


def load_grid(cfg):
    # `hp.grid_file` exists so the whole loop can be exercised on a one-point,
    # one-epoch grid before it is trusted with a ten-hour run.
    grid = OmegaConf.load(cfg.get("hp", {}).get("grid_file", None) or HP_GRID)
    points = list(grid.configs)
    wanted = cfg.get("hp", {}).get("configs", None)
    if wanted:
        names = set(wanted)
        points = [p for p in points if p.name in names]
        missing = names - {p.name for p in points}
        if missing:
            raise SystemExit(f"unknown config name(s) in hp.configs: {sorted(missing)}")
    n_val = int(cfg.get("hp", {}).get("n_val", grid.n_val))
    return points, n_val


def apply_point(cfg, point):
    """A copy of cfg with one grid point applied. Never mutates the original."""
    out = copy.deepcopy(cfg)
    # Hydra configs are struct-mode, so a key absent from the network yaml
    # cannot simply be assigned. `open_dict` is needed for any network config
    # that predates F2 -- which is all of them except nam/nam_b4.
    with open_dict(out):
        out.optimizer.epochs = int(point.epochs)
        out.optimizer.learning_rate = float(point.lr)
        out.network.hidden_channels = [int(point.hidden)] * len(cfg.network.in_channels)
        # Read via `.get` in init_classifier, so this is the only place it
        # enters. 0.0 inserts no module at all -- check_dropout_identity.py.
        out.network.dropout_p = float(point.dropout)
    return out


def inner_validation_ids(ids, outer_pat, n_val):
    """The inner validation infants for one outer fold.

    A deterministic rotation, not a random draw: it depends only on the position
    of the outer infant in the id list, so the split is fixed before any number
    exists and every infant takes a turn in inner validation. Re-running the
    script cannot quietly reshuffle it into a luckier arrangement.
    """
    rest = [i for i in ids if i != outer_pat]
    start = ids.index(outer_pat) % len(rest)
    return [rest[(start + k) % len(rest)] for k in range(n_val)]


def score_on(cfg, state_dict, dataset):
    """Per-patient AuROC for a trained model on one infant's dataset."""
    classifier = init_classifier(cfg)
    classifier.load_state_dict(state_dict)
    labels, scores, _logits, _bias = get_full_performance_and_logits(
        dataset, classifier
    )
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


@hydra.main(version_base=None, config_path="../config", config_name="neonatal_config")
def main(cfg):
    assert cfg.meta.experiment is not None and cfg.meta.tag is not None
    log.info("Cuda is available: " + str(torch.cuda.is_available()))
    if not torch.cuda.is_available():
        log.warning(
            "Running on CPU. The grid is sized for a GPU; on CPU this is days, "
            "not hours, and the result will not be numerically comparable with "
            "the existing VM runs."
        )

    points, n_val = load_grid(cfg)
    ids = list(cfg.dataset.ids)
    log.info(f"{len(points)} configurations x {len(ids)} outer folds, "
             f"inner validation on {n_val} infants")

    set_seed(cfg.meta.seed if cfg.meta.seed is not None else 0)

    log.info("Building datasets once for all folds.")
    datasets = {p: init_neonatal_dataset(p, cfg) for p in ids}

    folds, results = [], {}
    t_start = time.time()

    for outer_pat in ids:
        val_ids = inner_validation_ids(ids, outer_pat, n_val)
        fit_ids = [i for i in ids if i != outer_pat and i not in val_ids]
        log.info(
            f"=== outer {outer_pat}: fit on {len(fit_ids)}, "
            f"inner-validate on {val_ids}"
        )

        trials = []
        for point in points:
            cfg_i = apply_point(cfg, point)
            set_seed(cfg.meta.seed if cfg.meta.seed is not None else 0)
            state, _labs, _scores, train_auc = train_model_and_evaluate_performance(
                cfg_i,
                ConcatDataset([datasets[i] for i in fit_ids]),
                ConcatDataset([datasets[i] for i in val_ids]),
                log_to_wandb=False,
            )
            # Mean over infants, not pooled: the pre-specification's primary
            # statistic (PRE_SPECIFICATION.md 1.1), and pooled AuROC is unusable
            # here anyway (33.6).
            per_val = [score_on(cfg_i, state, datasets[i]) for i in val_ids]
            val_auc = float(np.nanmean(per_val))
            trials.append({"name": point.name, "val_auc": val_auc,
                           "train_auc": float(train_auc),
                           "per_val": dict(zip(val_ids, per_val))})
            log.info(f"    {point.name:15s} inner-val {val_auc:.3f} "
                     f"(train {train_auc:.3f})")

        best = max(trials, key=lambda t: t["val_auc"])
        log.info(f"    -> selected {best['name']} at {best['val_auc']:.3f}")

        # Refit the winner on all 14 and score the untouched outer infant.
        winner = next(p for p in points if p.name == best["name"])
        cfg_w = apply_point(cfg, winner)
        set_seed(cfg.meta.seed if cfg.meta.seed is not None else 0)
        state, labs, scores, train_auc = train_model_and_evaluate_performance(
            cfg_w,
            ConcatDataset([datasets[i] for i in ids if i != outer_pat]),
            datasets[outer_pat],
            log_to_wandb=False,
        )
        try:
            outer_auc = float(roc_auc_score(labs, scores))
        except ValueError:
            outer_auc = float("nan")
        log.info(f"    outer {outer_pat}: test {outer_auc:.3f} "
                 f"(train {train_auc:.3f}) with {best['name']}")

        results[outer_pat] = {"ys": np.asarray(labs).tolist(),
                              "scores": np.asarray(scores).tolist()}
        folds.append({"outer": outer_pat, "val_ids": val_ids,
                      "selected": best["name"], "outer_auc": outer_auc,
                      "outer_train_auc": float(train_auc), "trials": trials})

    aucs = [f["outer_auc"] for f in folds]
    trains = [f["outer_train_auc"] for f in folds]
    counts = {}
    for f in folds:
        counts[f["selected"]] = counts.get(f["selected"], 0) + 1

    log.info("=== SUMMARY ===")
    log.info(f"outer mean test AuROC  {np.nanmean(aucs):.4f}  "
             f"(this is the number to quote)")
    log.info(f"outer mean train AuROC {np.nanmean(trains):.4f}  "
             f"(compare with 0.655 published-config baseline, 0.859 on Robin)")
    log.info(f"selection counts: {counts}")
    log.info(f"elapsed {(time.time() - t_start) / 3600:.2f} h")

    out_dir = cfg.meta.result_path or "."
    os.makedirs(os.path.join(out_dir, cfg.meta.experiment), exist_ok=True)
    path = os.path.join(out_dir, cfg.meta.experiment, f"{cfg.meta.tag}_nested.json")
    with open(path, "w") as fh:
        json.dump({"folds": folds, "results": results,
                   "mean_test_auc": float(np.nanmean(aucs)),
                   "mean_train_auc": float(np.nanmean(trains)),
                   "selection_counts": counts,
                   "n_val": n_val,
                   "grid": [dict(p) for p in points]}, fh, indent=2)
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()
