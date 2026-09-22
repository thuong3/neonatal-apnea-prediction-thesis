"""GUARD FOR THE F3 SAMPLER OPTION.

Two things must hold, and both are easy to break silently:

1. THE DEFAULT IS UNCHANGED. `equal_per_dataset=False` must draw exactly what
   the published sampler drew, or every earlier result becomes incomparable.

2. THE OPTION ACTUALLY EQUALISES. With `equal_per_dataset=True` every infant
   must contribute the same number of windows per epoch, and each infant's
   contribution must still be class-balanced internally -- equalising the
   infants is worthless if it un-balances the classes inside them.

Synthetic datasets are used so this runs anywhere, with deliberately lopsided
class counts modelled on the real cohort (patient 015 has 306 positives,
patient 012 has 22).

Run:  python scripts/check_sampler_equal.py
"""
import os
import sys
from collections import Counter

import numpy as np
from torch.utils.data import ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.neonatal_utils import MultiDatasetBalancedSampler  # noqa: E402


class FakeDataset:
    """Minimal stand-in: the sampler only needs _get_labels() and __len__()."""

    def __init__(self, n_pos, n_neg):
        self.labels = np.array([1] * n_pos + [0] * n_neg)

    def _get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.labels[i], self.labels[i]


# (n_pos, n_neg) per infant, taken from the measured cohort counts.
COHORT = [(33, 1759), (206, 586), (112, 848), (61, 1441), (64, 1620),
          (250, 527), (51, 1617), (71, 1467), (23, 972), (84, 1435),
          (121, 1141), (22, 1689), (72, 1543), (197, 815), (306, 389)]


def contributions(sampler, datasets):
    """How many drawn indices belong to each dataset, and their class split."""
    bounds, offset = [], 0
    for d in datasets:
        bounds.append((offset, offset + len(d)))
        offset += len(d)
    per_ds = Counter()
    per_ds_pos = Counter()
    for idx in sampler:
        for i, (lo, hi) in enumerate(bounds):
            if lo <= idx < hi:
                per_ds[i] += 1
                if datasets[i].labels[idx - lo] == 1:
                    per_ds_pos[i] += 1
                break
    return per_ds, per_ds_pos


def main():
    np.random.seed(0)
    datasets = [FakeDataset(p, n) for p, n in COHORT]
    concat = ConcatDataset(datasets)

    print("[1] default path is the published behaviour")
    s = MultiDatasetBalancedSampler(concat, replacement=False)
    per_ds, _ = contributions(s, datasets)
    expected = {i: 2 * min(p, n) for i, (p, n) in enumerate(COHORT)}
    for i in range(len(COHORT)):
        assert per_ds[i] == expected[i], (
            f"infant {i}: drew {per_ds[i]}, published rule gives {expected[i]}"
        )
    total = sum(per_ds.values())
    share = sorted((c / total for c in per_ds.values()), reverse=True)
    print(f"    OK -- 2*min(n_pos, n_neg) per infant, total {total}")
    print(f"    largest infant {share[0]*100:.1f}% of the epoch, "
          f"smallest {share[-1]*100:.1f}%  ({share[0]/share[-1]:.0f}x spread)")

    print("\n[2] equal_per_dataset gives every infant the same count")
    s = MultiDatasetBalancedSampler(concat, equal_per_dataset=True)
    per_ds, per_pos = contributions(s, datasets)
    counts = [per_ds[i] for i in range(len(COHORT))]
    assert len(set(counts)) == 1, f"infants contributed unequally: {counts}"
    total = sum(counts)
    print(f"    OK -- every infant contributes {counts[0]} windows, total {total}")
    print(f"    each infant is {100/len(COHORT):.1f}% of the epoch (was "
          f"{share[0]*100:.1f}% for the largest)")

    print("\n[3] classes stay balanced inside each infant")
    for i in range(len(COHORT)):
        pos, tot = per_pos[i], per_ds[i]
        assert abs(pos / tot - 0.5) < 1e-9, (
            f"infant {i}: {pos}/{tot} positive -- equalising un-balanced the classes"
        )
    print("    OK -- 50/50 positive/negative within every infant")

    print("\n[4] the target defaults to the cohort median of min(n_pos, n_neg)")
    med = int(np.median([min(p, n) for p, n in COHORT]))
    assert counts[0] == 2 * med, f"expected {2*med} per infant, got {counts[0]}"
    print(f"    OK -- median min(n_pos, n_neg) = {med}, so {2*med} windows each")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
