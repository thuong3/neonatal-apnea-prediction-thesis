"""GUARD: adding `dropout_p` did not move the published architecture.

F2 asks for hyperparameter selection, and the honest way to run it is against a
baseline that is still the baseline. This script asserts that, and it is cheap
enough to run before every sweep.

  [1] `dropout_p = 0.0` inserts NOTHING. No `nn.Dropout` anywhere in the model,
      so the module list, the state-dict keys and their order are what they
      were before F2 touched the file.
  [2] Omitting the argument entirely and passing 0.0 give byte-identical
      state-dict keys and byte-identical forward output from the same seed.
      This is the case that matters: every config in `config/network/` except
      the sweep's own has no `dropout_p` key at all.
  [3] `dropout_p > 0` DOES insert dropout, in train mode only, and eval mode is
      unaffected. A regulariser that silently did nothing would produce a null
      that means nothing -- the failure mode this study has already been bitten
      by once (SS27.8a, the causal arms leaking through `filtfilt`).
  [4] The published config still parses and still reads as 6 channels x 20
      hidden, kernels 17/17/11/17/11/11.

Run:
    python scripts/check_dropout_identity.py
"""
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules import GlobalAveragePooling, NeuralAdditiveModel  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAM_CFG = os.path.join(REPO, "config/network/nam.yaml")


def has_dropout(module):
    return any(isinstance(m, torch.nn.Dropout) for m in module.modules())


def build(dropout_p=None, seed=0):
    torch.manual_seed(seed)
    kwargs = {} if dropout_p is None else {"dropout_p": dropout_p}
    return GlobalAveragePooling(in_channel=1, hidden_channel=20,
                                kernel_size=17, **kwargs)


def main():
    failures = []

    # [1] p = 0 inserts nothing
    m0 = build(0.0)
    if has_dropout(m0):
        failures.append("[1] dropout_p=0.0 inserted an nn.Dropout module")
    else:
        print("[1] ok   dropout_p=0.0 inserts no Dropout module")

    # [2] omitted == 0.0, in keys and in output
    m_default = build(None)
    if list(m_default.state_dict().keys()) != list(m0.state_dict().keys()):
        failures.append("[2] state-dict keys differ between omitted and 0.0")
    else:
        x = torch.randn(4, 1, 150)
        m_default.eval()
        m0.eval()
        with torch.no_grad():
            same = torch.equal(m_default.forward_logit(x), m0.forward_logit(x))
        if not same:
            failures.append("[2] forward output differs between omitted and 0.0")
        else:
            print("[2] ok   omitting dropout_p is identical to passing 0.0")

    # [3] p > 0 is real, and only in train mode
    m5 = build(0.25)
    if not has_dropout(m5):
        failures.append("[3] dropout_p=0.25 did NOT insert a Dropout module")
    else:
        x = torch.randn(8, 1, 150)
        m5.train()
        torch.manual_seed(1)
        a = m5.forward_logit(x)
        torch.manual_seed(2)
        b = m5.forward_logit(x)
        if torch.equal(a, b):
            failures.append("[3] train-mode output is unaffected by dropout")
        m5.eval()
        with torch.no_grad():
            c, d = m5.forward_logit(x), m5.forward_logit(x)
        if not torch.equal(c, d):
            failures.append("[3] eval-mode output is not deterministic")
        if not any(f.startswith("[3]") for f in failures):
            print("[3] ok   dropout_p=0.25 perturbs train mode, not eval mode")

    # [4] the published network config is unchanged
    cfg = OmegaConf.load(NAM_CFG)
    expected = {"self": "nam", "in_channels": [1] * 6,
                "hidden_channels": [20] * 6,
                "kernel_size": [17, 17, 11, 17, 11, 11]}
    for key, want in expected.items():
        got = OmegaConf.to_container(cfg.get(key)) if key != "self" else cfg.get(key)
        if got != want:
            failures.append(f"[4] config/network/nam.yaml {key}: {got} != {want}")
    if float(cfg.get("dropout_p", 0.0)) != 0.0:
        failures.append("[4] config/network/nam.yaml no longer defaults to dropout_p 0")
    if not any(f.startswith("[4]") for f in failures):
        print("[4] ok   config/network/nam.yaml is the published architecture")

    print()
    if failures:
        for f in failures:
            print("FAIL " + f)
        raise SystemExit(1)
    print("all checks passed - the F2 sweep can be compared against existing runs")


if __name__ == "__main__":
    main()
