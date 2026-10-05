"""Loss-side imbalance alternatives: focal at gamma=0 IS cross-entropy; effective-number weights behave."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pedpredict.config import load_config
from pedpredict.config.loader import ConfigError
from pedpredict.config.schema import TrainCfg
from pedpredict.data.sampler import class_weights_ce, class_weights_effective_number, loss_class_weights
from pedpredict.losses.multitask import FocalCrossEntropy, build_multitask_loss


def _batch(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=gen), torch.randint(0, 2, (n,), generator=gen)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_gamma_zero_is_cross_entropy(reduction: str) -> None:
    logits, target = _batch()
    weight = torch.tensor([0.3, 2.5])
    ref = nn.CrossEntropyLoss(weight=weight, reduction=reduction)(logits, target)
    torch.testing.assert_close(FocalCrossEntropy(weight, 0.0, reduction)(logits, target), ref)


def test_focal_downweights_easy_examples() -> None:
    target = torch.tensor([1, 1])
    logits = torch.tensor([[-4.0, 4.0], [0.0, 0.0]])          # one confident-correct, one undecided
    per = FocalCrossEntropy(torch.ones(2), 2.0, "none")(logits, target)
    ce = nn.CrossEntropyLoss(reduction="none")(logits, target)
    assert per[0] / ce[0] < 1e-3 and per[1] / ce[1] == pytest.approx(0.25)
    assert (per <= ce).all()


def test_focal_gradient_flows() -> None:
    logits, target = _batch()
    logits.requires_grad_(True)
    FocalCrossEntropy(torch.ones(2), 2.0)(logits, target).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_build_loss_uses_focal_only_when_asked() -> None:
    weights = {t: torch.ones(2) for t in ("actions", "looks", "crosses")}
    plain = build_multitask_loss(TrainCfg(), weights)
    focal = build_multitask_loss(TrainCfg(focal_gamma=2.0), weights)
    assert all(type(c) is nn.CrossEntropyLoss for c in plain.criteria.values())
    assert all(isinstance(c, FocalCrossEntropy) and c.gamma == 2.0 for c in focal.criteria.values())


def test_effective_number_weights() -> None:
    counts = {"actions": {0: 50, 1: 50}, "looks": {}, "crosses": {0: 3400, 1: 1400}}
    w = class_weights_effective_number(counts, 0.9999)
    assert w["actions"].tolist() == pytest.approx([1.0, 1.0])
    assert w["looks"].tolist() == [1.0, 1.0]
    assert float(w["crosses"].sum()) == pytest.approx(2.0)
    assert w["crosses"][1] > w["crosses"][0]
    inverse = class_weights_ce(counts)["crosses"]
    ratio_cb = float(w["crosses"][1] / w["crosses"][0])
    assert 1.0 < ratio_cb < float(inverse[1] / inverse[0])     # flatter than inverse frequency
    near_one = class_weights_effective_number(counts, 1 - 1e-9)["crosses"]
    assert float(near_one[1] / near_one[0]) == pytest.approx(float(inverse[1] / inverse[0]), rel=1e-3)
    assert loss_class_weights(TrainCfg(), counts)["crosses"].tolist() == inverse.tolist()
    cb_cfg = TrainCfg(class_weight_mode="effective_number")
    assert loss_class_weights(cb_cfg, counts)["crosses"].tolist() == w["crosses"].tolist()


@pytest.mark.parametrize("override", ["train.class_weight_mode=focal", "train.cb_beta=1.0", "train.focal_gamma=-1"])
def test_validation_rejects_bad_values(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config("configs", [override])


def test_defaults_are_the_legacy_loss() -> None:
    cfg = load_config("configs", [])
    assert (cfg.train.class_weight_mode, cfg.train.focal_gamma, cfg.train.use_class_weights) == ("inverse", 0.0, False)
