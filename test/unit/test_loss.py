"""
Tests for the `drumblender.loss` module.
"""
import jsonargparse
import pytest
import torch

import drumblender.loss as loss


def test_first_order_difference_loss():
    loss_fn = loss.FirstOrderDifferenceLoss()
    pred = torch.ones(1, 1, 100)
    target = torch.ones(1, 1, 100)
    assert loss_fn(pred, target) == 0.0


def test_weighted_loss_forwards():
    loss_fn = loss.WeightedLoss(
        [torch.nn.L1Loss(), torch.nn.L1Loss()], weights=[2.0, 1.0]
    )
    pred = torch.ones(1, 1, 100)
    target = torch.zeros(1, 1, 100)
    assert loss_fn(pred, target) == 3.0


def test_weighted_loss_forwards_no_weights():
    loss_fn = loss.WeightedLoss(
        [torch.nn.L1Loss(), torch.nn.L1Loss()],
    )
    pred = torch.ones(1, 1, 100)
    target = torch.zeros(1, 1, 100)

    # Default weights are 1.0
    assert loss_fn(pred, target) == 2.0


def test_weighted_loss_different_weights():
    with pytest.raises(AssertionError):
        loss.WeightedLoss([torch.nn.L1Loss()], weights=[2.0, 1.0])


def test_weighted_loss_with_jsonargparse_config(monkeypatch):
    # Monkeypatch the torch.nn.L1Loss and torch.nn.MSELoss classes to return a constant
    # value so that we can test the weighted loss.
    monkeypatch.setattr(torch.nn.L1Loss, "forward", lambda self, x, y: 1.0)
    monkeypatch.setattr(torch.nn.MSELoss, "forward", lambda self, x, y: 20.0)
    expected_loss = 4.0

    config = (
        "loss:\n"
        "  class_path: drumblender.loss.WeightedLoss\n"
        "  init_args:\n"
        "    loss_fns: \n"
        "    - class_path: torch.nn.L1Loss\n"
        "      init_args:\n"
        "        reduction: mean\n"
        "    - class_path: torch.nn.MSELoss\n"
        "      init_args:\n"
        "        reduction: sum\n"
        "    weights: [2.0, 0.1]"
    )

    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--loss", type=torch.nn.Module)
    args = parser.parse_string(config)
    objs = parser.instantiate_classes(args)

    shape = 13, 4, 9, 2
    a = torch.testing.make_tensor(*shape, dtype=torch.float32, device="cpu")
    b = torch.testing.make_tensor(*shape, dtype=torch.float32, device="cpu")

    actual_loss = objs.loss(a, b)
    assert actual_loss == expected_loss


@pytest.mark.parametrize(
    ("loss_fn", "weight_attr"),
    [
        (
            loss.MRSTFTWithLogRMSAuxLoss(amp_weight=0.0, mrstft=torch.nn.L1Loss()),
            "loss_aux_amp",
        ),
        (
            loss.MRSTFTWithSmoothL1AuxLoss(
                smooth_l1_weight=0.0,
                mrstft=torch.nn.L1Loss(),
            ),
            "loss_aux_smoothl1",
        ),
    ],
)
def test_mrstft_aux_losses_match_plain_baseline_when_weight_zero(loss_fn, weight_attr):
    pred = torch.tensor([[[1.0, -1.0, 0.5, 0.0]]], dtype=torch.float32)
    target = torch.tensor([[[0.0, -0.5, 0.25, 0.25]]], dtype=torch.float32)

    baseline = torch.nn.L1Loss()(pred, target)
    actual = loss_fn(pred, target)

    torch.testing.assert_close(actual, baseline)
    assert set(loss_fn.last_stats) == {"loss_total", "loss_mrstft", weight_attr}


@pytest.mark.parametrize(
    "loss_fn",
    [
        loss.MRSTFTWithLogRMSAuxLoss(amp_weight=0.01, mrstft=torch.nn.L1Loss()),
        loss.MRSTFTWithSmoothL1AuxLoss(
            smooth_l1_weight=0.02,
            smooth_l1_beta=0.1,
            mrstft=torch.nn.L1Loss(),
        ),
    ],
)
def test_mrstft_aux_losses_ignore_padded_tail(loss_fn):
    pred = torch.zeros(2, 1, 6, dtype=torch.float32)
    target = torch.zeros(2, 1, 6, dtype=torch.float32)
    lengths = torch.tensor([3, 6], dtype=torch.long)

    pred[0, 0, 3:] = 100.0
    target[0, 0, 3:] = -100.0

    actual = loss_fn(pred, target, lengths=lengths)
    torch.testing.assert_close(actual, torch.tensor(0.0))


@pytest.mark.parametrize(
    "loss_fn",
    [
        loss.MRSTFTWithLogRMSAuxLoss(amp_weight=0.01, mrstft=torch.nn.L1Loss()),
        loss.MRSTFTWithSmoothL1AuxLoss(
            smooth_l1_weight=0.02,
            smooth_l1_beta=0.1,
            mrstft=torch.nn.L1Loss(),
        ),
    ],
)
def test_mrstft_aux_losses_return_scalar_for_fixed_and_variable_length_batches(loss_fn):
    pred = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)

    fixed_length_loss = loss_fn(pred, target)
    variable_length_loss = loss_fn(pred, target, lengths=torch.tensor([4, 2]))

    assert fixed_length_loss.ndim == 0
    assert variable_length_loss.ndim == 0
