"""
Helpers for loss functions.
"""
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F


class FirstOrderDifferenceLoss(torch.nn.Module):
    """
    A loss function that calculates the first-order difference
    of the input and target tensors and then calculates the L1
    loss between the two. This essentially applies a high-pass
    filter to the signal before calculating the loss, which may
    potentially be useful for emphasizing transient components.

    Args:
        reduction (str): The reduction method to use, passed
            into the L1 loss. Defaults to "mean".
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_diff = torch.diff(pred)
        target_diff = torch.diff(target)
        return self.loss(pred_diff, target_diff)


class WeightedLoss(torch.nn.Module):
    """
    A loss function that combines and sums weightings of multiple loss functions.

    Args:
        losses: A list of loss functions.
        weights: A list of weights for each loss function. Defaults to None, which
            results in equal weighting of all loss functions.
    """

    def __init__(
        self,
        loss_fns: List[Union[Callable, torch.nn.Module]],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.loss_fns = loss_fns
        if weights is None:
            weights = [1.0] * len(loss_fns)
        else:
            assert len(loss_fns) == len(
                weights
            ), "Number of losses and weights must match."
        self.weights = weights

    def forward(self, *args, **kwargs):
        losses = [
            weight * loss_fn(*args, **kwargs)
            for loss_fn, weight in zip(self.loss_fns, self.weights)
        ]

        return sum(losses)


def _build_default_mrstft_loss() -> torch.nn.Module:
    import auraloss

    return auraloss.freq.MultiResolutionSTFTLoss()


class _LengthAwareMRSTFTAuxLoss(torch.nn.Module, ABC):
    """
    Base class for MR-STFT losses with a lightweight auxiliary term.

    The MR-STFT configuration stays identical to the plain baseline by default,
    while auxiliary terms are computed only on valid unpadded samples when
    `lengths` are provided.
    """

    def __init__(self, mrstft: Optional[torch.nn.Module] = None):
        super().__init__()
        self.mrstft = mrstft if mrstft is not None else _build_default_mrstft_loss()
        self.last_stats: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_lengths(
        lengths,
        *,
        batch_size: int,
        max_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if lengths is None:
            return None

        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=device)
        else:
            lengths = lengths.to(device=device)

        lengths = lengths.reshape(-1)
        if lengths.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} lengths entries, got {lengths.numel()}."
            )

        return torch.clamp(lengths.long(), min=1, max=max_length)

    @staticmethod
    def _is_uniform_length(lengths: Optional[torch.Tensor]) -> bool:
        if lengths is None:
            return True
        return bool(torch.all(lengths == lengths[0]).item())

    @staticmethod
    def _iter_valid_pairs(
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ):
        if lengths is None:
            for idx in range(pred.shape[0]):
                yield pred[idx : idx + 1], target[idx : idx + 1]
            return

        for idx in range(pred.shape[0]):
            length = int(lengths[idx].item())
            yield pred[idx : idx + 1, :, :length], target[idx : idx + 1, :, :length]

    @staticmethod
    def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x.square().mean(), min=eps))

    @property
    @abstractmethod
    def aux_weight(self) -> float:
        """
        Scalar auxiliary weight used in the total loss.
        """

    @property
    @abstractmethod
    def aux_stat_key(self) -> str:
        """
        Stats key used for logging the auxiliary term.
        """

    @abstractmethod
    def _compute_aux_loss_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean auxiliary loss for a uniform-length batch.
        """

    @abstractmethod
    def _compute_aux_loss_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the auxiliary loss for a single valid sample.
        """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, lengths=None):
        if pred.shape != target.shape:
            raise ValueError(
                f"`pred` and `target` must have the same shape, got "
                f"{pred.shape} and {target.shape}."
            )

        lengths = self._normalize_lengths(
            lengths,
            batch_size=pred.shape[0],
            max_length=pred.shape[-1],
            device=pred.device,
        )

        if self._is_uniform_length(lengths):
            valid_length = pred.shape[-1] if lengths is None else int(lengths[0].item())
            pred_valid = pred[..., :valid_length]
            target_valid = target[..., :valid_length]
            mrstft_loss = self.mrstft(pred_valid, target_valid)
            aux_loss = self._compute_aux_loss_batch(pred_valid, target_valid)
        else:
            mrstft_losses = []
            aux_losses = []
            for pred_i, target_i in self._iter_valid_pairs(pred, target, lengths):
                mrstft_losses.append(self.mrstft(pred_i, target_i))
                aux_losses.append(self._compute_aux_loss_single(pred_i, target_i))

            mrstft_loss = torch.stack(mrstft_losses).mean()
            aux_loss = torch.stack(aux_losses).mean()

        total_loss = mrstft_loss + (self.aux_weight * aux_loss)
        self.last_stats = {
            "loss_total": total_loss.detach(),
            "loss_mrstft": mrstft_loss.detach(),
            self.aux_stat_key: aux_loss.detach(),
        }
        return total_loss


class MRSTFTWithLogRMSAuxLoss(_LengthAwareMRSTFTAuxLoss):
    """
    Plain MR-STFT with a tiny scalar log-RMS auxiliary term.
    """

    def __init__(
        self,
        amp_weight: float = 0.01,
        eps: float = 1.0e-8,
        mrstft: Optional[torch.nn.Module] = None,
    ):
        super().__init__(mrstft=mrstft)
        self.amp_weight = amp_weight
        self.eps = eps

    @property
    def aux_weight(self) -> float:
        return self.amp_weight

    @property
    def aux_stat_key(self) -> str:
        return "loss_aux_amp"

    def _compute_aux_loss_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_rms = torch.sqrt(
            torch.clamp(pred.square().mean(dim=(-2, -1)), min=self.eps)
        )
        target_rms = torch.sqrt(
            torch.clamp(target.square().mean(dim=(-2, -1)), min=self.eps)
        )
        return torch.abs(torch.log(pred_rms) - torch.log(target_rms)).mean()

    def _compute_aux_loss_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_rms = self._rms(pred, self.eps)
        target_rms = self._rms(target, self.eps)
        return torch.abs(torch.log(pred_rms) - torch.log(target_rms))


class MRSTFTWithSmoothL1AuxLoss(_LengthAwareMRSTFTAuxLoss):
    """
    Plain MR-STFT with a tiny waveform SmoothL1 auxiliary term.
    """

    def __init__(
        self,
        smooth_l1_weight: float = 0.02,
        smooth_l1_beta: float = 0.1,
        mrstft: Optional[torch.nn.Module] = None,
    ):
        super().__init__(mrstft=mrstft)
        self.smooth_l1_weight = smooth_l1_weight
        self.smooth_l1_beta = smooth_l1_beta

    @property
    def aux_weight(self) -> float:
        return self.smooth_l1_weight

    @property
    def aux_stat_key(self) -> str:
        return "loss_aux_smoothl1"

    def _compute_aux_loss_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=self.smooth_l1_beta,
        ).mean()

    def _compute_aux_loss_single(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.smooth_l1_loss(
            pred,
            target,
            reduction="mean",
            beta=self.smooth_l1_beta,
        )
