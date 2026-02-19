"""
torchmetrics
"""
from typing import Any

import torch
import torchaudio
from torchmetrics import Metric


def _pad_to_min_length(x: torch.Tensor, min_len: int) -> torch.Tensor:
    """Pad last dimension with zeros so STFT/MFCC kernels can run on short clips."""
    if x.shape[-1] >= min_len:
        return x
    return torch.nn.functional.pad(x, (0, int(min_len - x.shape[-1])))


class LogSpectralDistance(Metric):
    """
    Log Spectral Distance (LSD) metric.

    Implementation based on https://arxiv.org/abs/1909.06628
    """

    full_state_update = False

    def __init__(
        self, n_fft=8092, hop_size=64, eps: float = 1e-8, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("lsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.eps = eps

    def _log_spectral_power_mag(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
            # ### HIGHLIGHT: Use zero padding to avoid reflection artifacts on short clips.
            pad_mode="constant",
        )
        return torch.log(torch.square(torch.abs(X)) + self.eps)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        # ### HIGHLIGHT: Guard very short clips for large-STFT LSD settings.
        min_len = max(int(self.n_fft), int(self.n_fft // 2 + 1))
        x = _pad_to_min_length(x, min_len)
        y = _pad_to_min_length(y, min_len)
        x = x.squeeze(1)
        y = y.squeeze(1)

        X = self._log_spectral_power_mag(x)
        Y = self._log_spectral_power_mag(y)

        # Mean of the squared difference along the frequency axis
        lsd = torch.mean(torch.square(X - Y), dim=-2)

        # Mean of the square root over the temporal axis
        lsd = torch.mean(torch.sqrt(lsd), dim=-1)

        self.lsd += torch.sum(lsd)
        self.count += lsd.shape[0]

    def compute(self) -> torch.Tensor:
        return self.lsd / self.count


class MFCCError(Metric):
    """
    MFCC Error
    """

    full_state_update = False

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state("mfcc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        # ### HIGHLIGHT: Ensure MFCC front-end has enough samples for n_fft.
        min_len = max(int(self.n_fft), int(self.n_fft // 2 + 1))
        x = _pad_to_min_length(x, min_len)
        y = _pad_to_min_length(y, min_len)
        x = x.squeeze(1).cpu()
        y = y.squeeze(1).cpu()

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length},
        )
        X = mfcc(x)
        Y = mfcc(y)
        mae = torch.mean(torch.abs(X - Y), dim=-1)

        self.mfcc += torch.sum(mae)
        self.count += mae.shape[0]

    def compute(self) -> torch.Tensor:
        return self.mfcc / self.count


class SpectralFluxOnsetError(Metric):
    """
    Error between spectral flux onset signals
    """

    full_state_update = False

    def __init__(self, n_fft=1024, hop_size=64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.n_fft = n_fft
        self.hop_size = hop_size

    def _onset_signal(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input must be of shape (batch, channels, length)"
        assert x.shape[1] == 1, "Input must be mono"

        x = x.squeeze(1)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=torch.hann_window(self.n_fft, device=x.device),
            return_complex=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
        )
        # L2-norm on the rectified difference of the magnitude spectrogram
        flux = torch.diff(torch.abs(X), dim=1)
        flux = (flux + torch.abs(flux)) / 2
        flux = torch.square(flux)
        flux = torch.sum(flux, dim=1)

        return flux

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        assert x.shape == y.shape
        assert x.ndim == 3 and x.shape[1] == 1, "Only mono audio is supported"
        # ### HIGHLIGHT: Guard short clips for onset-STFT as well.
        min_len = max(int(self.n_fft), int(self.n_fft // 2 + 1))
        x = _pad_to_min_length(x, min_len)
        y = _pad_to_min_length(y, min_len)

        x = self._onset_signal(x)
        y = self._onset_signal(y)

        # MAE
        onset_error = torch.mean(torch.abs(x - y), dim=-1)

        self.error += torch.sum(onset_error)
        self.count += onset_error.shape[0]

    def compute(self) -> torch.Tensor:
        return self.error / self.count
