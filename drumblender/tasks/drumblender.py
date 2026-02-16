"""
LightningModule for drum synthesis
"""
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange


class DrumBlender(pl.LightningModule):
    """
    LightningModule for kick synthesis from a modal frequency input

    # TODO: Alot of these are currently optional to help with testing and devlopment,
    # but they should be required in the future

    Args:
        modal_synth (nn.Module): Synthesis model takes modal parameters and generates
            audio
        loss_fn (Union[Callable, nn.Module]): Loss function to use for training
        noise_synth (Optional[nn.Module]): Receives noise parameters and generates
            noise audio signal
        transient_synth (Optional[nn.Module]): Receives audio plus transient parameters
            and generates transient audio signal
        modal_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates modal parameters
        noise_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates noise parameters
        transient_autoencoder (Optional[nn.Module]): Receives main embedding and
            generates transient parameters
        encoder (Optional[nn.Module]): Receives audio and generates main embedding
        float32_matmul_precision(Literal["medium", "high", "highest", None]): Sets
            the precision of float32 matmul operations.
    """

    def __init__(
        self,
        modal_synth: nn.Module,
        loss_fn: Union[Callable, nn.Module],
        noise_synth: Optional[nn.Module] = None,
        transient_synth: Optional[nn.Module] = None,
        modal_autoencoder: Optional[nn.Module] = None,
        noise_autoencoder: Optional[nn.Module] = None,
        transient_autoencoder: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        transient_parallel: bool = False,
        transient_takes_noise: bool = False,
        modal_autoencoder_accepts_audio: bool = False,
        noise_autoencoder_accepts_audio: bool = False,
        transient_autoencoder_accepts_audio: bool = False,
        test_metrics: Optional[torch.nn.ModuleDict] = None,
        float32_matmul_precision: Literal["medium", "high", "highest", None] = None,
    ):
        super().__init__()

        self.modal_synth = modal_synth
        self.loss_fn = loss_fn
        self.noise_synth = noise_synth
        self.transient_synth = transient_synth
        self.modal_autoencoder = modal_autoencoder
        self.noise_autoencoder = noise_autoencoder
        self.transient_autoencoder = transient_autoencoder
        self.encoder = encoder
        self.transient_parallel = transient_parallel
        self.modal_autoencoder_accepts_audio = modal_autoencoder_accepts_audio
        self.noise_autoencoder_accepts_audio = noise_autoencoder_accepts_audio
        self.transient_autoencoder_accepts_audio = transient_autoencoder_accepts_audio
        self.transient_takes_noise = transient_takes_noise

        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)

        if test_metrics is not None:
            self.metrics = test_metrics

    # def forward(
    #     self,
    #     original: torch.Tensor,
    #     params: torch.Tensor,
    # ):
    #     # Main embeddings
    #     embedding = None
    #     if self.encoder is not None:
    #         embedding = self.encoder(original)
    #
    #     # Modal parameter autoencoder
    #     modal_params = params
    #     if self.modal_autoencoder is not None:
    #         if self.modal_autoencoder_accepts_audio:
    #             modal_params = self.modal_autoencoder(original, params)
    #         else:
    #             modal_params, _ = self.modal_autoencoder(embedding, params)
    #
    #     noise_params = None
    #     if self.noise_autoencoder is not None:
    #         if self.noise_autoencoder_accepts_audio:
    #             noise_params = self.noise_autoencoder(original)
    #         else:
    #             noise_params, _ = self.noise_autoencoder(embedding)
    #
    #     transient_params = None
    #     if self.transient_autoencoder is not None:
    #         if self.transient_autoencoder_accepts_audio:
    #             transient_params = self.transient_autoencoder(original)
    #         else:
    #             transient_params, _ = self.transient_autoencoder(embedding)
    #
    #     # Synthesis
    #     y_hat = self.modal_synth(modal_params, original.shape[-1])
    #
    #     if self.noise_synth is not None:
    #         assert noise_params is not None, "Noise params must be provided"
    #         noise = self.noise_synth(noise_params, original.shape[-1])
    #         noise = rearrange(noise, "b n -> b () n")
    #         if self.transient_takes_noise:
    #             y_hat = y_hat + noise
    #
    #     if self.transient_synth is not None:
    #         transient = self.transient_synth(y_hat, transient_params)
    #
    #         # Transient can be added in parallel or in series
    #         if self.transient_parallel:
    #             y_hat = y_hat + transient
    #         else:
    #             y_hat = transient
    #
    #     # Finally, if we have noise and did not add it through
    #     # the TCN, add noise in parallel.
    #     if self.noise_synth is not None:
    #         if self.transient_takes_noise is False:
    #             y_hat = y_hat + noise
    #
    #     return y_hat

    def forward(
        self,
        original: torch.Tensor,
        params: torch.Tensor,
        lengths=None,
    ):
        """
        Variable-length support via `lengths` (padded batch).

        original: [B, 1, T_max]
        params:   [B, 3, num_modes, num_frames]
        lengths:  Optional [B] true lengths in samples
        """
        if lengths is not None:
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=original.device)
            else:
                lengths = lengths.to(device=original.device)
            lengths = torch.clamp(lengths.long(), min=1, max=original.shape[-1])

        # Main embeddings
        embedding = None
        if self.encoder is not None:
            embedding = self.encoder(original)

        # Modal parameter autoencoder
        modal_params = params
        if self.modal_autoencoder is not None:
            if self.modal_autoencoder_accepts_audio:
                modal_params = self.modal_autoencoder(original, params)
            else:
                modal_params, _ = self.modal_autoencoder(embedding, params)

        # Noise parameters
        noise_params = None
        if self.noise_autoencoder is not None:
            if self.noise_autoencoder_accepts_audio:
                noise_params = self.noise_autoencoder(original)
            else:
                noise_params, _ = self.noise_autoencoder(embedding)

        # Transient parameters
        transient_params = None
        if self.transient_autoencoder is not None:
            if self.transient_autoencoder_accepts_audio:
                transient_params = self.transient_autoencoder(original)
            else:
                transient_params, _ = self.transient_autoencoder(embedding)

        B, _, T_max = original.shape

        def _pad_to_Tmax(x_1ch: torch.Tensor, T: int) -> torch.Tensor:
            if x_1ch.dim() == 2:
                x_1ch = x_1ch.unsqueeze(1)
            t = x_1ch.shape[-1]
            if t == T:
                return x_1ch
            if t > T:
                return x_1ch[..., :T]
            return torch.nn.functional.pad(x_1ch, (0, T - t))

        # Fast path: fixed length batch
        if lengths is None or torch.all(lengths == lengths[0]):
            L = int(T_max if lengths is None else lengths[0].item())

            y_hat = self.modal_synth(modal_params, L)
            y_hat = _pad_to_Tmax(y_hat, T_max)

            noise = None
            if self.noise_synth is not None:
                assert noise_params is not None, "Noise params must be provided"
                n = self.noise_synth(noise_params, L)
                noise = rearrange(n, "b n -> b () n")
                noise = _pad_to_Tmax(noise, T_max)
                if self.transient_takes_noise:
                    y_hat = y_hat + noise

            if self.transient_synth is not None:
                transient = self.transient_synth(y_hat, transient_params)
                y_hat = (y_hat + transient) if self.transient_parallel else transient

            if self.noise_synth is not None and (self.transient_takes_noise is False):
                y_hat = y_hat + noise

            return y_hat

        # Variable lengths within a batch -> per-sample modal/noise, then transient once
        ys = []
        noises = [] if self.noise_synth is not None else None

        for i in range(B):
            Li = int(lengths[i].item())

            yi = self.modal_synth(modal_params[i : i + 1], Li)
            yi = _pad_to_Tmax(yi, T_max)
            ys.append(yi)

            if self.noise_synth is not None:
                assert noise_params is not None, "Noise params must be provided"
                ni = self.noise_synth(noise_params[i : i + 1], Li)
                ni = rearrange(ni, "b n -> b () n")
                ni = _pad_to_Tmax(ni, T_max)
                noises.append(ni)

        y_hat = torch.cat(ys, dim=0)

        noise = None
        if self.noise_synth is not None:
            noise = torch.cat(noises, dim=0)
            if self.transient_takes_noise:
                y_hat = y_hat + noise

        if self.transient_synth is not None:
            transient = self.transient_synth(y_hat, transient_params)
            y_hat = (y_hat + transient) if self.transient_parallel else transient

        if self.noise_synth is not None and (self.transient_takes_noise is False):
            y_hat = y_hat + noise

        return y_hat

    @staticmethod
    def _make_time_mask(lengths: torch.Tensor, T: int, device):
        t = torch.arange(T, device=device)[None, :]
        mask = (t < lengths[:, None]).to(torch.float32)
        return mask[:, None, :]

    def _do_step(self, batch):
        if len(batch) == 2:
            original, params = batch
            lengths = None
        elif len(batch) == 3:
            original, params, lengths = batch
        else:
            raise ValueError("Expected batch to be a tuple of length 2 or 3")

        y_hat = self(original, params, lengths=lengths)

        if lengths is not None:
            T = original.shape[-1]
            mask = self._make_time_mask(lengths, T, original.device)
            original = original * mask
            y_hat = y_hat * mask

        loss = self.loss_fn(y_hat, original)
        return loss, y_hat

    def training_step(self, batch, batch_idx: int):
        loss, _ = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, _ = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch, batch_idx: int):
        loss, y_hat = self._do_step(batch)
        self.log("test/loss", loss)
        if hasattr(self, "metrics"):
            for name, metric in self.metrics.items():
                self.log(f"test/{name}", metric(y_hat, batch[0]))
        return loss
