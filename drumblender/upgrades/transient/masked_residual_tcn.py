from typing import Optional

import torch
import torch.nn as nn
from einops import repeat

from drumblender.models.tcn import TCN


class MaskedResidualTransientTCN(nn.Module):
    """
    Optional transient upgrade that keeps baseline TCN topology, then adds:

    1) residual mode: y = x + (tcn(x) - x)
    2) time mask: constrain transient edits to early time region

    Both switches are independently configurable and default to ON for the
    upgrade preset. Keeping this class outside `drumblender/synths/` allows
    baseline behavior to stay unchanged.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        dilation_base: int = 2,
        dilation_blocks: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 13,
        film_conditioning: bool = False,
        film_embedding_size: Optional[int] = None,
        film_batch_norm: bool = True,
        transient_conditioning: bool = False,
        transient_conditioning_channels: int = 32,
        transient_conditioning_length: int = 24000,
        residual_mode: bool = True,
        mask_enabled: bool = True,
        sample_rate: int = 48000,
        fade_start_ms: float = 15.0,
        fade_end_ms: float = 70.0,
        tail_gain: float = 0.0,
    ):
        super().__init__()
        self.tcn = TCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dilation_base=dilation_base,
            dilation_blocks=dilation_blocks,
            num_layers=num_layers,
            kernel_size=kernel_size,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
        )

        if transient_conditioning:
            p = (
                torch.randn(
                    1, transient_conditioning_channels, transient_conditioning_length
                )
                / transient_conditioning_channels
            )
            self.transient_conditioning = nn.Parameter(p, requires_grad=True)

        self.residual_mode = bool(residual_mode)
        self.mask_enabled = bool(mask_enabled)
        self.sample_rate = int(sample_rate)
        self.fade_start_ms = float(fade_start_ms)
        self.fade_end_ms = float(fade_end_ms)
        self.tail_gain = float(tail_gain)

    def _time_mask(self, length: int, device, dtype) -> torch.Tensor:
        if not self.mask_enabled:
            return torch.ones(1, 1, length, device=device, dtype=dtype)

        start = int(round(self.fade_start_ms * self.sample_rate / 1000.0))
        end = int(round(self.fade_end_ms * self.sample_rate / 1000.0))
        start = max(0, min(start, length))
        end = max(start, min(end, length))

        mask = torch.ones(length, device=device, dtype=dtype)
        tail = torch.tensor(self.tail_gain, device=device, dtype=dtype)

        if end > start:
            n = end - start
            t = torch.linspace(0.0, 1.0, steps=n, device=device, dtype=dtype)
            # Smooth cosine fade from 1.0 to tail_gain.
            fade = 0.5 * (1.0 + torch.cos(torch.pi * t))
            mask[start:end] = tail + (1.0 - tail) * fade
        elif start < length:
            mask[start:] = tail

        if end < length:
            mask[end:] = tail

        return mask.view(1, 1, -1)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        x_in = x
        if hasattr(self, "transient_conditioning"):
            cond = repeat(self.transient_conditioning, "1 c l -> b c l", b=x.size(0))
            if cond.size(-1) < x.size(-1):
                cond = torch.nn.functional.pad(cond, (0, x.size(-1) - cond.size(-1)))
            elif cond.size(-1) > x.size(-1):
                cond = cond[..., : x.size(-1)]
            x_in = torch.cat([x, cond], dim=1)

        y = self.tcn(x_in, embedding)
        delta = (y - x) if self.residual_mode else y
        delta = delta * self._time_mask(x.size(-1), x.device, x.dtype)
        return x + delta if self.residual_mode else delta

