"""Hybrid encoder variants for rapid A/B tests.

These modules combine SoundStream-style and DAC-style encoders and fuse their
outputs into the same interface expected by existing DrumBlender pipelines.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from drumblender.models.soundstream import SoundStreamAttentionEncoder
from drumblender.models.soundstream import SoundStreamEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleAttentionEncoder
from drumblender.upgrades.encoders.dac_style import DACStyleEncoder


class _HybridFuse(nn.Module):
    """Fuse two same-shape feature tensors."""

    def __init__(
        self,
        output_channels: int,
        mode: Literal["sum", "concat_proj"] = "concat_proj",
    ):
        super().__init__()
        self.mode = mode
        if mode == "concat_proj":
            self.proj = nn.Linear(2 * output_channels, output_channels)
        elif mode == "sum":
            self.proj = None
        else:
            raise ValueError(
                f"Unsupported fusion mode: {mode}. Use 'sum' or 'concat_proj'."
            )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Hybrid branch shape mismatch: {a.shape} vs {b.shape}")

        if self.mode == "sum":
            return 0.5 * (a + b)

        x = torch.cat([a, b], dim=-1)
        return self.proj(x)


class HybridFrameEncoder(nn.Module):
    """Hybrid encoder for frame outputs (e.g., noise params: [B, T, C])."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 8),
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        transpose_output: bool = True,
        use_weight_norm: bool = True,
        fusion_mode: Literal["sum", "concat_proj"] = "concat_proj",
    ):
        super().__init__()
        self.soundstream = SoundStreamEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            transpose_output=transpose_output,
        )
        self.dac = DACStyleEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            transpose_output=transpose_output,
            use_weight_norm=use_weight_norm,
        )
        self.fuse = _HybridFuse(output_channels=output_channels, mode=fusion_mode)

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ss = self.soundstream(x, film_embedding)
        dac = self.dac(x, film_embedding)
        return self.fuse(ss, dac)


class HybridAttentionEncoder(nn.Module):
    """Hybrid encoder for pooled embeddings (e.g., transient params: [B, C])."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        strides: tuple[int, ...] = (2, 2, 4, 8),
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = True,
        fusion_mode: Literal["sum", "concat_proj"] = "concat_proj",
    ):
        super().__init__()
        self.soundstream = SoundStreamAttentionEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            transpose_output=False,
        )
        self.dac = DACStyleAttentionEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            transpose_output=False,
            use_weight_norm=use_weight_norm,
        )
        self.fuse = _HybridFuse(output_channels=output_channels, mode=fusion_mode)

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ss = self.soundstream(x, film_embedding)
        dac = self.dac(x, film_embedding)
        return self.fuse(ss, dac)
