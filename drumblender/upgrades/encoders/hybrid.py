"""Single-path hybrid encoder variants.

Unlike the previous parallel hybrid design, this module now builds one encoder
path and mixes SoundStream-style / DAC-style blocks inside that path.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from einops import rearrange
from torch import nn

from drumblender.models.components import AttentionPooling
from drumblender.models.components import FiLM
from drumblender.models.components import Pad
from drumblender.upgrades.encoders.dac_style import Snake1d
from drumblender.upgrades.encoders.dac_style import _conv1d

HybridMode = Literal["soundstream", "dac", "alternate"]


class _HybridResidualUnit(nn.Module):
    def __init__(
        self,
        width: int,
        dilation: int,
        style: Literal["soundstream", "dac"],
        *,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.style = style

        if style == "dac":
            conv1 = _conv1d(
                width,
                width,
                kernel_size,
                dilation=dilation,
                use_weight_norm=use_weight_norm,
            )
            act = Snake1d(width)
            conv2 = _conv1d(width, width, 1, use_weight_norm=use_weight_norm)
            self.post_activation = nn.Identity()
        else:
            conv1 = nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=0)
            act = nn.ELU()
            conv2 = nn.Conv1d(width, width, 1)
            self.post_activation = nn.ELU()

        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            conv1,
            act,
            conv2,
        )

        if film_conditioning:
            self.film = FiLM(
                film_embedding_size,
                width,
                use_batch_norm=film_batch_norm,
            )
        else:
            self.film = None

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.net(x)
        if self.film is not None:
            y = self.film(y, film_embedding)
        y = self.post_activation(y)
        return x + y


class _HybridEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        stride: int,
        block_index: int,
        hybrid_mode: HybridMode = "alternate",
        *,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        in_channels = width // 2

        if hybrid_mode == "soundstream":
            unit_styles = ("soundstream", "soundstream", "soundstream")
            block_style = "soundstream"
        elif hybrid_mode == "dac":
            unit_styles = ("dac", "dac", "dac")
            block_style = "dac"
        else:
            # Alternate styles per block while keeping a single forward path.
            if block_index % 2 == 0:
                unit_styles = ("soundstream", "dac", "soundstream")
                block_style = "soundstream"
            else:
                unit_styles = ("dac", "soundstream", "dac")
                block_style = "dac"

        self.net = nn.ModuleList(
            [
                _HybridResidualUnit(
                    in_channels,
                    1,
                    unit_styles[0],
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
                _HybridResidualUnit(
                    in_channels,
                    3,
                    unit_styles[1],
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
                _HybridResidualUnit(
                    in_channels,
                    9,
                    unit_styles[2],
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
            ]
        )

        if block_style == "dac":
            out_conv = _conv1d(
                in_channels,
                width,
                2 * stride,
                stride=stride,
                use_weight_norm=use_weight_norm,
            )
            self.output = nn.Sequential(
                Snake1d(in_channels),
                Pad(2 * stride, 1, causal=causal),
                out_conv,
            )
        else:
            out_conv = nn.Conv1d(in_channels, width, 2 * stride, stride=stride, padding=0)
            self.output = nn.Sequential(
                Pad(2 * stride, 1, causal=causal),
                out_conv,
                nn.ELU(),
            )

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.net:
            x = layer(x, film_embedding)
        return self.output(x)


class _HybridEncoderCore(nn.Module):
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
        transpose_output: bool = False,
        use_weight_norm: bool = True,
        hybrid_mode: HybridMode = "alternate",
        fusion_mode: Literal["sum", "concat_proj"] = "concat_proj",
    ):
        super().__init__()
        # Legacy arg kept for compatibility with existing yaml/run scripts.
        _ = fusion_mode

        self.hybrid_mode = hybrid_mode
        if hybrid_mode == "dac":
            in_conv = _conv1d(
                input_channels,
                hidden_channels,
                kernel_size,
                use_weight_norm=use_weight_norm,
            )
            self.input = nn.Sequential(Pad(kernel_size, 1, causal=causal), in_conv)
        else:
            in_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=0)
            self.input = nn.Sequential(Pad(kernel_size, 1, causal=causal), in_conv)

        encoder_blocks = []
        for i, stride in enumerate(strides):
            hidden_channels *= 2
            encoder_blocks.append(
                _HybridEncoderBlock(
                    hidden_channels,
                    stride,
                    block_index=i,
                    hybrid_mode=hybrid_mode,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        if hybrid_mode == "dac":
            out_conv = _conv1d(
                hidden_channels,
                output_channels,
                3,
                use_weight_norm=use_weight_norm,
            )
            self.output = nn.Sequential(
                Snake1d(hidden_channels),
                Pad(3, 1, causal=causal),
                out_conv,
            )
        else:
            out_conv = nn.Conv1d(hidden_channels, output_channels, 3, padding=0)
            self.output = nn.Sequential(
                Pad(3, 1, causal=causal),
                out_conv,
            )

        self.transpose_output = transpose_output

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.input(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, film_embedding)
        x = self.output(x)

        if self.transpose_output:
            x = rearrange(x, "b c t -> b t c")
        return x


class HybridFrameEncoder(nn.Module):
    """Single-path hybrid encoder for frame outputs (e.g., noise params)."""

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
        hybrid_mode: HybridMode = "alternate",
        fusion_mode: Literal["sum", "concat_proj"] = "concat_proj",
    ):
        super().__init__()
        self.encoder = _HybridEncoderCore(
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
            hybrid_mode=hybrid_mode,
            fusion_mode=fusion_mode,
        )

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(x, film_embedding)


class HybridAttentionEncoder(nn.Module):
    """Single-path hybrid encoder + attention pooling (e.g., transient params)."""

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
        hybrid_mode: HybridMode = "alternate",
        fusion_mode: Literal["sum", "concat_proj"] = "concat_proj",
        transpose_output: bool = False,
    ):
        super().__init__()
        self.encoder = _HybridEncoderCore(
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
            hybrid_mode=hybrid_mode,
            fusion_mode=fusion_mode,
        )
        self.pooling = AttentionPooling(output_channels)

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.encoder(x, film_embedding)
        return self.pooling(x)
