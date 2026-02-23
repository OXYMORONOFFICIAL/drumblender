from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from drumblender.models.components import AttentionPooling
from drumblender.models.components import FiLM
from drumblender.models.components import Pad


class Snake1d(nn.Module):
    """Periodic activation used in DAC-style encoders."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    use_weight_norm: bool = True,
) -> nn.Module:
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )
    return weight_norm(conv) if use_weight_norm else conv


class _DACResidualUnit(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            _conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                use_weight_norm=use_weight_norm,
            ),
            Snake1d(channels),
            _conv1d(channels, channels, 1, use_weight_norm=use_weight_norm),
        )

        if film_conditioning:
            self.film = FiLM(
                film_embedding_size,
                channels,
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
        return x + y


class _DACEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        stride: int,
        kernel_size: int = 7,
        causal: bool = False,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        in_channels = width // 2

        self.net = nn.ModuleList(
            [
                _DACResidualUnit(
                    in_channels,
                    1,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
                _DACResidualUnit(
                    in_channels,
                    3,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
                _DACResidualUnit(
                    in_channels,
                    9,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                ),
            ]
        )
        self.output = nn.Sequential(
            Snake1d(in_channels),
            Pad(2 * stride, 1, causal=causal),
            _conv1d(
                in_channels,
                width,
                2 * stride,
                stride=stride,
                use_weight_norm=use_weight_norm,
            ),
        )

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.net:
            x = layer(x, film_embedding)
        return self.output(x)


class DACStyleEncoder(nn.Module):
    """
    DAC-style waveform encoder for experimental A/B tests.

    This is a drop-in replacement for SoundStreamEncoder at the interface level.
    """

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
    ):
        super().__init__()

        self.input = nn.Sequential(
            Pad(kernel_size, 1, causal=causal),
            _conv1d(
                input_channels,
                hidden_channels,
                kernel_size,
                use_weight_norm=use_weight_norm,
            ),
        )

        encoder_blocks = []
        for stride in strides:
            hidden_channels *= 2
            encoder_blocks.append(
                _DACEncoderBlock(
                    hidden_channels,
                    stride,
                    kernel_size=kernel_size,
                    causal=causal,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.output = nn.Sequential(
            Snake1d(hidden_channels),
            Pad(3, 1, causal=causal),
            _conv1d(
                hidden_channels,
                output_channels,
                3,
                use_weight_norm=use_weight_norm,
            ),
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


class DACStyleAttentionEncoder(nn.Module):
    """DAC-style sequence encoder + attention pooling head."""

    def __init__(
        self, input_channels: int, hidden_channels: int, output_channels: int, **kwargs
    ):
        super().__init__()
        self.encoder = DACStyleEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            **kwargs,
        )
        self.pooling = AttentionPooling(output_channels)

    def forward(
        self, x: torch.Tensor, film_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.encoder(x, film_embedding)
        x = self.pooling(x)
        return x

