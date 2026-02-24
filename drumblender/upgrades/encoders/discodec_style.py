from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from drumblender.models.components import AttentionPooling
from drumblender.models.components import FiLM
from drumblender.models.components import Pad


def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    groups: int = 1,
    use_weight_norm: bool = False,
) -> nn.Module:
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        groups=groups,
    )
    return weight_norm(conv) if use_weight_norm else conv


class _LayerNorm1d(nn.Module):
    """Channel-wise layer norm for tensors shaped [B, C, T]."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c t -> b t c")
        x = self.norm(x)
        x = rearrange(x, "b t c -> b c t")
        return x


class _GRN1d(nn.Module):
    """Global response normalization (ConvNeXt-v2 style) for 1D features."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class _DisCodecResidualUnit(nn.Module):
    """Depthwise + pointwise residual block inspired by DisCodec/ConvNeXt family."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        *,
        kernel_size: int = 7,
        causal: bool = False,
        expansion: int = 2,
        use_grn: bool = True,
        layer_scale_init: float = 1e-4,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        inner_channels = int(channels * expansion)

        self.dw = nn.Sequential(
            Pad(kernel_size, dilation, causal=causal),
            _conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                groups=channels,
                use_weight_norm=use_weight_norm,
            ),
        )
        self.norm = _LayerNorm1d(channels)
        self.pw1 = _conv1d(
            channels,
            inner_channels,
            1,
            use_weight_norm=use_weight_norm,
        )
        self.act = nn.GELU()
        self.grn = _GRN1d(inner_channels) if use_grn else nn.Identity()
        self.pw2 = _conv1d(
            inner_channels,
            channels,
            1,
            use_weight_norm=use_weight_norm,
        )

        if film_conditioning:
            self.film = FiLM(
                film_embedding_size,
                channels,
                use_batch_norm=film_batch_norm,
            )
        else:
            self.film = None

        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1), float(layer_scale_init))
        )

    def forward(
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.grn(y)
        y = self.pw2(y)

        if self.film is not None:
            y = self.film(y, film_embedding)

        return x + self.layer_scale * y


class _TemporalSelfAttentionBlock(nn.Module):
    """Lightweight self-attention block over time for bottleneck refinement."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C]
        y = rearrange(x, "b c t -> b t c")
        y_attn = self.norm1(y)
        y_attn, _ = self.attn(y_attn, y_attn, y_attn, need_weights=False)
        y = y + y_attn
        y = y + self.mlp(self.norm2(y))
        return rearrange(y, "b t c -> b c t")


class _DisCodecEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        stride: int,
        *,
        kernel_size: int = 7,
        causal: bool = False,
        block_depth: int = 2,
        residual_dilations: tuple[int, ...] = (1, 3, 9),
        expansion: int = 2,
        use_grn: bool = True,
        layer_scale_init: float = 1e-4,
        film_conditioning: bool = False,
        film_embedding_size: int = 128,
        film_batch_norm: bool = False,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        in_channels = width // 2

        units = []
        for i in range(int(block_depth)):
            dilation = residual_dilations[i % len(residual_dilations)]
            units.append(
                _DisCodecResidualUnit(
                    channels=in_channels,
                    dilation=int(dilation),
                    kernel_size=kernel_size,
                    causal=causal,
                    expansion=expansion,
                    use_grn=use_grn,
                    layer_scale_init=layer_scale_init,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.units = nn.ModuleList(units)

        self.down = nn.Sequential(
            _LayerNorm1d(in_channels),
            nn.GELU(),
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
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for unit in self.units:
            x = unit(x, film_embedding=film_embedding)
        return self.down(x)


class DisCodecStyleEncoder(nn.Module):
    """
    DisCodec-inspired encoder for regression tasks (noise/transient parameter paths).

    Design notes:
    - Keeps the same I/O contract as SoundStream/DAC encoders.
    - Uses ConvNeXt-like depthwise residual units + optional bottleneck attention.
    - Defaults preserve total stride 128 to stay aligned with current OLA setup.
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
        use_weight_norm: bool = False,
        block_depth: int = 2,
        residual_dilations: tuple[int, ...] = (1, 3, 9),
        expansion: int = 2,
        use_grn: bool = True,
        layer_scale_init: float = 1e-4,
        bottleneck_attention: bool = True,
        attention_layers: int = 1,
        attention_heads: int = 4,
        attention_mlp_ratio: int = 2,
        attention_dropout: float = 0.0,
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
                _DisCodecEncoderBlock(
                    width=hidden_channels,
                    stride=int(stride),
                    kernel_size=kernel_size,
                    causal=causal,
                    block_depth=block_depth,
                    residual_dilations=residual_dilations,
                    expansion=expansion,
                    use_grn=use_grn,
                    layer_scale_init=layer_scale_init,
                    film_conditioning=film_conditioning,
                    film_embedding_size=film_embedding_size,
                    film_batch_norm=film_batch_norm,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        if bottleneck_attention and int(attention_layers) > 0:
            attn_blocks = []
            for _ in range(int(attention_layers)):
                attn_blocks.append(
                    _TemporalSelfAttentionBlock(
                        channels=hidden_channels,
                        num_heads=int(attention_heads),
                        mlp_ratio=int(attention_mlp_ratio),
                        dropout=float(attention_dropout),
                    )
                )
            self.bottleneck_attn = nn.Sequential(*attn_blocks)
        else:
            self.bottleneck_attn = nn.Identity()

        self.output = nn.Sequential(
            _LayerNorm1d(hidden_channels),
            nn.GELU(),
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
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, film_embedding)

        x = self.bottleneck_attn(x)
        x = self.output(x)

        if self.transpose_output:
            x = rearrange(x, "b c t -> b t c")
        return x


class DisCodecStyleAttentionEncoder(nn.Module):
    """DisCodec-style sequence encoder + attention pooling head."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        **kwargs,
    ):
        super().__init__()
        self.encoder = DisCodecStyleEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            **kwargs,
        )
        self.pooling = AttentionPooling(output_channels)

    def forward(
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encoder(x, film_embedding)
        x = self.pooling(x)
        return x
