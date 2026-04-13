from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from drumblender.models.components import AttentionPooling
from drumblender.models.components import FiLM
from drumblender.models.components import Pad


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

    return lengths.long().clamp(min=1, max=max_length)


def _make_time_mask(
    lengths: torch.Tensor,
    *,
    total_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    timeline = torch.arange(total_length, device=device).unsqueeze(0)
    return (timeline < lengths.unsqueeze(1)).to(dtype).unsqueeze(1)


def _conv_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    causal: bool = False,
) -> torch.Tensor:
    pad = dilation * (kernel_size - 1)
    if causal:
        left_pad = pad
        right_pad = 0
    else:
        left_pad = pad // 2
        right_pad = pad // 2

    numer = lengths + left_pad + right_pad - dilation * (kernel_size - 1) - 1
    return torch.div(numer, stride, rounding_mode="floor").add(1).clamp_min(1)


def _masked_temporal_conv(
    x: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    *,
    pad_layer: nn.Module,
    conv: nn.Conv1d,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x * mask
    x_padded = pad_layer(x)
    mask_padded = pad_layer(mask)
    y = conv(x_padded)

    kernel_size = int(conv.kernel_size[0])
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    mask_kernel = mask_padded.new_ones(1, 1, kernel_size)
    valid_count = F.conv1d(
        mask_padded,
        mask_kernel,
        stride=stride,
        dilation=dilation,
    )

    out_lengths = _conv_output_lengths(
        lengths,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        causal=causal,
    )
    out_mask = _make_time_mask(
        out_lengths,
        total_length=y.shape[-1],
        device=y.device,
        dtype=y.dtype,
    )

    scale = (float(kernel_size) / valid_count.clamp_min(1.0)).to(dtype=y.dtype)
    if conv.bias is not None:
        bias = conv.bias.view(1, -1, 1)
        y = (y - bias) * scale + bias
    else:
        y = y * scale

    y = y * out_mask
    return y, out_mask, out_lengths


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
        self.causal = causal
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

    def forward_masked(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, out_mask, out_lengths = _masked_temporal_conv(
            x,
            mask,
            lengths,
            pad_layer=self.net[0],
            conv=self.net[1],
            causal=self.causal,
        )
        y = self.net[2](y)
        y = self.net[3](y) * out_mask
        if self.film is not None:
            y = self.film(y, film_embedding) * out_mask
        x = (x + y) * out_mask
        return x, out_mask, out_lengths


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
        self.causal = causal
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

    def forward_masked(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.net:
            x, mask, lengths = layer.forward_masked(x, mask, lengths, film_embedding)
        x = self.output[0](x) * mask
        return _masked_temporal_conv(
            x,
            mask,
            lengths,
            pad_layer=self.output[1],
            conv=self.output[2],
            causal=self.causal,
        )


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
        padding_aware_backbone: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.strides = tuple(strides)
        self.causal = causal
        self.transpose_output = transpose_output
        self.padding_aware_backbone = padding_aware_backbone

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

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = _conv_output_lengths(
            lengths,
            kernel_size=self.kernel_size,
            stride=1,
            dilation=1,
            causal=self.causal,
        )
        for stride in self.strides:
            out = _conv_output_lengths(
                out,
                kernel_size=2 * stride,
                stride=stride,
                dilation=1,
                causal=self.causal,
            )
        out = _conv_output_lengths(
            out,
            kernel_size=3,
            stride=1,
            dilation=1,
            causal=self.causal,
        )
        return out.clamp_min(1)

    def _encode_plain(
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, film_embedding)
        return self.output(x)

    def _encode_masked(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = _make_time_mask(
            lengths,
            total_length=x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x, mask, lengths = _masked_temporal_conv(
            x,
            mask,
            lengths,
            pad_layer=self.input[0],
            conv=self.input[1],
            causal=self.causal,
        )
        for encoder_block in self.encoder_blocks:
            x, mask, lengths = encoder_block.forward_masked(
                x,
                mask,
                lengths,
                film_embedding,
            )
        x = self.output[0](x) * mask
        return _masked_temporal_conv(
            x,
            mask,
            lengths,
            pad_layer=self.output[1],
            conv=self.output[2],
            causal=self.causal,
        )

    def forward(
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
        lengths=None,
        return_frame_mask: bool = False,
    ):
        lengths = _normalize_lengths(
            lengths,
            batch_size=x.shape[0],
            max_length=x.shape[-1],
            device=x.device,
        )

        if self.padding_aware_backbone and lengths is not None:
            x, frame_mask, frame_lengths = self._encode_masked(x, lengths, film_embedding)
        else:
            x = self._encode_plain(x, film_embedding)
            frame_mask = None
            frame_lengths = None
            if return_frame_mask and lengths is not None:
                frame_lengths = self.output_lengths(lengths)
                frame_mask = _make_time_mask(
                    frame_lengths,
                    total_length=x.shape[-1],
                    device=x.device,
                    dtype=x.dtype,
                )

        if self.transpose_output:
            x = rearrange(x, "b c t -> b t c")
            if frame_mask is not None:
                frame_mask = rearrange(frame_mask, "b 1 t -> b t 1")

        if return_frame_mask:
            return x, frame_mask, frame_lengths
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
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
        lengths=None,
    ) -> torch.Tensor:
        if lengths is not None and self.encoder.padding_aware_backbone:
            x, frame_mask, _ = self.encoder(
                x,
                film_embedding,
                lengths=lengths,
                return_frame_mask=True,
            )
            if frame_mask is not None:
                x = x * frame_mask
        else:
            x = self.encoder(x, film_embedding)
        x = self.pooling(x)
        return x


class DACStyleSequenceEncoder(nn.Module):
    """
    DAC-style waveform encoder with a lightweight temporal LSTM tail.

    This variant is intended for frame-wise outputs such as the noise parameter path.
    It preserves the existing DAC convolutional backbone and only adds temporal
    modeling after the stride-128 frame sequence has already been formed.
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
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.0,
        padding_aware_backbone: bool = False,
        length_aware_lstm: bool = False,
        mask_invalid_frames: bool = False,
    ):
        super().__init__()
        self.encoder = DACStyleEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            strides=strides,
            causal=causal,
            film_conditioning=film_conditioning,
            film_embedding_size=film_embedding_size,
            film_batch_norm=film_batch_norm,
            transpose_output=True,
            use_weight_norm=use_weight_norm,
            padding_aware_backbone=padding_aware_backbone,
        )
        self.temporal = nn.LSTM(
            input_size=output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(lstm_hidden_size, output_channels)
        self.transpose_output = transpose_output
        self.length_aware_lstm = length_aware_lstm
        self.mask_invalid_frames = mask_invalid_frames

    def forward(
        self,
        x: torch.Tensor,
        film_embedding: Optional[torch.Tensor] = None,
        lengths=None,
    ) -> torch.Tensor:
        need_frame_metadata = (
            lengths is not None
            and (
                self.length_aware_lstm
                or self.mask_invalid_frames
                or self.encoder.padding_aware_backbone
            )
        )

        frame_mask = None
        frame_lengths = None
        if need_frame_metadata:
            x, frame_mask, frame_lengths = self.encoder(
                x,
                film_embedding,
                lengths=lengths,
                return_frame_mask=True,
            )
        else:
            x = self.encoder(x, film_embedding)

        if self.length_aware_lstm and frame_lengths is not None:
            packed = pack_padded_sequence(
                x,
                frame_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed, _ = self.temporal(packed)
            x, _ = pad_packed_sequence(
                packed,
                batch_first=True,
                total_length=x.shape[1],
            )
        else:
            x, _ = self.temporal(x)

        x = self.proj(x)

        if self.mask_invalid_frames and frame_mask is not None:
            x = x * frame_mask.to(dtype=x.dtype)

        if not self.transpose_output:
            x = rearrange(x, "b t c -> b c t")
        return x
