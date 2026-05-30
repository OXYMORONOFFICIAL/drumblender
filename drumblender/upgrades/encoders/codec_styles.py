"""Codec-inspired audio encoders for DrumBlender experiments.

These modules intentionally keep the DrumBlender encoder contract:

- frame/noise encoders return ``[B, T_frame, C]`` by default
- attention/transient encoders return ``[B, C]``

They borrow architectural ideas from recent audio codecs while staying fully
trainable inside this repository instead of depending on external checkpoints.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils import weight_norm

from drumblender.models.components import AttentionPooling


def _frame_lengths(lengths: torch.Tensor, hop_size: int) -> torch.Tensor:
    return torch.div(lengths + hop_size - 1, hop_size, rounding_mode="floor").clamp_min(1)


def _mask_from_lengths(
    lengths: torch.Tensor,
    total_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = torch.arange(total_length, device=device)
    return (t.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype).unsqueeze(1)


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
        raise ValueError(f"Expected {batch_size} lengths entries, got {lengths.numel()}.")
    return lengths.long().clamp(min=1, max=max_length)


def _zero_padded_audio(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    if lengths is None:
        return x
    mask = _mask_from_lengths(
        lengths,
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    return x * mask


def _resize_frames(x: torch.Tensor, target_frames: int) -> torch.Tensor:
    if x.shape[-1] == target_frames:
        return x
    return F.interpolate(x, size=target_frames, mode="linear", align_corners=False)


def _finalize_sequence(
    x: torch.Tensor,
    *,
    input_length: int,
    target_hop_size: int,
    lengths: Optional[torch.Tensor],
    transpose_output: bool,
    return_frame_mask: bool,
):
    target_frames = int(math.ceil(input_length / float(target_hop_size)))
    x = _resize_frames(x, target_frames)

    frame_mask = None
    frame_lengths = None
    if lengths is not None:
        frame_lengths = _frame_lengths(lengths, target_hop_size)
        frame_mask = _mask_from_lengths(
            frame_lengths,
            target_frames,
            device=x.device,
            dtype=x.dtype,
        )
        x = x * frame_mask

    if transpose_output:
        x = rearrange(x, "b c t -> b t c")
        if frame_mask is not None:
            frame_mask = rearrange(frame_mask, "b 1 t -> b t 1")

    if return_frame_mask:
        return x, frame_mask, frame_lengths
    return x


def _masked_attention_pool(
    pooling: AttentionPooling,
    x: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if frame_mask is None:
        return pooling(x)

    if frame_mask.shape[1] == 1:
        valid = frame_mask.squeeze(1)
    elif frame_mask.shape[-1] == 1:
        valid = frame_mask.squeeze(-1)
    else:
        valid = frame_mask

    x = rearrange(x, "b c t -> t b c")
    x = pooling.norm(x)
    q = pooling.query.expand(1, x.shape[1], -1)
    attn, _ = pooling.attn(
        q,
        x,
        x,
        key_padding_mask=~valid.to(dtype=torch.bool),
        need_weights=False,
    )
    if pooling.keep_seq_dim:
        return rearrange(attn, "t b c -> b c t")
    return attn.squeeze(dim=0)


def _pad1d(x: torch.Tensor, left: int, right: int, mode: str) -> torch.Tensor:
    if left == 0 and right == 0:
        return x
    if mode == "reflect":
        max_pad = max(left, right)
        extra = 0
        if x.shape[-1] <= max_pad:
            extra = max_pad - x.shape[-1] + 1
            x = F.pad(x, (0, extra))
        x = F.pad(x, (left, right), mode=mode)
        if extra:
            x = x[..., :-extra]
        return x
    return F.pad(x, (left, right))


class _SamePadConv1d(nn.Module):
    """Conv1d wrapper with EnCodec/SEANet-style padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        causal: bool = False,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
    ):
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        effective = (kernel_size - 1) * dilation + 1
        total = max(0, effective - stride)
        if self.causal:
            left, right = total, 0
        else:
            right = total // 2
            left = total - right
        return self.conv(_pad1d(x, left, right, self.pad_mode))


class _SLSTM(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(channels, channels, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(rearrange(x, "b c t -> t b c"))
        y = rearrange(y, "t b c -> b c t")
        return x + y


class _SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        compress: int = 2,
        activation: type[nn.Module] = nn.ELU,
        causal: bool = False,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
    ):
        super().__init__()
        hidden = max(1, channels // compress)
        self.block = nn.Sequential(
            activation(),
            _SamePadConv1d(
                channels,
                hidden,
                kernel_size,
                dilation=dilation,
                causal=causal,
                pad_mode=pad_mode,
                use_weight_norm=use_weight_norm,
            ),
            activation(),
            _SamePadConv1d(
                hidden,
                channels,
                1,
                causal=causal,
                pad_mode=pad_mode,
                use_weight_norm=use_weight_norm,
            ),
        )
        self.shortcut = _SamePadConv1d(
            channels,
            channels,
            1,
            causal=causal,
            pad_mode=pad_mode,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class _SEANetBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_channels: int,
        *,
        hidden_channels: int = 32,
        strides: Sequence[int] = (2, 2, 4, 8),
        residual_layers: int = 1,
        residual_kernel_size: int = 3,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        dilation_base: int = 2,
        lstm_layers: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        reverse_strides: bool = True,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        activation = nn.ELU
        width = hidden_channels
        layers: list[nn.Module] = [
            _SamePadConv1d(
                input_channels,
                width,
                kernel_size,
                causal=causal,
                pad_mode=pad_mode,
                use_weight_norm=use_weight_norm,
            )
        ]

        stride_order = tuple(reversed(tuple(strides))) if reverse_strides else tuple(strides)
        for stride in stride_order:
            for layer_index in range(residual_layers):
                layers.append(
                    _SEANetResnetBlock(
                        width,
                        kernel_size=residual_kernel_size,
                        dilation=dilation_base**layer_index,
                        activation=activation,
                        causal=causal,
                        pad_mode=pad_mode,
                        use_weight_norm=use_weight_norm,
                    )
                )
            layers.extend(
                [
                    activation(),
                    _SamePadConv1d(
                        width,
                        width * 2,
                        kernel_size=2 * stride,
                        stride=stride,
                        causal=causal,
                        pad_mode=pad_mode,
                        use_weight_norm=use_weight_norm,
                    ),
                ]
            )
            width *= 2

        if lstm_layers > 0:
            layers.append(_SLSTM(width, lstm_layers))

        layers.extend(
            [
                activation(),
                _SamePadConv1d(
                    width,
                    latent_channels,
                    last_kernel_size,
                    causal=causal,
                    pad_mode=pad_mode,
                    use_weight_norm=use_weight_norm,
                ),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WavTokenizerStyleEncoder(nn.Module):
    """SEANet/LSTM waveform encoder inspired by WavTokenizer."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 128,
        latent_channels: int = 512,
        strides: Sequence[int] = (2, 2, 4, 8),
        target_hop_size: int = 128,
        residual_layers: int = 1,
        lstm_layers: int = 2,
        transpose_output: bool = True,
        causal: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.target_hop_size = int(target_hop_size)
        self.transpose_output = transpose_output
        self.backbone = _SEANetBackbone(
            input_channels,
            latent_channels,
            hidden_channels=hidden_channels,
            strides=strides,
            residual_layers=residual_layers,
            lstm_layers=lstm_layers,
            causal=causal,
            use_weight_norm=use_weight_norm,
        )
        self.proj = nn.Conv1d(latent_channels, output_channels, 1)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return _frame_lengths(lengths, self.target_hop_size)

    def forward(self, x: torch.Tensor, lengths=None, return_frame_mask: bool = False):
        lengths = _normalize_lengths(
            lengths,
            batch_size=x.shape[0],
            max_length=x.shape[-1],
            device=x.device,
        )
        input_length = x.shape[-1]
        x = _zero_padded_audio(x, lengths)
        x = self.proj(self.backbone(x))
        return _finalize_sequence(
            x,
            input_length=input_length,
            target_hop_size=self.target_hop_size,
            lengths=lengths,
            transpose_output=self.transpose_output,
            return_frame_mask=return_frame_mask,
        )


class WavTokenizerStyleAttentionEncoder(nn.Module):
    def __init__(self, output_channels: int = 128, **kwargs):
        super().__init__()
        kwargs["transpose_output"] = False
        kwargs["output_channels"] = output_channels
        self.encoder = WavTokenizerStyleEncoder(**kwargs)
        self.pooling = AttentionPooling(output_channels)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x, frame_mask, _ = self.encoder(
            x,
            lengths=lengths,
            return_frame_mask=True,
        )
        return _masked_attention_pool(self.pooling, x, frame_mask)


class BSCodecStyleEncoder(nn.Module):
    """Lightweight band-split SEANet encoder inspired by BSCodec."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 128,
        latent_channels: int = 128,
        sample_rate: int = 48000,
        bands: Sequence[tuple[float, float]] = (
            (0.0, 2000.0),
            (2000.0, 8000.0),
            (8000.0, 24000.0),
        ),
        strides: Sequence[int] = (2, 2, 4, 8),
        target_hop_size: int = 128,
        residual_layers: int = 1,
        lstm_layers: int = 1,
        transpose_output: bool = True,
        causal: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError("BSCodecStyleEncoder currently expects mono [B, 1, T] audio.")
        self.sample_rate = int(sample_rate)
        self.target_hop_size = int(target_hop_size)
        self.transpose_output = transpose_output
        self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32))
        self.encoders = nn.ModuleList(
            [
                _SEANetBackbone(
                    input_channels,
                    latent_channels,
                    hidden_channels=hidden_channels,
                    strides=strides,
                    residual_layers=residual_layers,
                    lstm_layers=lstm_layers,
                    causal=causal,
                    use_weight_norm=use_weight_norm,
                )
                for _ in bands
            ]
        )
        self.proj = nn.Conv1d(latent_channels * len(bands), output_channels, 1)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return _frame_lengths(lengths, self.target_hop_size)

    def _split_bands(self, x: torch.Tensor) -> torch.Tensor:
        wav = x.squeeze(1)
        num_samples = wav.shape[-1]
        spec = torch.fft.rfft(wav, dim=-1)
        freqs = torch.fft.rfftfreq(
            num_samples,
            d=1.0 / float(self.sample_rate),
            device=x.device,
        )

        band_wavs = []
        for low, high in self.bands.to(device=x.device):
            if high >= self.sample_rate / 2:
                mask = (freqs >= low) & (freqs <= high)
            else:
                mask = (freqs >= low) & (freqs < high)
            band = torch.fft.irfft(spec * mask.to(spec.dtype), n=num_samples, dim=-1)
            band_wavs.append(band.unsqueeze(1))
        return torch.cat(band_wavs, dim=1)

    def forward(self, x: torch.Tensor, lengths=None, return_frame_mask: bool = False):
        lengths = _normalize_lengths(
            lengths,
            batch_size=x.shape[0],
            max_length=x.shape[-1],
            device=x.device,
        )
        input_length = x.shape[-1]
        x = _zero_padded_audio(x, lengths)
        bands = self._split_bands(x)
        encoded = [
            encoder(bands[:, index : index + 1])
            for index, encoder in enumerate(self.encoders)
        ]
        x = self.proj(torch.cat(encoded, dim=1))
        return _finalize_sequence(
            x,
            input_length=input_length,
            target_hop_size=self.target_hop_size,
            lengths=lengths,
            transpose_output=self.transpose_output,
            return_frame_mask=return_frame_mask,
        )


class BSCodecStyleAttentionEncoder(nn.Module):
    def __init__(self, output_channels: int = 128, **kwargs):
        super().__init__()
        kwargs["transpose_output"] = False
        kwargs["output_channels"] = output_channels
        self.encoder = BSCodecStyleEncoder(**kwargs)
        self.pooling = AttentionPooling(output_channels)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x, frame_mask, _ = self.encoder(
            x,
            lengths=lengths,
            return_frame_mask=True,
        )
        return _masked_attention_pool(self.pooling, x, frame_mask)


class _GRN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        norm = norm / (norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * norm) + self.beta + x


class _ConvNeXtBlock1d(nn.Module):
    def __init__(self, channels: int, intermediate_channels: int):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, intermediate_channels)
        self.act = nn.GELU()
        self.grn = _GRN(intermediate_channels)
        self.pwconv2 = nn.Linear(intermediate_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = rearrange(x, "b t c -> b c t")
        return residual + x


class _APCodecBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        branch_channels: int,
        num_layers: int,
        downsample_ratio: int,
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, hidden_channels, 7, padding=3)
        self.norm = nn.LayerNorm(hidden_channels, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                _ConvNeXtBlock1d(hidden_channels, hidden_channels * 2)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_channels, eps=1e-6)
        self.out = nn.Linear(hidden_channels, branch_channels)
        self.downsample = weight_norm(
            nn.Conv1d(
                branch_channels,
                branch_channels // 2,
                7,
                stride=downsample_ratio,
                padding=3,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm(rearrange(x, "b c t -> b t c"))
        x = rearrange(x, "b t c -> b c t")
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(rearrange(x, "b c t -> b t c"))
        x = self.out(x)
        x = rearrange(x, "b t c -> b c t")
        return self.downsample(x)


class APCodecStyleEncoder(nn.Module):
    """APCodec-inspired amplitude/phase spectral encoder."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 128,
        hidden_channels: int = 256,
        branch_channels: int = 256,
        latent_channels: int = 32,
        num_layers: int = 8,
        n_fft: int = 1024,
        hop_size: int = 40,
        win_size: int = 320,
        spectral_downsample_ratio: int = 8,
        target_hop_size: int = 128,
        transpose_output: bool = True,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError("APCodecStyleEncoder expects mono [B, 1, T] audio.")
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.win_size = int(win_size)
        self.target_hop_size = int(target_hop_size)
        self.transpose_output = transpose_output
        bins = self.n_fft // 2 + 1
        self.amp = _APCodecBranch(
            bins,
            hidden_channels,
            branch_channels,
            num_layers,
            spectral_downsample_ratio,
        )
        self.phase = _APCodecBranch(
            bins,
            hidden_channels,
            branch_channels,
            num_layers,
            spectral_downsample_ratio,
        )
        self.latent = weight_norm(
            nn.Conv1d(branch_channels, latent_channels, 7, padding=3)
        )
        self.proj = nn.Conv1d(latent_channels, output_channels, 1)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return _frame_lengths(lengths, self.target_hop_size)

    def _spectrum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wav = x.squeeze(1)
        window = torch.hann_window(self.win_size, dtype=x.dtype, device=x.device)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        log_amp = torch.log(spec.abs() + 1e-5)
        phase = torch.angle(spec)
        return log_amp, phase

    def forward(self, x: torch.Tensor, lengths=None, return_frame_mask: bool = False):
        lengths = _normalize_lengths(
            lengths,
            batch_size=x.shape[0],
            max_length=x.shape[-1],
            device=x.device,
        )
        input_length = x.shape[-1]
        x = _zero_padded_audio(x, lengths)
        log_amp, phase = self._spectrum(x)
        x = torch.cat([self.amp(log_amp), self.phase(phase)], dim=1)
        x = self.proj(self.latent(x))
        return _finalize_sequence(
            x,
            input_length=input_length,
            target_hop_size=self.target_hop_size,
            lengths=lengths,
            transpose_output=self.transpose_output,
            return_frame_mask=return_frame_mask,
        )


class APCodecStyleAttentionEncoder(nn.Module):
    def __init__(self, output_channels: int = 128, **kwargs):
        super().__init__()
        kwargs["transpose_output"] = False
        kwargs["output_channels"] = output_channels
        self.encoder = APCodecStyleEncoder(**kwargs)
        self.pooling = AttentionPooling(output_channels)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x, frame_mask, _ = self.encoder(
            x,
            lengths=lengths,
            return_frame_mask=True,
        )
        return _masked_attention_pool(self.pooling, x, frame_mask)


class SpectroStreamStyleEncoder(nn.Module):
    """Trainable spectrogram encoder inspired by SpectroStream.

    This is not the TensorFlow SavedModel codec. It keeps the idea that the
    encoder should operate on multi-resolution magnitude spectra, then maps the
    result back to DrumBlender's frame contract.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 128,
        hidden_channels: int = 256,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        target_hop_size: int = 128,
        num_layers: int = 4,
        transpose_output: bool = True,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError("SpectroStreamStyleEncoder expects mono [B, 1, T] audio.")
        self.n_ffts = tuple(int(v) for v in n_ffts)
        self.target_hop_size = int(target_hop_size)
        self.transpose_output = transpose_output
        spec_channels = sum(n_fft // 2 + 1 for n_fft in self.n_ffts)
        layers: list[nn.Module] = [
            nn.Conv1d(spec_channels, hidden_channels, 1),
            nn.GELU(),
        ]
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        7,
                        padding=3,
                        groups=hidden_channels,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_channels, hidden_channels, 1),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv1d(hidden_channels, output_channels, 1))
        self.net = nn.Sequential(*layers)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return _frame_lengths(lengths, self.target_hop_size)

    def _features(self, x: torch.Tensor, target_frames: int) -> torch.Tensor:
        wav = x.squeeze(1)
        features = []
        for n_fft in self.n_ffts:
            window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
            spec = torch.stft(
                wav,
                n_fft=n_fft,
                hop_length=self.target_hop_size,
                win_length=n_fft,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            mag = torch.log1p(spec.abs())
            features.append(_resize_frames(mag, target_frames))
        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor, lengths=None, return_frame_mask: bool = False):
        lengths = _normalize_lengths(
            lengths,
            batch_size=x.shape[0],
            max_length=x.shape[-1],
            device=x.device,
        )
        input_length = x.shape[-1]
        target_frames = int(math.ceil(input_length / float(self.target_hop_size)))
        x = _zero_padded_audio(x, lengths)
        x = self.net(self._features(x, target_frames))
        return _finalize_sequence(
            x,
            input_length=input_length,
            target_hop_size=self.target_hop_size,
            lengths=lengths,
            transpose_output=self.transpose_output,
            return_frame_mask=return_frame_mask,
        )


class SpectroStreamStyleAttentionEncoder(nn.Module):
    def __init__(self, output_channels: int = 128, **kwargs):
        super().__init__()
        kwargs["transpose_output"] = False
        kwargs["output_channels"] = output_channels
        self.encoder = SpectroStreamStyleEncoder(**kwargs)
        self.pooling = AttentionPooling(output_channels)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        x, frame_mask, _ = self.encoder(
            x,
            lengths=lengths,
            return_frame_mask=True,
        )
        return _masked_attention_pool(self.pooling, x, frame_mask)
