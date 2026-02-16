"""
Audio utility functions
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
from einops import repeat


def preprocess_audio_file(
    input_file: Path,
    output_file: Path,
    sample_rate: int,
    num_samples: Optional[int] = None,
    mono: bool = True,

    # --- silent sample filter (분리) ---
    filter_silent_all: bool = True,
    silent_all_threshold_db: float = -75.0,  # ✅ "silent sample 판정"은 더 낮게

    # --- start silence cut (기존 유지) ---
    remove_start_silence: bool = True,
    start_silence_threshold_db: float = -60.0,  # ✅ "시작 무음 컷"은 기존대로

    # --- tail cut (옵션) ---
    remove_end_silence: bool = True,  # ✅ 가변길이 + 14초 제한이면 기본 OFF 권장
    tail_silence_threshold_db: float = -60.0,
    tail_fade_out_ms: float = 5.0,
    tail_peak_ratio: float = 0.02,
    min_tail_silence_ms: float = 50.0,

    # --- frame params ---
    frame_size: int = 256,
    hop_size: int = 256,

    # --- max duration (가변 길이일 때만 reject) ---
    max_duration_sec: Optional[float] = 14.0,
):
    """
    Preprocess an audio file.

    Pipeline:
      load -> (mono ch0) -> resample
      -> [optional] silent_all filter (threshold 낮게)
      -> [optional] cut_start_silence (threshold 기존 유지)
      -> [optional] cut_end_silence
      -> [optional] fixed length pad/trunc
      -> [optional] max duration reject (variable length only)
      -> save

    Notes:
      - mono=True일 때 mean downmix가 아니라 채널 0만 사용
      - normalize(peak/loudness) 없음
    """
    waveform, orig_freq = torchaudio.load(input_file)
    assert waveform.ndim == 2, "Expecting a 2D tensor, channels x samples"

    # Convert to mono: channel 0 only
    if mono:
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :]
        assert waveform.shape[0] == 1, "Expecting a mono signal"

    # Resample
    if orig_freq != sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=sample_rate
        )(waveform)

    # --- silent sample filter (start 컷과 분리) ---
    if filter_silent_all:
        if is_entirely_silent(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=silent_all_threshold_db,
        ):
            raise ValueError(f"silent_all: below {silent_all_threshold_db}dB")

    # Cut leading silence (기존 방식대로)
    if remove_start_silence:
        waveform = cut_start_silence(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=start_silence_threshold_db,
        )

    # Cut trailing silence (옵션)
    if remove_end_silence:
        min_tail_silence_samples = int((min_tail_silence_ms / 1000.0) * sample_rate)
        fade_out_samples = int((tail_fade_out_ms / 1000.0) * sample_rate)

        waveform = cut_end_silence(
            waveform,
            frame_size=frame_size,
            hop_size=hop_size,
            threshold_db=tail_silence_threshold_db,
            min_silence_samples=min_tail_silence_samples,
            peak_ratio=tail_peak_ratio,
            fade_out_samples=fade_out_samples,
        )

    # --- max duration reject (variable length only) ---
    if max_duration_sec is not None and (num_samples is None):
        max_len = int(max_duration_sec * sample_rate)
        if waveform.shape[1] > max_len:
            raise ValueError(
                f"too_long: {waveform.shape[1]} samples > {max_len} samples"
            )

    # Optional fixed length (pad/trunc)
    if num_samples is not None and waveform.shape[1] != num_samples:
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        else:
            num_pad = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, num_pad))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_file, waveform, sample_rate)


def generate_sine_wave(
    frequency, num_samples, sample_rate, stereo: bool = False
) -> torch.Tensor:
    """Generate a sine wave."""
    n = torch.arange(num_samples)
    x = torch.sin(frequency * 2 * torch.pi * n / sample_rate)
    x = repeat(x, "n -> c n", c=2 if stereo else 1)
    return x


def first_non_silent_sample(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> Union[int, None]:
    """
    Returns the index of the first non-silent sample in a waveform.
    Implementation based on Essentia StartStopCut.
    """
    assert x.ndim == 1, "Expecting a 1D tensor"
    frames = torch.split(x, frame_size)
    thrshold_power = float(np.power(10.0, threshold_db / 10.0))

    for i, frame in enumerate(frames):
        power = torch.inner(frame, frame) / frame.shape[-1]
        if power > thrshold_power:
            return i * hop_size
    return None


def cut_start_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
) -> torch.Tensor:
    """
    Removes silent samples from the beginning of a waveform.
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"

    start_samples = []
    for channel in x:
        start_sample = first_non_silent_sample(
            channel, frame_size=frame_size, hop_size=hop_size, threshold_db=threshold_db
        )
        if start_sample is not None:
            start_samples.append(start_sample)

    if len(start_samples) == 0:
        # start 컷이 실패(전체가 threshold 아래)인 경우
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    return x[:, min(start_samples) :]


def is_entirely_silent(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -75.0,
) -> bool:
    """
    True if ALL frames are below threshold_db (power dB).
    x: [C, T]
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    C, T = x.shape
    if T == 0:
        return True

    thr_power = float(np.power(10.0, threshold_db / 10.0))
    num_frames = int(math.ceil(T / hop_size))

    for i in range(num_frames):
        start = i * hop_size
        end = min(start + frame_size, T)
        frame = x[:, start:end]
        if frame.numel() == 0:
            continue
        power = (frame * frame).mean(dim=1)  # [C]
        if bool(torch.any(power > thr_power).item()):
            return False

    return True


def cut_end_silence(
    x: torch.Tensor,
    frame_size: int = 256,
    hop_size: int = 256,
    threshold_db: float = -60.0,
    min_silence_samples: int = 0,
    peak_ratio: float = 0.02,
    fade_out_samples: int = 0,

) -> torch.Tensor:
    """
    Removes silent samples from the end of a waveform.

    Cut condition:
      - trailing region length >= min_silence_samples
      - tail peak <= peak_ratio * global_peak
      - tail frames (mean power) are below threshold_db (power dB)
    """
    assert x.ndim == 2, "Expecting (channels, num_samples)"
    C, T = x.shape
    if T == 0:
        return x

    global_peak = float(x.abs().max().item())
    if global_peak <= 1e-12:
        raise ValueError("Entire wavfile is (near) zero")

    thr_power = float(np.power(10.0, threshold_db / 10.0))

    num_frames = int(math.ceil(T / hop_size))
    frame_starts = [i * hop_size for i in range(num_frames)]

    silent_flags = []
    for start in frame_starts:
        end = min(start + frame_size, T)
        frame = x[:, start:end]
        if frame.numel() == 0:
            silent = True
        else:
            power = (frame * frame).mean(dim=1)  # [C]
            silent = bool(torch.all(power <= thr_power).item())
        silent_flags.append(silent)

    # find last non-silent frame from the end
    last_non_silent_idx = None
    for i in range(num_frames - 1, -1, -1):
        if not silent_flags[i]:
            last_non_silent_idx = i
            break

    if last_non_silent_idx is None:
        raise ValueError(f"Entire wavfile below threshold level {threshold_db}dB")

    candidate_cut = min(frame_starts[last_non_silent_idx] + frame_size, T)
    trailing_len = T - candidate_cut
    if trailing_len < min_silence_samples:
        return x

    tail_peak = (
        float(x[:, candidate_cut:].abs().max().item()) if candidate_cut < T else 0.0
    )
    if tail_peak <= peak_ratio * global_peak:
        y = x[:, :candidate_cut]

        # ✅ 끝부분 5ms(=fade_out_samples) 선형 페이드 아웃
        if fade_out_samples and fade_out_samples > 0:
            T2 = y.shape[1]
            n = min(int(fade_out_samples), T2)
            if n > 1:
                ramp = torch.linspace(1.0, 0.0, steps=n, device=y.device, dtype=y.dtype)
                y[:, T2 - n : T2] = y[:, T2 - n : T2] * ramp.unsqueeze(0)

        return y


    return x