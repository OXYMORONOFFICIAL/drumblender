"""
Methods for performing modal analysis on a CQT spectrogram
Based on Sinusoidal Modeling Synthesis (SMS) Toolbox by Xavier Serra and Julius Smith
Uses the nnAudio CQT implementation for better resolution with low frequncies.
Integrated with Frequency-Axis Median Saliency for robust percussive modal extraction.
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import median_filter

try:
    import nnAudio.features as features
except ImportError:
    features = None


class CQTModalAnalysis:
    def __init__(
        self,
        sample_rate: int,
        hop_length: int = 64,
        fmin: float = 24.0,
        n_bins: int = 96,
        bins_per_octave: int = 12,
        min_length: int = 20,
        num_modes: Optional[int] = None,
        threshold: float = -20.0,
        diff_threshold: float = 2.5,
        p_kernel: int = 21,  # Width of the frequency-axis median filter
        **kwargs,
    ) -> None:

        if features is None:
            raise ImportError(
                "nnAudio is not installed. Please install it to use this class."
                'You can install it with `pip install -e ".[modal]"`'
            )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.min_length = min_length
        self.num_modes = num_modes
        self.threshold = threshold
        self.diff_threshold = diff_threshold / 100.0
        self.p_kernel = p_kernel

        self.cqt = features.CQT(
            sr=sample_rate,
            fmin=fmin,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins,
            output_format="Complex",
            **kwargs,
        )

    def __call__(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert audio.ndim == 2, "Audio must be a batch of waveforms"
        x = self.spectrogram(audio, complex=True)
        (_, _, num_hops, _) = x.shape

        x = x.cpu().numpy()

        batch_freqs = []
        batch_amps = []
        batch_phases = []
        
        for i in range(x.shape[0]):
            freqs, amps, phases = self.modal_tracking(x[i], threshold=self.threshold)
            freqs, amps, phases = self.create_modal_tensors(
                freqs, amps, phases, num_hops=num_hops, min_length=self.min_length
            )

            # [수정됨] 패딩된 빈 프레임(0)을 기록해둡니다.
            active_mask = (freqs > 0).float()

            # Convert to Hz
            freqs = torch.pow(2.0, freqs / self.bins_per_octave) * self.fmin
            
            # [수정됨] 패딩된 영역이 24.0Hz로 둔갑하지 않도록 다시 0.0Hz로 강제 처리
            freqs = freqs * active_mask

            if self.num_modes is not None:
                modal_energy = amps.sum(dim=1)
                modal_energy, idx = torch.sort(modal_energy, descending=True)
                freqs = freqs[idx[: self.num_modes]]
                amps = amps[idx[: self.num_modes]]
                phases = phases[idx[: self.num_modes]]

            batch_freqs.append(freqs)
            batch_amps.append(amps)
            batch_phases.append(phases)

        batch_freqs = torch.stack(batch_freqs, dim=0)
        batch_amps = torch.stack(batch_amps, dim=0)
        batch_phases = torch.stack(batch_phases, dim=0)
        return batch_freqs, batch_amps, batch_phases

    def spectrogram(self, audio: torch.Tensor, complex: bool = True) -> torch.Tensor:
        x = self.cqt(audio, normalization_type="wrap")
        if not complex:
            x = torch.sqrt(torch.sum(torch.square(x), dim=-1))
        return x

    def modal_tracking(
        self, spec: np.ndarray, threshold: float = -20.0
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:

        assert spec.ndim == 3, "Spectrogram must a complex spectrogram"
        assert spec.shape[-1] == 2, "Spectrogram must be complex"

        # 역방향 추적 유지 (Percussive audio decay 대응)
        spec = np.flip(spec, axis=1)

        X_mag_full = np.sqrt(np.sum(np.square(spec), axis=-1))
        B = median_filter(X_mag_full, size=(self.p_kernel, 1))
        S = np.maximum(0, X_mag_full - B)
        S_db_full = 20.0 * np.log10(S + 1e-8)

        freqs = []
        amps = []
        phases = []

        for i in range(spec.shape[1]):
            frame = spec[:, i]
            X_mag = X_mag_full[:, i]
            X_phase = np.arctan2(frame[:, 1], frame[:, 0])
            frame_S_db = S_db_full[:, i]

            peaks = peak_detection(frame_S_db, threshold)
            if len(peaks) == 0:
                continue

            peaks_loc, peaks_mag, peaks_phase = peak_interpolation(
                X_mag, X_phase, peaks
            )

            if len(freqs) == 0:
                freqs.append(list(peaks_loc))
                amps.append(list(peaks_mag))
                phases.append(list(peaks_phase))
                continue

            for j in range(len(freqs)):
                if len(peaks_loc) > 0:
                    prev_freq = freqs[j][-1]
                    peak_diff = np.abs(peaks_loc - prev_freq)

                    if np.min(peak_diff) < prev_freq * self.diff_threshold:
                        closest_peak = np.argmin(peak_diff)
                        freqs[j].append(peaks_loc[closest_peak])
                        amps[j].append(peaks_mag[closest_peak])
                        phases[j].append(peaks_phase[closest_peak])

                        peaks_loc = np.delete(peaks_loc, closest_peak)
                        peaks_mag = np.delete(peaks_mag, closest_peak)
                        peaks_phase = np.delete(peaks_phase, closest_peak)
                    else:
                        freqs[j].append(freqs[j][-1])
                        amps[j].append(0.0)
                        phases[j].append(phases[j][-1])
                else:
                    freqs[j].append(freqs[j][-1])
                    amps[j].append(0.0)
                    phases[j].append(phases[j][-1])

            for peak in peaks_loc:
                freqs.append([peak])
            for peak in peaks_mag:
                amps.append([peak])
            for peak in peaks_phase:
                phases.append([peak])

        return freqs, amps, phases

    def create_modal_tensors(
        self,
        freqs: List[List[float]],
        amps: List[List[float]],
        phases: List[List[float]],
        num_hops: int,
        min_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_modes = len(freqs)
        freq_env = []
        amp_env = []
        phase_env = []

        for i in range(num_modes):
            if len(freqs[i]) < min_length:
                continue

            freq_env.append(torch.zeros(num_hops))
            amp_env.append(torch.zeros(num_hops))
            phase_env.append(torch.zeros(num_hops))

            freq_vals = freqs[i][::-1]
            amp_vals = amps[i][::-1]
            phase_vals = phases[i][::-1]
            track_len = min(num_hops, len(freq_vals))

            if track_len > 0:
                dtype = freq_env[-1].dtype
                freq_env[-1][:track_len] = torch.as_tensor(
                    freq_vals[:track_len], dtype=dtype
                )
                amp_env[-1][:track_len] = torch.as_tensor(
                    amp_vals[:track_len], dtype=dtype
                )
                phase_env[-1][:track_len] = torch.as_tensor(
                    phase_vals[:track_len], dtype=dtype
                )

        if not freq_env:
            return torch.zeros((0, num_hops)), torch.zeros((0, num_hops)), torch.zeros((0, num_hops))

        freq_env = torch.stack(freq_env)
        amp_env = torch.stack(amp_env)
        phase_env = torch.stack(phase_env)

        return freq_env, amp_env, phase_env

    def frequencies(self) -> np.ndarray:
        return self.cqt.frequencies


def peak_detection(
    x: np.ndarray,
    threshold: float,
) -> np.ndarray:
    thresh = np.where(np.greater(x[1:-1], threshold), x[1:-1], 0)
    next_minor = np.where(x[1:-1] > x[2:], x[1:-1], 0)
    prev_minor = np.where(x[1:-1] > x[:-2], x[1:-1], 0)
    ploc = thresh * next_minor * prev_minor
    ploc = ploc.nonzero()[0] + 1
    return ploc


def peak_interpolation(
    magnitude: np.ndarray,
    phase: np.ndarray,
    ploc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parabolic interpolation for sub-bin precision (with numerical stabilization).
    """
    val = magnitude[ploc]
    lval = magnitude[ploc - 1]
    rval = magnitude[ploc + 1]

    # [수정됨] 수치 안정화 패치 (NaN, Inf 방지)
    denominator = lval - 2 * val + rval
    
    shift = np.zeros_like(ploc, dtype=np.float32)
    valid_mask = np.abs(denominator) > 1e-6
    
    if np.any(valid_mask):
        shift[valid_mask] = 0.5 * (lval[valid_mask] - rval[valid_mask]) / denominator[valid_mask]
    
    shift = np.clip(shift, -0.5, 0.5)

    iploc = ploc + shift
    ipmag = val - 0.25 * (lval - rval) * shift
    ipphase = np.interp(iploc, np.arange(0, phase.size), phase)

    return iploc, ipmag, ipphase
