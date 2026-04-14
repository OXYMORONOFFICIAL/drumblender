"""
Methods for performing modal analysis on a CQT spectrogram
Uses the nnAudio CQT implementation for better resolution with low frequencies.
Integrated with Frequency-Axis Median Saliency for robust percussive modal extraction,
optimized for fast machine learning training loops.
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
        """
        Class for performing sinusoidal modelling using a CQT spectrogram.

        Args:
            sample_rate: Sample rate of the incoming audio
            hop_length: Hop length of the CQT spectrogram
            fmin: Minimum frequency of the CQT spectrogram
            n_bins: Number of bins in the CQT spectrogram
            bins_per_octave: Number of bins per octave in the CQT spectrogram
            min_length: Minimum length of a track in frames
            num_modes: Number of modes to return. If None, will return all
            threshold: Threshold for amplitude in dB for a mode to be considered
            diff_threshold: Peak matching threshold percentage
            p_kernel: Kernel size for the vertical median filter (saliency)
            **kwargs: Additional keyword arguments to pass to the nnAudio CQT
        """

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
        """
        Performs modal analysis on a batch of audio waveforms

        Args:
            audio: Audio waveform of shape (batch, samples)

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (batch, modes, frames)
        """
        assert audio.ndim == 2, "Audio must be a batch of waveforms"
        x = self.spectrogram(audio, complex=True)
        (_, _, num_hops, _) = x.shape

        # After nnAudio spectrogram, everything is on the CPU with numpy
        x = x.cpu().numpy()

        batch_freqs = []
        batch_amps = []
        batch_phases = []
        
        for i in range(x.shape[0]):
            freqs, amps, phases = self.modal_tracking(x[i], threshold=self.threshold)
            freqs, amps, phases = self.create_modal_tensors(
                freqs, amps, phases, num_hops=num_hops, min_length=self.min_length
            )

            # Convert to Hz
            freqs = torch.pow(2.0, freqs / self.bins_per_octave) * self.fmin

            # Filter out modes by energy if num_modes is specified
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
        """
        CQT spectrogram of the audio
        """
        x = self.cqt(audio, normalization_type="wrap")
        if not complex:
            x = torch.sqrt(torch.sum(torch.square(x), dim=-1))
        return x

    def modal_tracking(
        self, spec: np.ndarray, threshold: float = -20.0
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        Performs modal tracking on a CQT spectrogram using Frequency-Axis
        Median Saliency to reject broad noise while preserving isolated modes.
        Reverse-time tracking is used to prioritize percussive decays.
        """
        assert spec.ndim == 3, "Spectrogram must a complex spectrogram"
        assert spec.shape[-1] == 2, "Spectrogram must be complex"

        # Reverse the spectrogram along the temporal axis
        spec = np.flip(spec, axis=1)

        # 1. Get full magnitude spectrogram across all frames
        X_mag_full = np.sqrt(np.sum(np.square(spec), axis=-1))

        # 2. Calculate dynamic noise floor using a vertical (frequency) median filter
        B = median_filter(X_mag_full, size=(self.p_kernel, 1))

        # 3. Subtract background from original magnitude to get Saliency map 'S'
        S = np.maximum(0, X_mag_full - B)

        # 4. Convert the Saliency map to dB for peak picking
        S_db_full = 20.0 * np.log10(S + 1e-8)

        freqs = []
        amps = []
        phases = []

        # Select modes from the spectrogram
        for i in range(spec.shape[1]):
            frame = spec[:, i]
            
            # We need the true magnitude and phase for interpolation and synthesis
            X_mag = X_mag_full[:, i]
            X_phase = np.arctan2(frame[:, 1], frame[:, 0])

            # We use the computed saliency map STRICTLY for the peak picker
            frame_S_db = S_db_full[:, i]

            # Find peaks in the SALIENCY map
            peaks = peak_detection(frame_S_db, threshold)
            if len(peaks) == 0:
                continue

            # Interpolate parameters using the TRUE audio magnitudes/phases
            peaks_loc, peaks_mag, peaks_phase = peak_interpolation(
                X_mag, X_phase, peaks
            )

            # Initialize modes
            if len(freqs) == 0:
                freqs.append(list(peaks_loc))
                amps.append(list(peaks_mag))
                phases.append(list(peaks_phase))
                continue

            # Try to continue the mode from the previous frame
            for j in range(len(freqs)):
                if len(peaks_loc) > 0:
                    prev_freq = freqs[j][-1]
                    peak_diff = np.abs(peaks_loc - prev_freq)

                    # Check if the peak is within the difference threshold
                    if np.min(peak_diff) < prev_freq * self.diff_threshold:
                        closest_peak = np.argmin(peak_diff)
                        freqs[j].append(peaks_loc[closest_peak])
                        amps[j].append(peaks_mag[closest_peak])
                        phases[j].append(peaks_phase[closest_peak])

                        # Remove the peak from the list
                        peaks_loc = np.delete(peaks_loc, closest_peak)
                        peaks_mag = np.delete(peaks_mag, closest_peak)
                        peaks_phase = np.delete(peaks_phase, closest_peak)
                    else:
                        # No good matching peaks, copy the last peak with 0.0 amplitude
                        freqs[j].append(freqs[j][-1])
                        amps[j].append(0.0)
                        phases[j].append(phases[j][-1])
                else:
                    # No more peaks available, copy the last peak
                    freqs[j].append(freqs[j][-1])
                    amps[j].append(0.0)
                    phases[j].append(phases[j][-1])

            # Add any remaining peaks as new modes
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
        """
        Converts lists of modal freqs, amps, and phases into padded tensors.
        """
        num_modes = len(freqs)
        freq_env = []
        amp_env = []
        phase_env = []

        for i in range(num_modes):
            # Check if the mode meets the minimum length requirement
            if len(freqs[i]) < min_length:
                continue

            freq_env.append(torch.zeros(num_hops))
            amp_env.append(torch.zeros(num_hops))
            phase_env.append(torch.zeros(num_hops))

            # Reverse back to forward-time chronological order
            freqs[i].reverse()
            amps[i].reverse()
            phases[i].reverse()

            for h in range(num_hops):
                if h < len(freqs[i]):
                    freq_env[-1][h] = freqs[i][h]
                    amp_env[-1][h] = amps[i][h]
                    phase_env[-1][h] = phases[i][h]
                else:
                    break

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
    """
    Detect spectral peak locations.
    """
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
    Parabolic interpolation for sub-bin precision.
    """
    val = magnitude[ploc]
    lval = magnitude[ploc - 1]
    rval = magnitude[ploc + 1]

    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval + 1e-8)
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)
    ipphase = np.interp(iploc, np.arange(0, phase.size), phase)

    return iploc, ipmag, ipphase