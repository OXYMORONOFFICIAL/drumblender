"""
Methods for performing modal analysis on a CQT spectrogram
Based on Sinusoidal Modeling Synthesis (SMS) Toolbox by Xavier Serra and Julius Smith
Uses the nnAudio CQT implementation for better resolution with low frequncies.
Integrated with Frequency-Axis Median Saliency for robust percussive modal extraction.
"""
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import median_filter

# nnAudio may not be installed -- some of the methods will not work here,
# but they are only required for pre-processing the audio so we won't
# raise an error, unless the user tries to use a method that requires nnAudio
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

        # For each item in the batch perform modal tracking
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

            # Filter out modes
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

        Args:
            audio: Audio waveform of shape (batch, samples)
            complex: Whether to return a complex spectrogram, or a magnitude spectrogram

        Returns:
            A complex CQT spectrogram of shape (batch, bins, frames, 2) or a magnitude
            spectrogram of shape (batch, bins, frames) if complex is false
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
        Median Saliency to reject broad noise (cymbals) while perfectly
        preserving isolated modes (kicks/toms) across reverse-time tracking.

        Args:
            spec: A complex CQT spectrogram of shape (bins, frames, 2)
            threshold: Threshold for peak detection in dB

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (modes, frames)
        """

        assert spec.ndim == 3, "Spectrogram must a complex spectrogram"
        assert spec.shape[-1] == 2, "Spectrogram must be complex"

        # Reverse the spectrogram along the temporal axis
        # Assuming percussive audio here and that the most
        # important modes will have the longest decay
        spec = np.ascontiguousarray(np.flip(spec, axis=1), dtype=np.float32)
        n_frames = spec.shape[1]
        diff_threshold = self.diff_threshold

        # =====================================================================
        # NEW: PRE-COMPUTE SALIENCY MAP
        # =====================================================================
        # 1. Get full magnitude spectrogram across all frames
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
        X_mag_full = np.hypot(spec_real, spec_imag).astype(np.float32, copy=False)
        X_phase_full = np.arctan2(spec_imag, spec_real).astype(np.float32, copy=False)

        # 2. Calculate dynamic noise floor using a vertical (frequency) median filter
        B = median_filter(X_mag_full, size=(self.p_kernel, 1)).astype(
            np.float32, copy=False
        )

        # 3. Subtract background from original magnitude to get Saliency map 'S'
        S_db_full = X_mag_full - B
        np.maximum(S_db_full, 1e-8, out=S_db_full)

        # 4. Convert the Saliency map to dB for peak picking
        np.log10(S_db_full, out=S_db_full)
        S_db_full *= 20.0
        # =====================================================================

        peak_mask = peak_detection_fast(S_db_full, threshold)

        # Initialize lists to store each mode's frequency, amplitude, and phase
        freqs = []
        amps = []
        phases = []

        # Select modes from the spectrogram
        for i in range(n_frames):
            peaks = np.flatnonzero(peak_mask[:, i]) + 1
            if peaks.size == 0:
                continue

            X_mag = X_mag_full[:, i]
            X_phase = X_phase_full[:, i]

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
            remaining = int(peaks_loc.size)
            if remaining > 0:
                available = np.ones(remaining, dtype=bool)
            else:
                available = None

            for j in range(len(freqs)):
                if remaining > 0:
                    # Find difference between previous peak and current peaks
                    prev_freq = freqs[j][-1]
                    peak_diff = np.abs(peaks_loc - prev_freq)
                    peak_diff[~available] = np.inf

                    # If the difference is less than the threshold of the previous
                    # frequency, then we assume that the peak is the same mode
                    closest_peak = int(np.argmin(peak_diff))
                    if peak_diff[closest_peak] < prev_freq * diff_threshold:
                        freqs[j].append(peaks_loc[closest_peak])
                        amps[j].append(peaks_mag[closest_peak])
                        phases[j].append(peaks_phase[closest_peak])

                        available[closest_peak] = False
                        remaining -= 1
                    else:
                        # No good matching peaks, just copy the last peak, but
                        # with an amplitude of 0.
                        freqs[j].append(freqs[j][-1])
                        amps[j].append(0.0)
                        phases[j].append(phases[j][-1])
                else:
                    # If there are no more peaks, just copy the last peak
                    freqs[j].append(freqs[j][-1])
                    amps[j].append(0.0)
                    phases[j].append(phases[j][-1])

            # Add any remaining peaks as new modes
            if remaining > 0:
                new_peak_idx = np.flatnonzero(available)
                for peak in peaks_loc[new_peak_idx]:
                    freqs.append([peak])

                for peak in peaks_mag[new_peak_idx]:
                    amps.append([peak])

                for peak in peaks_phase[new_peak_idx]:
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
        Converts lists of modal freqs, amps, and phases into
        tensors of shape (modes, num_hops). Input lists will be of
        different lengths so they will be padded with zeros to be of length num_hops.
        Modes are also filtered out if they are shorter than min_length.

        Args:
            freqs: List of list of frequencies of shape (modes, frames)
            amps: List of list of amplitudes of shape (modes, frames)
            phases: List of list of phases of shape (modes, frames)
            num_hops: Number of frames in the spectrogram
            min_length: Minimum length of a mode to be included

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (modes, num_hops)
        """

        num_modes = len(freqs)
        freq_env = []
        amp_env = []
        phase_env = []

        for i in range(num_modes):
            # Check if the mode is long enough
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
            empty = torch.zeros((0, num_hops))
            return empty, empty.clone(), empty.clone()

        freq_env = torch.stack(freq_env)
        amp_env = torch.stack(amp_env)
        phase_env = torch.stack(phase_env)

        return freq_env, amp_env, phase_env

    def frequencies(self) -> np.ndarray:
        """
        Returns the frequencies of the CQT bins
        """
        return self.cqt.frequencies


def peak_detection(
    x: np.ndarray,  # magnitude spectrum
    threshold: float,  # threshold
) -> np.ndarray:
    """
    Detect spectral peak locations
    From: https://github.com/MTG/sms-tools

    Args:
        x: magnitude spectrum
        threshold: threshold for peak picking

    Returns:
        Peak locations
    """
    thresh = np.where(
        np.greater(x[1:-1], threshold), x[1:-1], 0
    )  # locations above threshold
    next_minor = np.where(
        x[1:-1] > x[2:], x[1:-1], 0
    )  # locations higher than the next one
    prev_minor = np.where(
        x[1:-1] > x[:-2], x[1:-1], 0
    )  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps
    return ploc


def peak_interpolation(
    magnitude: np.ndarray,
    phase: np.ndarray,
    ploc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate peak values using parabolic interpolation
    From: https://github.com/MTG/sms-tools

    Args:
        magnitude: magnitude spectrum
        phase: phase spectrum
        ploc: peak locations

    Returns:
        Interpolated peak locations, magnitudes, and phases
    """
    # Magnitude of the peak bin and its neighbours
    val = magnitude[ploc]
    lval = magnitude[ploc - 1]
    rval = magnitude[ploc + 1]

    # Parabolic interpolation
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval + 1e-8)
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)
    left = np.clip(np.floor(iploc).astype(np.int32), 0, phase.size - 1)
    right = np.minimum(left + 1, phase.size - 1)
    frac = iploc - left
    ipphase = phase[left] + frac * (phase[right] - phase[left])

    return iploc, ipmag, ipphase


def peak_detection_fast(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Detect spectral peak locations over all frames at once.

    Returns:
        Boolean mask of shape (bins - 2, frames) corresponding to x[1:-1].
    """
    center = x[1:-1]
    prev_ = x[:-2]
    next_ = x[2:]
    return (center > threshold) & (center > prev_) & (center > next_)
