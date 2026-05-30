"""
Methods for performing modal analysis on a CQT spectrogram
Based on Sinusoidal Modeling Synthesis (SMS) Toolbox by Xavier Serra and Julius Smith
Uses the nnAudio CQT implementation for better resolution with low frequncies
"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

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
        min_length: int = 4,
        num_modes: Optional[int] = None,
        threshold: float = -80.0,
        diff_threshold: float = 2.5,
        max_gap: int = 2,
        min_active_frames: Optional[int] = None,
        min_streak: Optional[int] = None,
        min_active_ratio: float = 0.25,
        min_track_energy: float = 0.0,
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
            max_gap: Number of inactive frames a track can span before closing
            min_active_frames: Minimum matched-peak frames required for a track.
                Defaults to min_length.
            min_streak: Minimum consecutive matched-peak frames required for a track.
                Defaults to min_active_frames.
            min_active_ratio: Minimum ratio of matched-peak frames to tracked frames
            min_track_energy: Minimum integrated amplitude over matched-peak frames
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
        self.max_gap = max_gap
        self.min_active_frames = min_active_frames
        self.min_streak = min_streak
        self.min_active_ratio = min_active_ratio
        self.min_track_energy = min_track_energy

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
            freqs, amps, phases, active, start_frames = self.modal_tracking(
                x[i], threshold=self.threshold, return_metadata=True
            )
            freqs, amps, phases = self.create_modal_tensors(
                freqs,
                amps,
                phases,
                num_hops=num_hops,
                min_length=self.min_length,
                active=active,
                start_frames=start_frames,
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
        self,
        spec: np.ndarray,
        threshold: float = -80.0,
        return_metadata: bool = False,
    ) -> Union[
        Tuple[List[List[float]], List[List[float]], List[List[float]]],
        Tuple[
            List[List[float]],
            List[List[float]],
            List[List[float]],
            List[List[bool]],
            List[int],
        ],
    ]:
        """
        Performs modal tracking on a CQT spectrogram -- i.e. finds sinusoidal tracks
        and attempts to continue them across frames.

        Args:
            spec: A complex CQT spectrogram of shape (bins, frames, 2)
            threshold: Threshold for peak detection in dB

        Returns:
            Tuple of (frequencies, amplitudes) of shape (modes, frames)
        """

        assert spec.ndim == 3, "Spectrogram must a complex spectrogram"
        assert spec.shape[-1] == 2, "Spectrogram must be complex"

        # Reverse the spectrogram along the temporal axis
        # Assuming percussive audio here and that the most
        # important modes will have the longest decay
        spec = np.flip(spec, axis=1)

        # Initialize lists to store each mode's values and active-state metadata.
        freqs = []
        amps = []
        phases = []
        active = []
        start_frames = []
        gap_counts = []
        open_tracks = []

        # Select modes from the spectrogram
        for i in range(spec.shape[1]):
            # Get the magnitude, dB, and phase of the current frame
            frame = spec[:, i]
            assert frame.shape == (self.n_bins, 2)

            X_mag = np.sqrt(np.sum(np.square(frame), axis=-1))
            X_db = 20.0 * np.log10(X_mag + 1e-8)
            X_phase = np.arctan2(frame[:, 1], frame[:, 0])

            # Find peaks in the current frame
            peaks = peak_detection(X_db, threshold)
            if len(peaks) == 0:
                open_tracks = self._append_inactive_frame(
                    freqs, amps, phases, active, gap_counts, open_tracks
                )
                continue

            peaks_loc, peaks_mag, peaks_phase = peak_interpolation(
                X_mag, X_phase, peaks
            )

            # Try to continue the mode from the previous frame
            next_open_tracks = []
            for j in open_tracks:
                if len(peaks_loc) > 0:
                    # Find difference between previous peak and current peaks
                    prev_freq = freqs[j][-1]
                    peak_diff = np.abs(peaks_loc - prev_freq)

                    # If the difference is less than 2.5% of the previous frequency,
                    # then we assume that the peak is the same mode
                    if np.min(peak_diff) < prev_freq * self.diff_threshold:
                        closest_peak = np.argmin(peak_diff)
                        freqs[j].append(peaks_loc[closest_peak])
                        amps[j].append(peaks_mag[closest_peak])
                        phases[j].append(peaks_phase[closest_peak])
                        active[j].append(True)
                        gap_counts[j] = 0
                        next_open_tracks.append(j)

                        # Remove the peak from the list
                        peaks_loc = np.delete(peaks_loc, closest_peak)
                        peaks_mag = np.delete(peaks_mag, closest_peak)
                        peaks_phase = np.delete(peaks_phase, closest_peak)
                    else:
                        # No good matching peaks, just copy the last peak, but
                        # with an amplitude of 0.
                        self._append_inactive_track(
                            freqs,
                            amps,
                            phases,
                            active,
                            gap_counts,
                            next_open_tracks,
                            j,
                        )
                else:
                    # If there are no more peaks, just copy the last peak
                    self._append_inactive_track(
                        freqs,
                        amps,
                        phases,
                        active,
                        gap_counts,
                        next_open_tracks,
                        j,
                    )
            open_tracks = next_open_tracks

            # Add any remaining peaks as new modes
            for peak_loc, peak_mag, peak_phase in zip(peaks_loc, peaks_mag, peaks_phase):
                freqs.append([peak_loc])
                amps.append([peak_mag])
                phases.append([peak_phase])
                active.append([True])
                start_frames.append(i)
                gap_counts.append(0)
                open_tracks.append(len(freqs) - 1)

        if return_metadata:
            return freqs, amps, phases, active, start_frames
        return freqs, amps, phases

    def create_modal_tensors(
        self,
        freqs: List[List[float]],
        amps: List[List[float]],
        phases: List[List[float]],
        num_hops: int,
        min_length: int,
        active: Optional[List[List[bool]]] = None,
        start_frames: Optional[List[int]] = None,
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
            min_length: Minimum active-frame count of a mode to be included
            active: Optional active-state mask for each track frame
            start_frames: Optional start frame for each track in reversed time

        Returns:
            Tuple of (frequencies, amplitudes, phases) of shape (modes, num_hops)
        """

        freq_env = []
        amp_env = []
        phase_env = []

        for i in range(len(freqs)):
            active_i = active[i] if active is not None else [a > 0.0 for a in amps[i]]
            if not self._valid_track(amps[i], active_i, min_length):
                continue

            freq_env.append(torch.zeros(num_hops))
            amp_env.append(torch.zeros(num_hops))
            phase_env.append(torch.zeros(num_hops))

            if start_frames is None:
                freqs_i = freqs[i][::-1]
                amps_i = amps[i][::-1]
                phases_i = phases[i][::-1]
                track_len = min(num_hops, len(freqs_i))
                for h in range(track_len):
                    freq_env[-1][h] = freqs_i[h]
                    amp_env[-1][h] = amps_i[h]
                    phase_env[-1][h] = phases_i[h]
            else:
                for k, (freq, amp, phase) in enumerate(
                    zip(freqs[i], amps[i], phases[i])
                ):
                    frame = start_frames[i] + k
                    if frame >= num_hops:
                        break
                    h = num_hops - 1 - frame
                    if h < 0:
                        break
                    freq_env[-1][h] = freq
                    amp_env[-1][h] = amp
                    phase_env[-1][h] = phase

        if not freq_env:
            shape = (0, num_hops)
            return torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)

        return torch.stack(freq_env), torch.stack(amp_env), torch.stack(phase_env)

    def _append_inactive_frame(
        self,
        freqs: List[List[float]],
        amps: List[List[float]],
        phases: List[List[float]],
        active: List[List[bool]],
        gap_counts: List[int],
        open_tracks: List[int],
    ) -> List[int]:
        next_open_tracks = []
        for j in open_tracks:
            self._append_inactive_track(
                freqs, amps, phases, active, gap_counts, next_open_tracks, j
            )
        return next_open_tracks

    def _append_inactive_track(
        self,
        freqs: List[List[float]],
        amps: List[List[float]],
        phases: List[List[float]],
        active: List[List[bool]],
        gap_counts: List[int],
        next_open_tracks: List[int],
        track_idx: int,
    ) -> None:
        gap_counts[track_idx] += 1
        if gap_counts[track_idx] <= self.max_gap:
            freqs[track_idx].append(freqs[track_idx][-1])
            amps[track_idx].append(0.0)
            phases[track_idx].append(phases[track_idx][-1])
            active[track_idx].append(False)
            next_open_tracks.append(track_idx)

    def _valid_track(
        self,
        amps: List[float],
        active: List[bool],
        min_length: int,
    ) -> bool:
        if len(active) == 0:
            return False

        total_active = sum(active)
        min_active_frames = (
            min_length if self.min_active_frames is None else self.min_active_frames
        )
        min_streak = min_active_frames if self.min_streak is None else self.min_streak
        active_energy = sum(amp for amp, is_active in zip(amps, active) if is_active)
        active_ratio = total_active / len(active)

        return (
            total_active >= min_active_frames
            and max_consecutive_true(active) >= min_streak
            and active_ratio >= self.min_active_ratio
            and active_energy >= self.min_track_energy
        )

    def frequencies(self) -> np.ndarray:
        """
        Returns the frequencies of the CQT bins
        """
        return self.cqt.frequencies


# TODO: Maybe switch this out for scipy.signal.find_peaks?
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


def max_consecutive_true(values: List[bool]) -> int:
    max_streak = 0
    current = 0
    for value in values:
        if value:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


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
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)
    ipphase = np.interp(iploc, np.arange(0, phase.size), phase)

    return iploc, ipmag, ipphase
