"""
Methods for performing modal analysis on a CQT spectrogram
Replaces the baseline nearest-neighbor tracking with a forward-time
HPSS + Kalman Tracking + Confidence pipeline.
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

try:
    import nnAudio.features as features
except ImportError:
    features = None


class KalmanTrack:
    """State tracker for modal components."""
    MAX_MISS = 3

    def __init__(self, frame: int, bin_idx: int, k: float, k_v: float) -> None:
        self.bin = float(bin_idx)
        self.velocity = 0.0
        self.history: List[Tuple[int, int]] = [(frame, bin_idx)]
        self.alive_since = frame
        self.last_frame = frame
        self.miss_count = 0
        self.active = True

        # Tracking gains dynamically set from init
        self.K = k
        self.K_V = k_v

    @property
    def streak(self) -> int:
        return self.last_frame - self.alive_since + 1

    @property
    def tolerance(self) -> float:
        if self.streak < 2:
            return 12.0
        return max(2.0, abs(self.velocity) * 2.0)

    def predict(self) -> float:
        return self.bin + self.velocity

    def update(self, frame: int, obs_bin: int) -> None:
        pred = self.predict()
        inn = obs_bin - pred
        self.bin = pred + self.K * inn
        self.velocity = self.velocity + self.K_V * inn
        self.miss_count = 0
        self.history.append((frame, obs_bin))
        self.last_frame = frame

    def advance_miss(self, frame: int) -> None:
        self.bin = self.predict()
        self.last_frame = frame
        self.miss_count += 1
        if self.miss_count > self.MAX_MISS:
            self.active = False


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
        # --- New Pipeline Parameters ---
        mask_threshold: float = 0.8,         # HPSS Gate
        kalman_k: float = 0.4,               # Alpha (Position Gain)
        kalman_kv: float = 0.4,              # Beta (Velocity Gain)
        confidence_threshold: float = 0.5,   # Final strictness gate
        h_kernel: int = 31,                  # Horizontal median filter
        p_kernel: int = 31,                  # Vertical median filter
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

        # Store Pipeline Params
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.kalman_k = kalman_k
        self.kalman_kv = kalman_kv
        self.confidence_threshold = confidence_threshold
        self.h_kernel = h_kernel
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

        batch_freqs, batch_amps, batch_phases = [], [], []

        for i in range(x.shape[0]):
            freqs, amps, phases = self.modal_tracking(x[i])
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

        return torch.stack(batch_freqs, dim=0), torch.stack(batch_amps, dim=0), torch.stack(batch_phases, dim=0)

    def spectrogram(self, audio: torch.Tensor, complex: bool = True) -> torch.Tensor:
        x = self.cqt(audio, normalization_type="wrap")
        if not complex:
            x = torch.sqrt(torch.sum(torch.square(x), dim=-1))
        return x

    def modal_tracking(
        self, spec: np.ndarray
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        Replaces nearest-neighbor tracking with HPSS + Kalman + Confidence.
        """
        assert spec.ndim == 3 and spec.shape[-1] == 2

        # 1. Extract Arrays
        X_complex = spec[..., 0] + 1j * spec[..., 1]
        X_mag = np.abs(X_complex)
        X_db = 20.0 * np.log10(X_mag + 1e-8)
        X_phase = np.angle(X_complex)
        n_bins, n_frames = X_mag.shape

        # 2. Stage 1: HPSS Mask
        H = median_filter(X_mag, size=(1, self.h_kernel))
        P = median_filter(X_mag, size=(self.p_kernel, 1))
        tonal_mask = H / (H + P + 1e-8)

        # 3. Peak Detection & HPSS Gating
        peaks_per_frame = []
        masked_set = set()

        for t in range(n_frames):
            peaks, _ = find_peaks(X_db[:, t], height=self.threshold)
            peaks_per_frame.append(peaks)
            for b in peaks:
                if tonal_mask[b, t] >= self.mask_threshold:
                    masked_set.add((t, int(b)))

        # 4. Stage 2: Kalman Tracking (FORWARD in time)
        active: List[KalmanTrack] = []
        finished: List[KalmanTrack] = []

        for t, peaks in enumerate(peaks_per_frame):
            frame_peaks = {int(p) for p in peaks if (t, int(p)) in masked_set}
            unmatched = set(frame_peaks)
            matched_ids = set()

            for track in sorted(active, key=lambda tr: -tr.streak):
                pred = track.predict()
                tol = track.tolerance
                candidates = [(abs(p - pred), p) for p in unmatched if abs(p - pred) <= tol]
                if candidates:
                    _, best = min(candidates)
                    track.update(t, best)
                    unmatched.discard(best)
                    matched_ids.add(id(track))

            for track in active:
                if id(track) not in matched_ids:
                    track.advance_miss(t)

            dead = [tr for tr in active if not tr.active]
            active = [tr for tr in active if tr.active]
            finished.extend(dead)
            for p in unmatched:
                active.append(KalmanTrack(t, p, self.kalman_k, self.kalman_kv))

        finished.extend(active)

        # 5. Stage 3: Confidence
        stability = {}
        for tr in finished:
            stab = 0.0 if tr.streak < 3 else min(1.0, tr.streak / 10.0)
            for (fr, ob) in tr.history:
                stability[(fr, ob)] = stab

        prominence = {}
        for t, peaks in enumerate(peaks_per_frame):
            if not len(peaks):
                continue
            frame_db = X_db[:, t]
            raw = np.empty(len(peaks))
            for i, b in enumerate(peaks):
                w = 18 if b < 60 else (12 if b < 120 else 6)
                lo, hi = max(0, b - w), min(n_bins, b + w + 1)
                nb = np.concatenate([frame_db[lo:b], frame_db[b + 1:hi]])
                raw[i] = frame_db[b] / (nb.mean() + 1e-8) if len(nb) else 0.0

            lo_, hi_ = raw.min(), raw.max()
            norm = (raw - lo_) / (hi_ - lo_) if hi_ > lo_ else np.zeros_like(raw)
            for b, s in zip(peaks, norm):
                prominence[(t, int(b))] = float(s)

        # 6. Construct Contiguous Modes for Additive Synthesis
        freqs, amps, phases = [], [], []

        for track in finished:
            if track.streak < self.min_length:
                continue

            # Check if track meets confidence overall to be preserved
            track_confs = [
                tonal_mask[b, t] * prominence.get((t, b), 0.0) * stability.get((t, b), 0.0)
                for t, b in track.history
            ]

            if not track_confs or max(track_confs) < self.confidence_threshold:
                continue

            # Build full length track arrays
            t_freqs = [0.0] * n_frames
            t_amps = [0.0] * n_frames
            t_phases = [0.0] * n_frames

            for i, (t, b) in enumerate(track.history):
                conf = track_confs[i]
                if conf >= self.confidence_threshold:
                    # Sub-bin interpolation (skip edges to prevent out-of-bounds)
                    if 0 < b < n_bins - 1:
                        ip_loc, ip_mag, ip_phase = peak_interpolation(X_mag[:, t], X_phase[:, t], np.array([b]))
                        t_freqs[t] = float(ip_loc[0])
                        t_amps[t] = float(ip_mag[0])
                        t_phases[t] = float(ip_phase[0])
                    else:
                        t_freqs[t] = float(b)
                        t_amps[t] = float(X_mag[b, t])
                        t_phases[t] = float(X_phase[b, t])
                else:
                    t_freqs[t] = float(b)
                    t_amps[t] = 0.0
                    t_phases[t] = 0.0

            # Forward/Backward fill frequencies so dead periods don't jump to 0 Hz
            active_idx = [i for i, a in enumerate(t_amps) if a > 0]
            if active_idx:
                first, last = active_idx[0], active_idx[-1]
                for i in range(first):
                    t_freqs[i] = t_freqs[first]
                for i in range(last + 1, n_frames):
                    t_freqs[i] = t_freqs[last]

            freqs.append(t_freqs)
            amps.append(t_amps)
            phases.append(t_phases)

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
        freq_env, amp_env, phase_env = [], [], []

        for i in range(num_modes):
            freq_env.append(torch.zeros(num_hops))
            amp_env.append(torch.zeros(num_hops))
            phase_env.append(torch.zeros(num_hops))

            # Tracks are processed chronologically forward
            for h in range(num_hops):
                if h < len(freqs[i]):
                    freq_env[-1][h] = freqs[i][h]
                    amp_env[-1][h] = amps[i][h]
                    phase_env[-1][h] = phases[i][h]
                else:
                    break

        # Handle empty tracks gracefully if no modes survived
        if not freq_env:
            return torch.zeros((0, num_hops)), torch.zeros((0, num_hops)), torch.zeros((0, num_hops))

        return torch.stack(freq_env), torch.stack(amp_env), torch.stack(phase_env)


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
