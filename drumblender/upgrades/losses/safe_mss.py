"""Safe loss upgrades for robust training on wide dynamic-range percussion datasets."""
from __future__ import annotations

import math
from typing import Optional
from typing import Sequence

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeScaleInvariantMSSLoss(nn.Module):
    """
    Safe multi-term objective for wide dynamic-range percussion audio.

    Core objective:
        L = ((1 - g) * L_mss + g * L_si_mss)
            + w_amp * L_amp
            + w_env * L_env

    Optional weak physical prior in decay region:
        + w_inc  * L_inc   (only over-predicted positive band-energy jumps)
        + w_curv * L_curv  (only over-predicted curvature spikes)
        + w_floor* L_floor (prevent over-short decay tails)
        + w_onset*L_onset  (only over-predicted onset band energy)

    Length-aware safety:
        For very short samples (absolute duration), reduce prior strength to avoid
        over-regularizing transient-only clips.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        base_mss: Optional[nn.Module] = None,
        si_mss: Optional[nn.Module] = None,
        si_enabled: bool = True,
        si_rms_floor: float = 1e-3,
        gate_center_db: float = -36.0,
        gate_width_db: float = 8.0,
        amp_weight: float = 0.02,
        env_weight: float = 0.02,
        envelope_windows_ms: Sequence[float] = (20.0, 100.0),
        eps: float = 1e-8,
        detach_gate: bool = True,
        prior_enabled: bool = False,
        prior_weight_inc: float = 0.0,
        prior_weight_curv: float = 0.0,
        prior_weight_floor: float = 0.0,
        prior_weight_onset_band: float = 0.0,
        prior_decay_start_ms: float = 40.0,
        prior_onset_window_ms: float = 25.0,
        prior_n_fft: int = 1024,
        prior_hop_length: int = 256,
        prior_win_length: Optional[int] = None,
        prior_num_bands: int = 16,
        prior_eps_base: float = 1e-6,
        prior_eps_rel: float = 0.05,
        prior_curv_eps_base: float = 1e-6,
        prior_curv_eps_rel: float = 0.05,
        prior_floor_margin_db: float = 2.0,
        prior_floor_min_db: float = -72.0,
        prior_huber_delta: float = 1e-3,
        prior_length_aware: bool = True,
        prior_short_ms_start: float = 80.0,
        prior_short_ms_end: float = 260.0,
        prior_min_decay_frames: int = 8,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.base_mss = (
            base_mss if base_mss is not None else auraloss.freq.MultiResolutionSTFTLoss()
        )
        self.si_mss = (
            si_mss if si_mss is not None else auraloss.freq.MultiResolutionSTFTLoss()
        )

        self.si_enabled = bool(si_enabled)
        self.si_rms_floor = float(si_rms_floor)
        self.gate_center_db = float(gate_center_db)
        self.gate_width_db = max(float(gate_width_db), 1e-6)
        self.amp_weight = float(amp_weight)
        self.env_weight = float(env_weight)
        self.envelope_windows_ms = tuple(float(x) for x in envelope_windows_ms)
        self.eps = float(eps)
        self.detach_gate = bool(detach_gate)

        self.prior_enabled = bool(prior_enabled)
        self.prior_weight_inc = float(prior_weight_inc)
        self.prior_weight_curv = float(prior_weight_curv)
        self.prior_weight_floor = float(prior_weight_floor)
        self.prior_weight_onset_band = float(prior_weight_onset_band)
        self.prior_decay_start_ms = float(prior_decay_start_ms)
        self.prior_onset_window_ms = float(prior_onset_window_ms)
        self.prior_n_fft = int(prior_n_fft)
        self.prior_hop_length = int(prior_hop_length)
        self.prior_win_length = (
            int(prior_win_length) if prior_win_length is not None else int(prior_n_fft)
        )
        self.prior_num_bands = max(1, int(prior_num_bands))
        self.prior_eps_base = float(prior_eps_base)
        self.prior_eps_rel = float(prior_eps_rel)
        self.prior_curv_eps_base = float(prior_curv_eps_base)
        self.prior_curv_eps_rel = float(prior_curv_eps_rel)
        self.prior_floor_margin_db = float(prior_floor_margin_db)
        self.prior_floor_min_db = float(prior_floor_min_db)
        self.prior_huber_delta = max(float(prior_huber_delta), 1e-12)
        self.prior_length_aware = bool(prior_length_aware)
        self.prior_short_ms_start = float(prior_short_ms_start)
        self.prior_short_ms_end = float(prior_short_ms_end)
        self.prior_min_decay_frames = max(0, int(prior_min_decay_frames))

        win = torch.hann_window(self.prior_win_length)
        self.register_buffer("_prior_window", win, persistent=False)

    @staticmethod
    def _make_time_mask(
        batch_size: int,
        total_length: int,
        lengths: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if lengths is None:
            return torch.ones(batch_size, 1, total_length, device=device)

        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=device)
        lengths = lengths.to(device=device).long().clamp(min=1, max=total_length)
        timeline = torch.arange(total_length, device=device).unsqueeze(0)
        return (timeline < lengths.unsqueeze(1)).to(torch.float32).unsqueeze(1)

    def _masked_rms(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared_sum = (signal.square() * mask).sum(dim=(1, 2))
        valid_count = mask.sum(dim=(1, 2)).clamp_min(1.0)
        return torch.sqrt(squared_sum / valid_count + self.eps)

    def _compute_gate(self, target_rms: torch.Tensor) -> torch.Tensor:
        target_db = 20.0 * torch.log10(target_rms + self.eps)
        gate = torch.sigmoid((target_db - self.gate_center_db) / self.gate_width_db)
        if self.detach_gate:
            gate = gate.detach()
        return gate

    @staticmethod
    def _per_sample_reduce(value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 1:
            return value
        return value.reshape(value.shape[0], -1).mean(dim=1)

    def _mag_distance_per_sample(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        *,
        log_scale: bool,
        distance: str,
        log_eps: float,
        log_fac: float,
    ) -> torch.Tensor:
        if log_scale:
            pred_mag = torch.log(log_fac * pred_mag + log_eps)
            target_mag = torch.log(log_fac * target_mag + log_eps)

        if distance == "L1":
            diff = (pred_mag - target_mag).abs()
        elif distance == "L2":
            diff = (pred_mag - target_mag).pow(2)
        else:
            raise ValueError(f"Unsupported STFT magnitude distance: {distance}")

        return self._per_sample_reduce(diff)

    def _stft_loss_per_sample(
        self,
        stft_loss: nn.Module,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Mirror auraloss STFTLoss, but keep one scalar per batch item."""
        required_attrs = [
            "window",
            "stft",
            "w_sc",
            "w_log_mag",
            "w_lin_mag",
            "w_phs",
            "eps",
            "mag_distance",
            "scale_invariance",
            "perceptual_weighting",
            "scale",
        ]
        if any(not hasattr(stft_loss, attr) for attr in required_attrs):
            return None

        bs, chs, _ = pred.shape
        pred_work = pred
        target_work = target
        stft_eps = float(getattr(stft_loss, "eps", self.eps))

        if getattr(stft_loss, "perceptual_weighting", False):
            prefilter = getattr(stft_loss, "prefilter", None)
            if prefilter is None:
                return None
            pred_work = pred_work.reshape(bs * chs, 1, -1)
            target_work = target_work.reshape(bs * chs, 1, -1)
            prefilter.to(pred_work.device)
            pred_work, target_work = prefilter(pred_work, target_work)
            pred_work = pred_work.reshape(bs, chs, -1)
            target_work = target_work.reshape(bs, chs, -1)

        stft_loss.window = stft_loss.window.to(device=pred_work.device, dtype=pred_work.dtype)
        pred_mag, pred_phs = stft_loss.stft(pred_work.reshape(-1, pred_work.size(-1)))
        target_mag, target_phs = stft_loss.stft(target_work.reshape(-1, target_work.size(-1)))

        if getattr(stft_loss, "scale", None) is not None:
            if not hasattr(stft_loss, "fb"):
                return None
            stft_loss.fb = stft_loss.fb.to(device=pred_work.device, dtype=pred_mag.dtype)
            pred_mag = torch.matmul(stft_loss.fb, pred_mag)
            target_mag = torch.matmul(stft_loss.fb, target_mag)

        if getattr(stft_loss, "scale_invariance", False):
            denom = target_mag.pow(2).sum(dim=(-2, -1)).clamp_min(stft_eps)
            alpha = (pred_mag * target_mag).sum(dim=(-2, -1)) / denom
            target_mag = target_mag * alpha.unsqueeze(-1)

        loss_flat = pred_mag.new_zeros((pred_mag.shape[0],))

        if float(getattr(stft_loss, "w_sc", 0.0)) > 0.0:
            sc_num = torch.linalg.vector_norm(
                (target_mag - pred_mag).reshape(pred_mag.shape[0], -1), ord=2, dim=1
            )
            sc_den = torch.linalg.vector_norm(
                target_mag.reshape(target_mag.shape[0], -1), ord=2, dim=1
            ).clamp_min(stft_eps)
            loss_flat = loss_flat + float(stft_loss.w_sc) * (sc_num / sc_den)

        if float(getattr(stft_loss, "w_log_mag", 0.0)) > 0.0:
            log_mag = getattr(stft_loss, "logstft", None)
            if log_mag is None:
                return None
            log_loss = self._mag_distance_per_sample(
                pred_mag,
                target_mag,
                log_scale=True,
                distance=str(getattr(stft_loss, "mag_distance", "L1")),
                log_eps=float(getattr(log_mag, "log_eps", 0.0)),
                log_fac=float(getattr(log_mag, "log_fac", 1.0)),
            )
            loss_flat = loss_flat + float(stft_loss.w_log_mag) * log_loss

        if float(getattr(stft_loss, "w_lin_mag", 0.0)) > 0.0:
            lin_loss = self._mag_distance_per_sample(
                pred_mag,
                target_mag,
                log_scale=False,
                distance=str(getattr(stft_loss, "mag_distance", "L1")),
                log_eps=0.0,
                log_fac=1.0,
            )
            loss_flat = loss_flat + float(stft_loss.w_lin_mag) * lin_loss

        if bool(getattr(stft_loss, "phs_used", False)) and pred_phs is not None and target_phs is not None:
            phs_loss = self._per_sample_reduce((pred_phs - target_phs).pow(2))
            loss_flat = loss_flat + float(stft_loss.w_phs) * phs_loss

        return loss_flat.reshape(bs, chs).mean(dim=1)

    def _spectral_loss_per_sample(
        self,
        loss_module: nn.Module,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        stft_losses = getattr(loss_module, "stft_losses", None)
        if stft_losses is not None:
            per_resolution = []
            for stft_loss in stft_losses:
                loss_vec = self._stft_loss_per_sample(stft_loss, pred, target)
                if loss_vec is None:
                    return None
                per_resolution.append(loss_vec)
            return torch.stack(per_resolution, dim=0).mean(dim=0)

        return self._stft_loss_per_sample(loss_module, pred, target)

    def _log_rms_envelope(self, signal: torch.Tensor, window_ms: float) -> torch.Tensor:
        frame = max(1, int(round(self.sample_rate * window_ms / 1000.0)))
        hop = max(1, frame // 2)
        pad = frame // 2

        squared = signal.square()
        squared = F.pad(squared, (pad, pad), mode="constant", value=0.0)
        mean_square = F.avg_pool1d(
            squared,
            kernel_size=frame,
            stride=hop,
            ceil_mode=True,
        )
        env = torch.sqrt(mean_square + self.eps)
        return torch.log(env + self.eps)

    def _envelope_anchor(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.envelope_windows_ms:
            return pred.new_zeros(())

        terms = []
        for window_ms in self.envelope_windows_ms:
            pred_env = self._log_rms_envelope(pred, window_ms)
            target_env = self._log_rms_envelope(target, window_ms)
            terms.append(F.l1_loss(pred_env, target_env))

        return torch.stack(terms).mean()

    def _prior_band_energy(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.squeeze(1)
        window = self._prior_window.to(device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x,
            n_fft=self.prior_n_fft,
            hop_length=self.prior_hop_length,
            win_length=self.prior_win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)

        freq_bins = power.size(1)
        band_size = int(math.ceil(freq_bins / self.prior_num_bands))
        padded_freq_bins = band_size * self.prior_num_bands
        if padded_freq_bins != freq_bins:
            power = F.pad(power, (0, 0, 0, padded_freq_bins - freq_bins))

        power = power.view(power.size(0), self.prior_num_bands, band_size, power.size(2))
        return power.mean(dim=2)

    def _huber(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs()
        delta = self.prior_huber_delta
        quadratic = 0.5 * abs_x.square() / delta
        linear = abs_x - 0.5 * delta
        return torch.where(abs_x <= delta, quadratic, linear)

    def _masked_huber_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if value.numel() == 0:
            return value.new_zeros(())

        if mask.dtype != value.dtype:
            mask = mask.to(value.dtype)

        numer = (self._huber(value) * mask).sum()
        denom = mask.sum().clamp_min(1.0)
        return numer / denom

    def _decay_mask_from_target(self, total_true: torch.Tensor) -> torch.Tensor:
        # total_true shape: [B, T_frames]
        if total_true.size(1) < 2:
            return torch.zeros_like(total_true, dtype=torch.bool)

        diff_true = total_true[:, 1:] - total_true[:, :-1]
        onset_frame = diff_true.argmax(dim=1) + 1

        decay_offset_frames = int(
            round(self.prior_decay_start_ms * self.sample_rate / (1000.0 * self.prior_hop_length))
        )
        decay_start = onset_frame + decay_offset_frames

        t = torch.arange(total_true.size(1), device=total_true.device).unsqueeze(0)
        return t >= decay_start.unsqueeze(1)

    def _onset_mask_from_target(self, total_true: torch.Tensor) -> torch.Tensor:
        # total_true shape: [B, T_frames]
        if total_true.size(1) < 2:
            return torch.zeros_like(total_true, dtype=torch.bool)

        diff_true = total_true[:, 1:] - total_true[:, :-1]
        onset_frame = diff_true.argmax(dim=1) + 1

        onset_window_frames = int(
            round(
                self.prior_onset_window_ms
                * self.sample_rate
                / (1000.0 * self.prior_hop_length)
            )
        )
        onset_window_frames = max(0, onset_window_frames)
        onset_end = torch.clamp(
            onset_frame + onset_window_frames,
            min=0,
            max=total_true.size(1) - 1,
        )

        t = torch.arange(total_true.size(1), device=total_true.device).unsqueeze(0)
        # Include frame 0 by design to directly target first-frame wideband artifacts.
        return t <= onset_end.unsqueeze(1)

    def _prior_length_gate(
        self,
        pred: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return per-sample prior gate in [0, 1] based on absolute duration (ms)."""
        bsz = pred.size(0)
        if not self.prior_length_aware:
            return pred.new_ones((bsz,))

        if lengths is None:
            return pred.new_ones((bsz,))

        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=pred.device)
        lengths = lengths.to(device=pred.device, dtype=pred.dtype).clamp_min(1.0)

        length_ms = lengths * (1000.0 / float(self.sample_rate))

        lo = self.prior_short_ms_start
        hi = self.prior_short_ms_end
        if hi <= lo:
            return pred.new_ones((bsz,))

        gate = (length_ms - lo) / (hi - lo)
        return gate.clamp(0.0, 1.0)

    def _physical_prior(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.prior_enabled:
            return pred.new_zeros(())

        if (
            self.prior_weight_inc <= 0.0
            and self.prior_weight_curv <= 0.0
            and self.prior_weight_floor <= 0.0
            and self.prior_weight_onset_band <= 0.0
        ):
            return pred.new_zeros(())

        band_pred = self._prior_band_energy(pred)
        band_true = self._prior_band_energy(target)
        length_gate = self._prior_length_gate(pred, lengths).view(-1, 1, 1)

        total_pred = band_pred.sum(dim=1)
        total_true = band_true.sum(dim=1)
        onset_mask = self._onset_mask_from_target(total_true)
        decay_mask = self._decay_mask_from_target(total_true)
        decay_valid = (decay_mask.sum(dim=1) >= self.prior_min_decay_frames).to(
            dtype=pred.dtype
        )
        decay_valid = decay_valid.view(-1, 1, 1)

        prior_loss = pred.new_zeros(())

        # One-sided increase prior: only penalize over-predicted positive jumps.
        if self.prior_weight_inc > 0.0 and band_pred.size(-1) >= 2:
            d_pred = band_pred[:, :, 1:] - band_pred[:, :, :-1]
            d_true = band_true[:, :, 1:] - band_true[:, :, :-1]
            eps_bt = self.prior_eps_base + self.prior_eps_rel * band_true[:, :, 1:]
            excess = torch.relu(d_pred - d_true - eps_bt)
            mask = decay_mask[:, 1:].unsqueeze(1).to(excess.dtype)
            mask = mask * length_gate * decay_valid
            inc_loss = self._masked_huber_mean(excess, mask)
            prior_loss = prior_loss + self.prior_weight_inc * inc_loss

        # Curvature prior: only penalize curvature exceeding target by a margin.
        if self.prior_weight_curv > 0.0 and band_pred.size(-1) >= 3:
            dd_pred = band_pred[:, :, 2:] - 2.0 * band_pred[:, :, 1:-1] + band_pred[:, :, :-2]
            dd_true = band_true[:, :, 2:] - 2.0 * band_true[:, :, 1:-1] + band_true[:, :, :-2]
            kappa = self.prior_curv_eps_base + self.prior_curv_eps_rel * dd_true.abs()
            excess = torch.relu(dd_pred.abs() - dd_true.abs() - kappa)
            mask = decay_mask[:, 2:].unsqueeze(1).to(excess.dtype)
            mask = mask * length_gate * decay_valid
            curv_loss = self._masked_huber_mean(excess, mask)
            prior_loss = prior_loss + self.prior_weight_curv * curv_loss

        # Tail floor prior: prevent predicted tail from collapsing too quickly.
        if self.prior_weight_floor > 0.0:
            db_pred = 10.0 * torch.log10(total_pred + self.eps)
            db_true = 10.0 * torch.log10(total_true + self.eps)

            floor_target = db_true - self.prior_floor_margin_db
            excess = torch.relu(floor_target - db_pred)

            floor_gate = 0.5 + 0.5 * length_gate.squeeze(-1).squeeze(-1)
            valid_mask = (decay_mask & (db_true > self.prior_floor_min_db)).to(excess.dtype)
            valid_mask = valid_mask * floor_gate.unsqueeze(1)
            floor_loss = self._masked_huber_mean(excess, valid_mask)
            prior_loss = prior_loss + self.prior_weight_floor * floor_loss

        # Onset band-energy prior: penalize only over-predicted band energy at onset.
        if self.prior_weight_onset_band > 0.0:
            eps_bt = self.prior_eps_base + self.prior_eps_rel * band_true
            excess = torch.relu(band_pred - band_true - eps_bt)
            mask = onset_mask.unsqueeze(1).to(excess.dtype)
            mask = mask * length_gate
            onset_loss = self._masked_huber_mean(excess, mask)
            prior_loss = prior_loss + self.prior_weight_onset_band * onset_loss

        return prior_loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have identical shape, got {pred.shape} vs {target.shape}"
            )

        bsz, _, t_max = pred.shape
        mask = self._make_time_mask(bsz, t_max, lengths, pred.device)

        pred_masked = pred * mask
        target_masked = target * mask

        target_rms = self._masked_rms(target_masked, mask)

        base_loss_per_sample = self._spectral_loss_per_sample(
            self.base_mss, pred_masked, target_masked
        )
        base_loss = (
            base_loss_per_sample.mean()
            if base_loss_per_sample is not None
            else self.base_mss(pred_masked, target_masked)
        )
        if self.si_enabled:
            gate = self._compute_gate(target_rms)

            norm = target_rms.clamp_min(self.si_rms_floor).view(-1, 1, 1)
            pred_si = pred_masked / norm
            target_si = target_masked / norm
            si_loss_per_sample = self._spectral_loss_per_sample(
                self.si_mss, pred_si, target_si
            )
            if base_loss_per_sample is not None and si_loss_per_sample is not None:
                mss_loss = ((1.0 - gate) * base_loss_per_sample + gate * si_loss_per_sample).mean()
            else:
                gate_mix = gate.mean()
                si_loss = self.si_mss(pred_si, target_si)
                mss_loss = (1.0 - gate_mix) * base_loss + gate_mix * si_loss
        else:
            mss_loss = base_loss

        amp_loss = pred.new_zeros(())
        if self.amp_weight > 0.0:
            pred_rms = self._masked_rms(pred_masked, mask)
            amp_loss = F.l1_loss(
                torch.log(pred_rms + self.eps),
                torch.log(target_rms + self.eps),
            )

        env_loss = pred.new_zeros(())
        if self.env_weight > 0.0 and len(self.envelope_windows_ms) > 0:
            env_loss = self._envelope_anchor(pred_masked, target_masked)

        prior_loss = pred.new_zeros(())
        if self.prior_enabled:
            prior_loss = self._physical_prior(pred_masked, target_masked, lengths=lengths)

        return mss_loss + self.amp_weight * amp_loss + self.env_weight * env_loss + prior_loss
