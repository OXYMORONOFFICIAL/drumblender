"""Safe loss upgrades for robust training on wide dynamic-range percussion datasets."""
from __future__ import annotations

from typing import Optional
from typing import Sequence

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeScaleInvariantMSSLoss(nn.Module):
    """
    Safe multi-term objective for wide dynamic-range percussion audio.

    This module keeps baseline MSS behavior but adds two stabilizers:
    1) RMS-gated scale-invariant MSS branch
    2) Weak amplitude and multi-scale envelope anchors

    Final objective:
        L = ((1 - g) * L_mss + g * L_si_mss)
            + w_amp * L_amp
            + w_env * L_env

    where g is derived from target RMS (higher RMS -> stronger SI contribution).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        base_mss: Optional[nn.Module] = None,
        si_mss: Optional[nn.Module] = None,
        si_rms_floor: float = 1e-3,
        gate_center_db: float = -36.0,
        gate_width_db: float = 8.0,
        amp_weight: float = 0.02,
        env_weight: float = 0.02,
        envelope_windows_ms: Sequence[float] = (20.0, 100.0),
        eps: float = 1e-8,
        detach_gate: bool = True,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.base_mss = (
            base_mss if base_mss is not None else auraloss.freq.MultiResolutionSTFTLoss()
        )
        self.si_mss = (
            si_mss if si_mss is not None else auraloss.freq.MultiResolutionSTFTLoss()
        )

        self.si_rms_floor = float(si_rms_floor)
        self.gate_center_db = float(gate_center_db)
        self.gate_width_db = max(float(gate_width_db), 1e-6)
        self.amp_weight = float(amp_weight)
        self.env_weight = float(env_weight)
        self.envelope_windows_ms = tuple(float(x) for x in envelope_windows_ms)
        self.eps = float(eps)
        self.detach_gate = bool(detach_gate)

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
        pred_rms = self._masked_rms(pred_masked, mask)

        gate = self._compute_gate(target_rms)
        gate_mix = gate.mean()

        norm = target_rms.clamp_min(self.si_rms_floor).view(-1, 1, 1)
        pred_si = pred_masked / norm
        target_si = target_masked / norm

        base_loss = self.base_mss(pred_masked, target_masked)
        si_loss = self.si_mss(pred_si, target_si)
        mss_loss = (1.0 - gate_mix) * base_loss + gate_mix * si_loss

        amp_loss = F.l1_loss(
            torch.log(pred_rms + self.eps),
            torch.log(target_rms + self.eps),
        )
        env_loss = self._envelope_anchor(pred_masked, target_masked)

        return mss_loss + self.amp_weight * amp_loss + self.env_weight * env_loss
