import torch
from torch.nn.utils.rnn import pad_sequence

def pad_audio_params_collate(batch, treat_scalar_params_as_none: bool = True):
    """
    Supports items of:
      - (audio, params)
      - (audio, params, length)
    where audio is Tensor [..., T] (typically [1, T] or [T]),
    params is Tensor or dict-like, and length is int.
    """

    # 2-tuple vs 3-tuple 모두 처리
    if len(batch[0]) == 3:
        audios, params, lengths = zip(*batch)
        lengths = torch.tensor(lengths, dtype=torch.long)
    elif len(batch[0]) == 2:
        audios, params = zip(*batch)
        lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    # audio를 [T] 형태로 맞춘 뒤 pad
    audios_1d = []
    for a in audios:
        if a.dim() == 2 and a.shape[0] == 1:      # [1, T] -> [T]
            audios_1d.append(a.squeeze(0))
        elif a.dim() == 1:                        # [T]
            audios_1d.append(a)
        else:
            # 예상 밖 형태면 마지막 차원을 time으로 보고 flatten은 하지 않음
            # (필요하면 여기서 더 엄격히 assert 걸어도 됨)
            audios_1d.append(a.reshape(-1))

    padded = pad_sequence(audios_1d, batch_first=True)  # [B, Tmax]
    padded = padded.unsqueeze(1)                        # [B, 1, Tmax] 로 복구

    # params 정규화: dict / Tensor / scalar / None 모두 지원
    if params[0] is None:
        params_out = None

    elif isinstance(params[0], dict):
        out = {}
        for k in params[0].keys():
            vals = [p[k] for p in params]
            if torch.is_tensor(vals[0]):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = torch.tensor(vals)
        params_out = out

    elif torch.is_tensor(params[0]):
        # params 텐서가 길이가 다를 수 있는 경우(예: [3,64,F]에서 F가 다름) 패딩
        p0 = params[0]
        if p0.dim() == 3:
            # [P, M, F] -> F 기준으로 pad해서 [B, P, M, Fmax]
            # pad_sequence는 [F] 1D만 잘 받으니, (P*M, F)로 펴서 pad 후 복원
            flats = []
            PM = p0.shape[0] * p0.shape[1]
            for p in params:
                flats.append(p.reshape(PM, p.shape[-1]).transpose(0, 1))  # [F, PM]
            # 이제 각 원소가 [F_i, PM]이므로 pad_sequence로 [B, Fmax, PM]
            padded_flat = pad_sequence(flats, batch_first=True)           # [B, Fmax, PM]
            padded_flat = padded_flat.transpose(1, 2)                     # [B, PM, Fmax]
            params_out = padded_flat.reshape(len(params), p0.shape[0], p0.shape[1], -1)  # [B,P,M,Fmax]
        else:
            params_out = torch.stack(list(params), dim=0)



    return padded, params_out, lengths
