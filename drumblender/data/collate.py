import torch
from torch.nn.utils.rnn import pad_sequence

def pad_audio_params_collate(batch):
    """
    Supports items of:
      - (audio, params)
      - (audio, params, length)
    where audio is Tensor [..., T] (typically [1, T] or [T]),
    params is Tensor or dict-like, and length is int.
    """

    # 2-tuple vs 3-tuple 紐⑤몢 泥섎━
    if len(batch[0]) == 3:
        audios, params, lengths = zip(*batch)
        lengths = torch.tensor(lengths, dtype=torch.long)
    elif len(batch[0]) == 2:
        audios, params = zip(*batch)
        lengths = torch.tensor([a.shape[-1] for a in audios], dtype=torch.long)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    # audio瑜?[T] ?뺥깭濡?留욎텣 ??pad
    audios_1d = []
    for a in audios:
        if a.dim() == 2 and a.shape[0] == 1:      # [1, T] -> [T]
            audios_1d.append(a.squeeze(0))
        elif a.dim() == 1:                        # [T]
            audios_1d.append(a)
        else:
            # ?덉긽 諛??뺥깭硫?留덉?留?李⑥썝??time?쇰줈 蹂닿퀬 flatten? ?섏? ?딆쓬
            # (?꾩슂?섎㈃ ?ш린?????꾧꺽??assert 嫄몄뼱????
            audios_1d.append(a.reshape(-1))

    padded = pad_sequence(audios_1d, batch_first=True)  # [B, Tmax]
    padded = padded.unsqueeze(1)                        # [B, 1, Tmax] 濡?蹂듦뎄

    # params ?뺢퇋?? dict / Tensor / scalar / None 紐⑤몢 吏??
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
        # params ?먯꽌媛 湲몄씠媛 ?ㅻ? ???덈뒗 寃쎌슦(?? [3,64,F]?먯꽌 F媛 ?ㅻ쫫) ?⑤뵫
        p0 = params[0]
        if p0.dim() == 3:
            # [P, M, F] -> F 湲곗??쇰줈 pad?댁꽌 [B, P, M, Fmax] (Batch size, # of Param, # of Modal, max Frame len)
            # pad_sequence??[F] 1D留???諛쏆쑝?? (P*M, F)濡??댁꽌 pad ??蹂듭썝
            flats = []
            PM = p0.shape[0] * p0.shape[1]
            for p in params:
                flats.append(p.reshape(PM, p.shape[-1]).transpose(0, 1))  # [F, PM]
            # ?댁젣 媛??먯냼媛 [F_i, PM]?대?濡?pad_sequence濡?[B, Fmax, PM]
            padded_flat = pad_sequence(flats, batch_first=True)           # [B, Fmax, PM]
            padded_flat = padded_flat.transpose(1, 2)                     # [B, PM, Fmax]
            params_out = padded_flat.reshape(len(params), p0.shape[0], p0.shape[1], -1)  # [B,P,M,Fmax]
        else:
            params_out = torch.stack(list(params), dim=0)



    return padded, params_out, lengths
