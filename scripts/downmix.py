# scripts/mono_downmix_to_new_folder.py
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# =========================
# ✅ 여기만 바꾸면 됨
INPUT_DIR = "/public/datasets/datasets_pure/splice"
# =========================


def list_wavs(root: Path) -> list[Path]:
    wavs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort()
    return wavs


def main():
    src_root = Path(INPUT_DIR)
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    # output root: "<input>_mono" (same parent)
    out_root = src_root.with_name(src_root.name + "_mono")
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(src_root)
    print(f"[scan] {src_root} -> {len(wavs)} wavs")
    print(f"[out ] {out_root}")

    changed = 0
    copied = 0
    failed = 0

    pbar = tqdm(wavs, desc="mono_downmix", unit="file", dynamic_ncols=True)
    for in_path in pbar:
        rel = in_path.relative_to(src_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr = torchaudio.load(in_path)  # [C,T]
            if wav.ndim != 2:
                raise ValueError(f"Unexpected wav shape: {tuple(wav.shape)}")

            if wav.shape[0] > 1:
                # ✅ 요구사항: mean downmix가 아니라 channel 0만 사용
                wav = wav[:1, :]
                changed += 1
            else:
                # mono면 그대로 저장(복사와 동일 효과)
                copied += 1

            # 안전 저장: tmp로 쓰고 rename (부분 파일 방지)
            tmp = out_path.with_suffix(".tmp.wav")
            torchaudio.save(tmp, wav, sr)
            tmp.replace(out_path)

        except Exception as e:
            failed += 1
            # 실패 파일이 생기면 지움
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass

        pbar.set_postfix(changed=changed, copied=copied, failed=failed)

    print("[done]")
    print(" changed(stereo->mono ch0):", changed)
    print(" copied(already mono):    ", copied)
    print(" failed:                 ", failed)
    print(" output root:            ", out_root)


if __name__ == "__main__":
    main()
