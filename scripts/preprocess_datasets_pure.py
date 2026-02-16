# scripts/preprocess_datasets_pure.py
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from drumblender.utils.audio import preprocess_audio_file


@dataclass
class Config:
    raw_root: Path = Path("/public/datasets/datasets_pure")
    processed_root: Path = Path("/private/datasets/processed")
    rejected_root: Path = Path("/private/datasets/rejected")
    logs_root: Path = Path("/private/datasets/logs")

    sample_rate: int = 48000
    num_samples: Optional[int] = None  # ✅ 가변 길이 저장

    # ✅ silent_all 판정(“진짜 무음 파일만” 걸러내기): 더 낮게
    filter_silent_all: bool = True
    silent_all_threshold_db: float = -75.0

    # ✅ 시작 무음 컷은 기존 방식대로(-60 유지)
    remove_start_silence: bool = True
    start_silence_threshold_db: float = -60.0

    # ✅ tail cut은 기본 OFF (필요하면 True로)
    remove_end_silence: bool = True
    tail_silence_threshold_db: float = -60.0
    tail_peak_ratio: float = 0.02
    min_tail_silence_ms: float = 50.0
    tail_fade_out_ms: float = 5.0

    frame_size: int = 256
    hop_size: int = 256

    # ✅ 길이 제한(14초 초과 reject)
    max_duration_sec: float = 14.0

    copy_rejected: bool = True


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def classify_reason(err: Exception) -> str:
    msg = str(err).lower()

    if "too_long" in msg:
        return "too_long"

    # audio.py에서 silent_all을 이렇게 던짐:
    #   ValueError(f"silent_all: below {silent_all_threshold_db}dB")
    if "silent_all" in msg:
        return "silent_all"

    # start cut 쪽에서 전체가 threshold 아래면 이런 메시지로도 나올 수 있음
    if "entire wavfile below threshold" in msg:
        return "silent_all"

    if "near) zero" in msg or "near zero" in msg:
        return "zero"

    if "sox" in msg or "ffmpeg" in msg:
        return "decode_error"

    return "error"


def write_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    cfg = Config()

    mkdir(cfg.processed_root)
    mkdir(cfg.rejected_root)
    mkdir(cfg.logs_root)

    ok_log = cfg.logs_root / "manifest_ok.jsonl"
    bad_log = cfg.logs_root / "manifest_bad.jsonl"

    # ✅ .wav / .WAV 모두 포함 (대소문자 무시)
    wavs = [p for p in cfg.raw_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    wavs.sort()
    print(f"[scan] {cfg.raw_root} -> {len(wavs)} wavs")

    ok = 0
    bad = 0

    pbar = tqdm(wavs, desc="preprocess", unit="file", dynamic_ncols=True)
    for in_path in pbar:
        rel = in_path.relative_to(cfg.raw_root)
        out_path = cfg.processed_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            preprocess_audio_file(
                input_file=in_path,
                output_file=out_path,
                sample_rate=cfg.sample_rate,
                num_samples=cfg.num_samples,
                mono=True,  # 채널0 고정

                # ✅ 논의 반영: silent_all 필터 / start 컷 threshold 분리
                filter_silent_all=cfg.filter_silent_all,
                silent_all_threshold_db=cfg.silent_all_threshold_db,

                remove_start_silence=cfg.remove_start_silence,
                start_silence_threshold_db=cfg.start_silence_threshold_db,

                # ✅ tail cut 옵션
                remove_end_silence=cfg.remove_end_silence,
                tail_silence_threshold_db=cfg.tail_silence_threshold_db,
                tail_peak_ratio=cfg.tail_peak_ratio,
                min_tail_silence_ms=cfg.min_tail_silence_ms,
                tail_fade_out_ms=cfg.tail_fade_out_ms,

                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,

                # ✅ 14초 제한
                max_duration_sec=cfg.max_duration_sec,
            )

            ok += 1
            write_jsonl(
                ok_log,
                {
                    "ts": datetime.utcnow().isoformat(),
                    "status": "ok",
                    "input": str(in_path),
                    "output": str(out_path),
                },
            )

        except Exception as e:
            bad += 1
            reason = classify_reason(e)

            rej_path = cfg.rejected_root / reason / rel
            rej_path.parent.mkdir(parents=True, exist_ok=True)

            if cfg.copy_rejected:
                try:
                    shutil.copy2(in_path, rej_path)
                except Exception:
                    pass

            # processed에 부분 생성된 파일 있으면 제거(최종본 정책)
            if out_path.exists():
                try:
                    out_path.unlink()
                except Exception:
                    pass

            write_jsonl(
                bad_log,
                {
                    "ts": datetime.utcnow().isoformat(),
                    "status": "bad",
                    "reason": reason,
                    "input": str(in_path),
                    "error": str(e),
                },
            )

        pbar.set_postfix(ok=ok, bad=bad)

    print("[done]")
    print(f"  ok : {ok_log}")
    print(f"  bad: {bad_log}")
    print(f"  processed root: {cfg.processed_root}")
    print(f"  rejected  root: {cfg.rejected_root}")


if __name__ == "__main__":
    main()
