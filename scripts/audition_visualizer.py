#!/usr/bin/env python3
"""
Interactive HTML audition/visualization server for reconstruction bundles.

Features:
- Randomly shows N samples (default 10) from a result folder
- Plays target/reconstruction audio
- Displays high-resolution waveform and spectrogram for target/reconstruction
- Shows per-sample metrics (from per_file_loss.csv + optional per_file_metrics.csv)
- Refresh button to load a new random set

Expected folder layout (from export_recon_wavs.py):
  <result_dir>/
    manifest.csv
    recon/<source_filename>.wav
    target/<source_filename>.wav                # optional
    evaluation/per_file_loss.csv
    evaluation/per_file_metrics.csv             # optional
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import random
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib

# ### HIGHLIGHT: Use non-interactive backend for server-side image rendering.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None

try:
    from scipy.io import wavfile as scipy_wavfile  # type: ignore
except Exception:
    scipy_wavfile = None


@dataclass
class SampleRow:
    index: int
    meta_key: str
    source_filename: str
    length: int
    recon_path: Path
    target_path: Optional[Path]
    metrics: Dict[str, float]


def _to_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def _safe_rel(path_str: str) -> str:
    return str(Path(path_str.replace("\\", "/")))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _merge_metrics(
    loss_rows: Optional[List[Dict[str, str]]],
    metric_rows: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Merge per-file loss/metric tables.
    Priority:
    1) key-based merge using common identity column
    2) row-order merge fallback
    """
    if not loss_rows and not metric_rows:
        return []

    if loss_rows is None:
        return metric_rows or []
    if metric_rows is None:
        return loss_rows

    key_candidates = ("index", "meta_key", "source_filename")
    chosen = None
    if len(loss_rows) > 0 and len(metric_rows) > 0:
        for k in key_candidates:
            if k in loss_rows[0] and k in metric_rows[0]:
                chosen = k
                break

    if chosen is not None:
        by_key: Dict[str, Dict[str, str]] = {}
        for row in loss_rows:
            kv = str(row.get(chosen, "")).strip()
            if kv:
                by_key[kv] = row
        out: List[Dict[str, str]] = []
        for row in metric_rows:
            kv = str(row.get(chosen, "")).strip()
            merged = {}
            if kv and kv in by_key:
                merged.update(by_key[kv])
            merged.update(row)
            out.append(merged)
        return out

    n = min(len(loss_rows), len(metric_rows))
    out = []
    for i in range(n):
        merged = {}
        merged.update(loss_rows[i])
        merged.update(metric_rows[i])
        out.append(merged)
    return out


def _collect_rows(result_dir: Path) -> List[SampleRow]:
    manifest = result_dir / "manifest.csv"
    eval_dir = result_dir / "evaluation"
    loss_csv = eval_dir / "per_file_loss.csv"
    metrics_csv = eval_dir / "per_file_metrics.csv"

    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.csv not found: {manifest}")

    manifest_rows = _read_csv(manifest)
    loss_rows = _read_csv(loss_csv) if loss_csv.is_file() else None
    metric_rows = _read_csv(metrics_csv) if metrics_csv.is_file() else None
    merged_metric_rows = _merge_metrics(loss_rows, metric_rows)

    # ### HIGHLIGHT: Build fast lookup for metric rows.
    metric_by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in merged_metric_rows:
        k = (
            str(r.get("index", "")),
            str(r.get("meta_key", "")),
            _safe_rel(str(r.get("source_filename", ""))),
        )
        metric_by_key[k] = r

    rows: List[SampleRow] = []
    for m in manifest_rows:
        idx_str = str(m.get("index", "")).strip()
        meta_key = str(m.get("meta_key", "")).strip()
        src = _safe_rel(str(m.get("source_filename", "")).strip())
        length_i = int(_to_float(m.get("length", 0)))

        recon_path_raw = str(m.get("recon_path", "")).strip()
        target_path_raw = str(m.get("target_path", "")).strip()

        recon_path = Path(recon_path_raw)
        if not recon_path.is_absolute():
            recon_path = result_dir / recon_path
        if not recon_path.exists():
            # fallback: map by source filename under recon/
            recon_path = result_dir / "recon" / Path(src)

        target_path: Optional[Path] = None
        if target_path_raw:
            tp = Path(target_path_raw)
            if not tp.is_absolute():
                tp = result_dir / tp
            if tp.exists():
                target_path = tp
        if target_path is None:
            tp2 = result_dir / "target" / Path(src)
            if tp2.exists():
                target_path = tp2

        k = (idx_str, meta_key, src)
        metric_row = metric_by_key.get(k, {})
        metric_dict: Dict[str, float] = {}
        for kk, vv in metric_row.items():
            if kk in ("index", "meta_key", "source_filename", "length"):
                continue
            fx = _to_float(vv)
            if _finite(fx):
                metric_dict[kk] = fx

        rows.append(
            SampleRow(
                index=int(_to_float(idx_str)),
                meta_key=meta_key,
                source_filename=src,
                length=length_i,
                recon_path=recon_path,
                target_path=target_path,
                metrics=metric_dict,
            )
        )

    # Keep only rows with valid recon audio files.
    rows = [r for r in rows if r.recon_path.is_file()]
    if len(rows) == 0:
        raise RuntimeError("No valid recon audio rows found.")
    return rows


def _slugify_metric(name: str) -> str:
    name = name.replace("test/", "")
    name = name.replace("flux_onset", "SF")
    name = name.replace("lsd", "LSD")
    name = name.replace("loss", "MSS/Loss")
    name = name.replace("mss_sc", "MSS_SC")
    name = name.replace("mss_log", "MSS_LOG")
    return name


def _load_mono(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32 using robust backend fallback.
    Priority:
      1) torchaudio
      2) soundfile
      3) scipy.io.wavfile
    This avoids Windows torchcodec/ffmpeg dependency issues.
    """
    # 1) torchaudio path (fast in torch-enabled environments)
    try:
        wav, sr = torchaudio.load(str(path))
        wav_np = wav[:1, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        return wav_np, int(sr)
    except Exception:
        pass

    # 2) soundfile fallback (handles float WAV well on Windows)
    if sf is not None:
        try:
            data, sr = sf.read(str(path), always_2d=True)
            mono = data[:, :1].squeeze(1).astype(np.float32, copy=False)
            return mono, int(sr)
        except Exception:
            pass

    # 3) scipy fallback
    if scipy_wavfile is not None:
        try:
            sr, data = scipy_wavfile.read(str(path))
            if data.ndim == 2:
                data = data[:, 0]
            if np.issubdtype(data.dtype, np.integer):
                maxv = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max(maxv, 1)
            else:
                data = data.astype(np.float32, copy=False)
            return data, int(sr)
        except Exception:
            pass

    raise RuntimeError(
        f"Failed to read audio: {path}. Install one of: torchaudio-compatible torchcodec,"
        " soundfile, or scipy."
    )


def _plot_waveform_png(audio: np.ndarray, sr: int, title: str) -> bytes:
    # ### HIGHLIGHT: High-resolution waveform render for presentation quality.
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=180)
    t = np.arange(audio.shape[0], dtype=np.float64) / max(sr, 1)
    ax.plot(t, audio, linewidth=0.9, color="#0f766e")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amp", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlim(0, t[-1] if t.size > 0 else 1.0)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _plot_spectrogram_png(audio: np.ndarray, sr: int, title: str) -> bytes:
    # ### HIGHLIGHT: High-resolution STFT spectrogram with dB scale.
    x = torch.from_numpy(audio.astype(np.float32))
    n_fft = 2048
    hop = 256
    win = torch.hann_window(n_fft)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=True,
        pad_mode="constant",
    )
    mag = spec.abs().clamp_min(1e-8)
    db = 20.0 * torch.log10(mag)
    db_np = db.numpy()

    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=180)
    t_max = max(audio.shape[0] / max(sr, 1), 1e-6)
    f_max = sr / 2.0
    im = ax.imshow(
        db_np,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0.0, t_max, 0.0, f_max],
        vmin=max(float(np.percentile(db_np, 5)), -120.0),
        vmax=min(float(np.percentile(db_np, 99.7)), 0.0),
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Freq (Hz)", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.ax.set_ylabel("dB", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _b64_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _metric_table_html(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "<div class='muted'>No per-file metrics found.</div>"
    parts = ["<table class='metric-table'><tbody>"]
    for k in sorted(metrics.keys()):
        parts.append(
            f"<tr><td>{html.escape(_slugify_metric(k))}</td><td>{metrics[k]:.6f}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


class AuditionHandler(BaseHTTPRequestHandler):
    rows: List[SampleRow] = []
    n_show: int = 10
    cache_dir: Path

    def _render_png_cached(self, audio_path: Path, kind: str) -> bytes:
        stat = audio_path.stat()
        key = f"{audio_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{kind}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        out_path = self.cache_dir / f"{digest}.png"
        if out_path.is_file():
            return out_path.read_bytes()

        audio, sr = _load_mono(audio_path)
        if kind == "wave":
            png = _plot_waveform_png(audio, sr, f"Waveform | {audio_path.name}")
        elif kind == "spec":
            png = _plot_spectrogram_png(audio, sr, f"Spectrogram | {audio_path.name}")
        else:
            raise ValueError(f"Unknown kind: {kind}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        return png

    def _serve_audio(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, f"Audio not found: {path}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _pick_rows(self, seed: Optional[int]) -> List[SampleRow]:
        rng = random.Random(seed if seed is not None else time.time_ns())
        n = min(self.n_show, len(self.rows))
        return rng.sample(self.rows, n)

    def _render_index(self, seed: Optional[int]) -> bytes:
        selected = self._pick_rows(seed)
        cards: List[str] = []
        for i, row in enumerate(selected, start=1):
            sample_q = urllib.parse.quote(row.source_filename, safe="")
            recon_q = urllib.parse.quote(str(row.recon_path), safe="")
            target_q = urllib.parse.quote(str(row.target_path) if row.target_path else "", safe="")

            recon_wave = _b64_png(self._render_png_cached(row.recon_path, "wave"))
            recon_spec = _b64_png(self._render_png_cached(row.recon_path, "spec"))

            target_audio_html = "<div class='muted'>Target not available.</div>"
            target_wave_html = "<div class='muted'>Target waveform unavailable.</div>"
            target_spec_html = "<div class='muted'>Target spectrogram unavailable.</div>"
            if row.target_path is not None and row.target_path.is_file():
                target_wave = _b64_png(self._render_png_cached(row.target_path, "wave"))
                target_spec = _b64_png(self._render_png_cached(row.target_path, "spec"))
                target_audio_html = (
                    f"<audio controls preload='none' src='/audio?path={target_q}'></audio>"
                )
                target_wave_html = f"<img class='plot' src='{target_wave}' alt='target_wave'>"
                target_spec_html = f"<img class='plot' src='{target_spec}' alt='target_spec'>"

            card = f"""
            <section class="card">
              <div class="card-header">
                <div>
                  <h3>#{i} {html.escape(row.source_filename)}</h3>
                  <div class="muted">meta_key={html.escape(row.meta_key)} | length={row.length}</div>
                </div>
                <div class="metrics">{_metric_table_html(row.metrics)}</div>
              </div>
              <div class="audio-grid">
                <div>
                  <h4>Target Audio (padded/eval version)</h4>
                  {target_audio_html}
                </div>
                <div>
                  <h4>Reconstruction Audio</h4>
                  <audio controls preload='none' src='/audio?path={recon_q}'></audio>
                </div>
              </div>
              <div class="viz-grid">
                <div>
                  <h4>Target Waveform</h4>
                  {target_wave_html}
                </div>
                <div>
                  <h4>Target Spectrogram</h4>
                  {target_spec_html}
                </div>
                <div>
                  <h4>Recon Waveform</h4>
                  <img class='plot' src='{recon_wave}' alt='recon_wave'>
                </div>
                <div>
                  <h4>Recon Spectrogram</h4>
                  <img class='plot' src='{recon_spec}' alt='recon_spec'>
                </div>
              </div>
            </section>
            """
            cards.append(card)

        html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DrumBlender Audition Viewer</title>
  <style>
    :root {{
      --bg: #f3f5f7;
      --card: #ffffff;
      --text: #1a1e24;
      --muted: #5f6b7a;
      --accent: #0f766e;
      --border: #d8dee6;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 0% 0%, #e8f5ef 0, #f3f5f7 35%);
    }}
    header {{
      position: sticky;
      top: 0;
      backdrop-filter: blur(6px);
      background: rgba(243,245,247,0.88);
      border-bottom: 1px solid var(--border);
      padding: 14px 20px;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    h1 {{ margin: 0; font-size: 20px; }}
    .controls {{ display: flex; gap: 10px; align-items: center; }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 8px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }}
    .wrap {{ padding: 16px; max-width: 1400px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 2px 10px rgba(16,24,40,0.04);
    }}
    .card-header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: start;
    }}
    .card h3 {{ margin: 0 0 4px 0; font-size: 18px; }}
    .card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: #1f2937; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .metrics {{ min-width: 240px; }}
    .metric-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    .metric-table td {{
      border-bottom: 1px solid #edf1f5;
      padding: 4px 6px;
      vertical-align: top;
    }}
    .audio-grid, .viz-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .viz-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    audio {{ width: 100%; }}
    .plot {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    @media (max-width: 900px) {{
      .card-header, .audio-grid, .viz-grid {{
        grid-template-columns: 1fr;
      }}
      .metrics {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>DrumBlender Audition Viewer</h1>
    <div class="controls">
      <div class="muted">Showing {len(selected)} random samples</div>
      <button onclick="refreshSamples()">Refresh 10 Random Samples</button>
    </div>
  </header>
  <main class="wrap">
    {''.join(cards)}
  </main>
  <script>
    function refreshSamples() {{
      const u = new URL(window.location.href);
      u.searchParams.set("seed", String(Date.now()));
      window.location.href = u.toString();
    }}
  </script>
</body>
</html>
"""
        return html_doc.encode("utf-8")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        route = parsed.path

        if route == "/" or route == "/index.html":
            seed = None
            if "seed" in qs and len(qs["seed"]) > 0:
                try:
                    seed = int(qs["seed"][0])
                except Exception:
                    seed = None
            page = self._render_index(seed)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        if route == "/audio":
            p = qs.get("path", [""])[0]
            if p == "":
                self.send_error(400, "Missing path")
                return
            path = Path(urllib.parse.unquote(p))
            self._serve_audio(path)
            return

        self.send_error(404, "Not Found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, help="Result folder exported by export_recon_wavs.py")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per page")
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open browser automatically (default: on)",
    )
    parser.add_argument(
        "--no-open",
        dest="open_browser",
        action="store_false",
        help="Do not open browser automatically",
    )
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)

    rows = _collect_rows(result_dir)
    print(f"[OK] Loaded {len(rows)} samples from: {result_dir}")

    AuditionHandler.rows = rows
    AuditionHandler.n_show = max(1, int(args.num_samples))
    AuditionHandler.cache_dir = result_dir / ".viz_cache"
    AuditionHandler.cache_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), AuditionHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[OK] Serving audition viewer at: {url}")

    if args.open_browser:
        # ### HIGHLIGHT: Browser open is non-fatal for headless environments.
        def _open_later() -> None:
            try:
                time.sleep(0.4)
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open_later, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
