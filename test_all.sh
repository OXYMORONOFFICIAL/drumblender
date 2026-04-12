#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/drumblender}"
CFG_PATH="${CFG_PATH:-cfg/05_all_parallel.yaml}"
DATA_CFG="${DATA_CFG:-cfg/data/custom_all.yaml}"
METRICS_CFG="${METRICS_CFG:-cfg/metrics/drumblender_metrics.yaml}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/ckpt/last.ckpt}"
RUN_ROOT_PARENT="${RUN_ROOT_PARENT:-$(cd "$REPO_ROOT/.." && pwd)}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT_PARENT/$RUN_NAME}"
RUN_PREFIX="${RUN_PREFIX:-$(basename "$CFG_PATH" .yaml)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/all/${RUN_PREFIX}_all}"
REPORT_DIR="${REPORT_DIR:-$RUN_DIR/reports}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
SAVE_TARGET="${SAVE_TARGET:-on}"
MAKE_TAR="${MAKE_TAR:-off}"

cd "$REPO_ROOT" || exit 1

if [[ ! -f "$CFG_PATH" ]]; then
  printf '[test_all.sh] missing config: %s\n' "$CFG_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATA_CFG" ]]; then
  printf '[test_all.sh] missing data config: %s\n' "$DATA_CFG" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  printf '[test_all.sh] missing checkpoint: %s\n' "$CKPT_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" "$REPORT_DIR"

CMD=(
  python scripts/export_recon_wavs.py
  --config "$CFG_PATH"
  --ckpt "$CKPT_PATH"
  --data-config "$DATA_CFG"
  --split "$SPLIT"
  --metrics-config "$METRICS_CFG"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

if [[ "$SAVE_TARGET" == "on" ]]; then
  CMD+=(--save-target)
fi

if [[ "$MAKE_TAR" == "on" ]]; then
  CMD+=(--make-tar)
fi

printf '[test_all.sh] output: %s\n' "$OUTPUT_DIR"
printf '[test_all.sh] data config: %s\n' "$DATA_CFG"
"${CMD[@]}"

python scripts/postprocess_results.py "$RUN_DIR" --report-dir "$REPORT_DIR"
