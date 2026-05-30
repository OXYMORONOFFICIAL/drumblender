#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/drumblender}"
CFG_PATH="${CFG_PATH:-cfg/05_all_parallel.yaml}"
PACK_GLOB="${PACK_GLOB:-cfg/data/custom_pack_*.yaml}"
METRICS_CFG="${METRICS_CFG:-cfg/metrics/drumblender_metrics.yaml}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/ckpt/last.ckpt}"
RUN_ROOT_PARENT="${RUN_ROOT_PARENT:-$(cd "$REPO_ROOT/.." && pwd)}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT_PARENT/$RUN_NAME}"
RESULT_DIR="${RESULT_DIR:-$RUN_DIR/per_pack}"
REPORT_DIR="${REPORT_DIR:-$RUN_DIR/reports}"
RUN_PREFIX="${RUN_PREFIX:-$(basename "$CFG_PATH" .yaml)}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
SAVE_TARGET="${SAVE_TARGET:-on}"
MAKE_TAR="${MAKE_TAR:-off}"
LOSS_MODE="${1:-${LOSS_MODE:-plain}}"
NOISE_ENCODER_MODE="${2:-${NOISE_ENCODER_MODE:-baseline}}"
TRANSIENT_ENCODER_MODE="${3:-${TRANSIENT_ENCODER_MODE:-baseline}}"

cd "$REPO_ROOT" || exit 1
source "$REPO_ROOT/scripts/encoder_modes.sh"

usage() {
  printf 'usage: bash test_per_pack.sh [plain|amp|smooth|si|legacy_si] [%s] [%s]\n' \
    "$DRUMBLENDER_NOISE_ENCODER_MODE_USAGE" \
    "$DRUMBLENDER_TRANSIENT_ENCODER_MODE_USAGE" >&2
}

case "$LOSS_MODE" in
  plain|baseline|off)
    LOSS_CFG_PATH=""
    ;;
  amp)
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/mss_log_rms.yaml"
    ;;
  smooth)
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/mss_smoothl1.yaml"
    ;;
  si|legacy_si|on)
    LOSS_CFG_PATH="$REPO_ROOT/cfg/loss/safe_mss.yaml"
    ;;
  *)
    usage
    exit 1
    ;;
esac

IFS=$'\t' read -r _ NOISE_ENCODER_CFG_PATH < <(
  drumblender_resolve_encoder_mode noise "$NOISE_ENCODER_MODE" "$REPO_ROOT"
) || {
  usage
  exit 1
}

IFS=$'\t' read -r _ TRANSIENT_ENCODER_CFG_PATH < <(
  drumblender_resolve_encoder_mode transient "$TRANSIENT_ENCODER_MODE" "$REPO_ROOT"
) || {
  usage
  exit 1
}

if [[ ! -f "$CFG_PATH" ]]; then
  printf '[test_per_pack.sh] missing config: %s\n' "$CFG_PATH" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  printf '[test_per_pack.sh] missing checkpoint: %s\n' "$CKPT_PATH" >&2
  exit 1
fi

shopt -s nullglob
PACK_CFGS=($PACK_GLOB)
shopt -u nullglob

if [[ "${#PACK_CFGS[@]}" -eq 0 ]]; then
  printf '[test_per_pack.sh] no pack configs matched: %s\n' "$PACK_GLOB" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR" "$REPORT_DIR"

for DATA_CFG in "${PACK_CFGS[@]}"; do
  pack_name="$(basename "$DATA_CFG" .yaml)"
  pack_name="${pack_name#custom_pack_}"
  output_dir="${RESULT_DIR}/${RUN_PREFIX}_${pack_name}"

  CMD=(
    python scripts/export_recon_wavs.py
    --config "$CFG_PATH"
    --ckpt "$CKPT_PATH"
    --data-config "$DATA_CFG"
    --split "$SPLIT"
    --metrics-config "$METRICS_CFG"
    --output-dir "$output_dir"
    --device "$DEVICE"
  )

  if [[ -n "$LOSS_CFG_PATH" ]]; then
    CMD+=(--loss-cfg "$LOSS_CFG_PATH")
  fi

  if [[ -n "$NOISE_ENCODER_CFG_PATH" ]]; then
    CMD+=(--noise-encoder-cfg "$NOISE_ENCODER_CFG_PATH")
  fi

  if [[ -n "$TRANSIENT_ENCODER_CFG_PATH" ]]; then
    CMD+=(--transient-encoder-cfg "$TRANSIENT_ENCODER_CFG_PATH")
  fi

  if [[ "$SAVE_TARGET" == "on" ]]; then
    CMD+=(--save-target)
  fi

  if [[ "$MAKE_TAR" == "on" ]]; then
    CMD+=(--make-tar)
  fi

  printf '[test_per_pack.sh] pack: %s\n' "$pack_name"
  printf '[test_per_pack.sh] output: %s\n' "$output_dir"
  printf '[test_per_pack.sh] loss mode: %s\n' "$LOSS_MODE"
  printf '[test_per_pack.sh] noise encoder mode: %s\n' "$NOISE_ENCODER_MODE"
  if [[ -n "$NOISE_ENCODER_CFG_PATH" ]]; then
    printf '[test_per_pack.sh] noise encoder cfg: %s\n' "$NOISE_ENCODER_CFG_PATH"
  fi
  printf '[test_per_pack.sh] transient encoder mode: %s\n' "$TRANSIENT_ENCODER_MODE"
  if [[ -n "$TRANSIENT_ENCODER_CFG_PATH" ]]; then
    printf '[test_per_pack.sh] transient encoder cfg: %s\n' "$TRANSIENT_ENCODER_CFG_PATH"
  fi
  "${CMD[@]}"
done

python scripts/postprocess_results.py "$RUN_DIR" --report-dir "$REPORT_DIR"
