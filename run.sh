#!/usr/bin/env bash
set -euo pipefail

cd /workspace/drumblender || exit 1

WANDB_PROJECT="${WANDB_PROJECT:-drumblender}"
WANDB_NAME="${WANDB_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-/workspace/drumblender/logs/wandb}"
RUN_SEED="${RUN_SEED:-20260218}"

CFG="${CFG:-/workspace/drumblender/cfg/05_all_parallel.yaml}"
DATA_DIR="${DATA_DIR:-/private/datasets/modal_features/processed_modal_flat}"
CKPT_DIR="${CKPT_DIR:-/workspace/drumblender/ckpt}"
RUN_CKPT_DIR="${CKPT_DIR}/${WANDB_NAME}"
RUN_LOG_FILE="${RUN_CKPT_DIR}/train.log"
RESUME_CKPT="${RESUME_CKPT:-}"
MAX_EPOCHS="${MAX_EPOCHS:-70}"
ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-2}"
DRY_RUN="${DRY_RUN:-off}"
LOSS_UPGRADE="${LOSS_UPGRADE:-off}"
LOSS_CFG="${LOSS_CFG:-}"
SI_NORM="${SI_NORM:-on}"
DECAY_PRIOR="${DECAY_PRIOR:-off}"
NOISE_ENCODER_BACKBONE="${NOISE_ENCODER_BACKBONE:-soundstream}"
TRANSIENT_ENCODER_BACKBONE="${TRANSIENT_ENCODER_BACKBONE:-soundstream}"
NOISE_ENCODER_CFG="${NOISE_ENCODER_CFG:-}"
TRANSIENT_ENCODER_CFG="${TRANSIENT_ENCODER_CFG:-}"

CFG_DIR="$(cd "$(dirname "$CFG")" && pwd)"
if [[ -z "$LOSS_CFG" ]]; then
  if [[ "$LOSS_UPGRADE" == "on" ]]; then
    if [[ -f "${CFG_DIR}/upgrades/loss/safe_mss.yaml" ]]; then
      LOSS_CFG="${CFG_DIR}/upgrades/loss/safe_mss.yaml"
    else
      LOSS_CFG="${CFG_DIR}/loss/safe_mss.yaml"
    fi
  else
    LOSS_CFG="${CFG_DIR}/loss/mss.yaml"
  fi
fi

resolve_encoder_cfg() {
  local kind="$1"      # noise | transient
  local backbone="$2"  # soundstream | dac | hybrid | apcodec | discodec
  local out=""
  case "$backbone" in
    soundstream)
      out=""
      ;;
    dac)
      out="${CFG_DIR}/upgrades/encoders/${kind}_dac_style.yaml"
      ;;
    hybrid)
      out="${CFG_DIR}/upgrades/encoders/${kind}_hybrid_style.yaml"
      ;;
    apcodec)
      out="${CFG_DIR}/upgrades/encoders/${kind}_apcodec_style.yaml"
      ;;
    discodec)
      out="${CFG_DIR}/upgrades/encoders/${kind}_discodec_style.yaml"
      ;;
    *)
      echo "Invalid ${kind} encoder backbone: '${backbone}'" >&2
      echo "Valid values: soundstream | dac | hybrid | apcodec | discodec" >&2
      exit 1
      ;;
  esac
  printf '%s' "$out"
}

if [[ -z "$NOISE_ENCODER_CFG" ]]; then
  NOISE_ENCODER_CFG="$(resolve_encoder_cfg noise "$NOISE_ENCODER_BACKBONE")"
fi
if [[ -z "$TRANSIENT_ENCODER_CFG" ]]; then
  TRANSIENT_ENCODER_CFG="$(resolve_encoder_cfg transient "$TRANSIENT_ENCODER_BACKBONE")"
fi

if [[ -n "$NOISE_ENCODER_CFG" && ! -f "$NOISE_ENCODER_CFG" ]]; then
  echo "Noise encoder config not found for backbone '$NOISE_ENCODER_BACKBONE': $NOISE_ENCODER_CFG" >&2
  exit 1
fi
if [[ -n "$TRANSIENT_ENCODER_CFG" && ! -f "$TRANSIENT_ENCODER_CFG" ]]; then
  echo "Transient encoder config not found for backbone '$TRANSIENT_ENCODER_BACKBONE': $TRANSIENT_ENCODER_CFG" >&2
  exit 1
fi

to_bool() {
  case "${1,,}" in
    on|true|1|yes|y) echo "true" ;;
    off|false|0|no|n) echo "false" ;;
    *)
      echo "Invalid boolean toggle value: '$1' (use on/off)" >&2
      exit 1
      ;;
  esac
}

# ### HIGHLIGHT: Always create output directories used by logger/checkpoint callbacks.
mkdir -p "$WANDB_DIR" "$CKPT_DIR" "$RUN_CKPT_DIR" /workspace/drumblender/lightning_logs

# (Optional) CUDA allocator fragmentation mitigation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CMD=(
  drumblender fit -c "$CFG"
  --seed_everything "$RUN_SEED"
  --trainer.accelerator gpu
  --trainer.devices 1
  --trainer.precision 32
  --trainer.max_epochs "$MAX_EPOCHS"
  --trainer.accumulate_grad_batches "$ACCUM_GRAD_BATCHES"
  --trainer.log_every_n_steps 40
  --trainer.num_sanity_val_steps 0
  --trainer.val_check_interval 1.0
  --trainer.default_root_dir /workspace/drumblender/lightning_logs
  --trainer.logger pytorch_lightning.loggers.WandbLogger
  --trainer.logger.init_args.project "$WANDB_PROJECT"
  --trainer.logger.init_args.name "$WANDB_NAME"
  --trainer.logger.init_args.save_dir "$WANDB_DIR"
  --trainer.logger.init_args.log_model false
  --model.init_args.loss_fn "$LOSS_CFG"
  --data.class_path drumblender.data.AudioDataModule
  --data.data_dir "$DATA_DIR"
  --data.meta_file metadata.json
  --data.dataset_class drumblender.data.AudioWithParametersDataset
  --data.dataset_kwargs "{parameter_key: feature_file, split_strategy: sample_pack, expected_num_modes: 64, seed: $RUN_SEED}"
  --data.seed "$RUN_SEED"
  --data.sample_rate 48000
  --data.num_samples null
  --data.batch_size 1
  --data.num_workers 6
  --data.skip_prepare_data true
  --data.use_bucketing true
  --data.bucket_boundaries "[48000, 96000, 192000, 384000]"
  --data.drop_last true
)

# ### HIGHLIGHT: Resume training from an explicit checkpoint when needed.
if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(--ckpt_path "$RESUME_CKPT")
fi

if [[ -n "$NOISE_ENCODER_CFG" ]]; then
  CMD+=(
    --model.init_args.noise_autoencoder "$NOISE_ENCODER_CFG"
    --model.init_args.noise_autoencoder_accepts_audio true
  )
fi

if [[ -n "$TRANSIENT_ENCODER_CFG" ]]; then
  CMD+=(
    --model.init_args.transient_autoencoder "$TRANSIENT_ENCODER_CFG"
    --model.init_args.transient_autoencoder_accepts_audio true
  )
fi

if [[ "$LOSS_UPGRADE" == "on" ]]; then
  SI_NORM_BOOL="$(to_bool "$SI_NORM")"
  DECAY_PRIOR_BOOL="$(to_bool "$DECAY_PRIOR")"
  CMD+=(
    --model.init_args.loss_fn.init_args.si_enabled "$SI_NORM_BOOL"
    --model.init_args.loss_fn.init_args.prior_enabled "$DECAY_PRIOR_BOOL"
  )
fi

LAUNCH_CMD="${CMD[*]}"
RUN_CONTEXT_JSON="$(cat <<JSON
{
  "script": "run.sh",
  "cfg": "$CFG",
  "data_dir": "$DATA_DIR",
  "seed": "$RUN_SEED",
  "max_epochs": "$MAX_EPOCHS",
  "accumulate_grad_batches": "$ACCUM_GRAD_BATCHES",
  "loss_upgrade": "$LOSS_UPGRADE",
  "loss_cfg": "$LOSS_CFG",
  "si_norm": "$SI_NORM",
  "decay_prior": "$DECAY_PRIOR",
  "noise_encoder_backbone": "$NOISE_ENCODER_BACKBONE",
  "noise_encoder_cfg": "$NOISE_ENCODER_CFG",
  "transient_encoder_backbone": "$TRANSIENT_ENCODER_BACKBONE",
  "transient_encoder_cfg": "$TRANSIENT_ENCODER_CFG",
  "run_ckpt_dir": "$RUN_CKPT_DIR",
  "run_log_file": "$RUN_LOG_FILE",
  "resume_ckpt": "$RESUME_CKPT",
  "launch_cmd": "$LAUNCH_CMD"
}
JSON
)"
export DRUMBLENDER_RUN_CONTEXT_JSON="$RUN_CONTEXT_JSON"
RUN_CONTEXT_FILE="${RUN_CKPT_DIR}/run-context-${WANDB_NAME}.json"
printf '%s\n' "$RUN_CONTEXT_JSON" > "$RUN_CONTEXT_FILE"
echo "[RUN_CONTEXT] Saved: $RUN_CONTEXT_FILE"

case "${DRY_RUN,,}" in
  on|true|1|yes|y)
    echo "[DRY_RUN] Final launch command:"
    printf '%q ' "${CMD[@]}"
    echo
    echo "[DRY_RUN] Run checkpoint dir: $RUN_CKPT_DIR"
    echo "[DRY_RUN] Run log file: $RUN_LOG_FILE"
    exit 0
    ;;
esac

"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
