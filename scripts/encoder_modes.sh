#!/usr/bin/env bash

DRUMBLENDER_NOISE_ENCODER_MODE_USAGE="baseline|dac|dac_lstm|dac_len|dac_lstm_len|dac_len_lstm_len|wavtokenizer|bscodec|apcodec|spectrostream"
DRUMBLENDER_TRANSIENT_ENCODER_MODE_USAGE="baseline|dac|wavtokenizer|bscodec|apcodec|spectrostream"

drumblender_resolve_encoder_mode() {
  local kind="$1"
  local mode="$2"
  local repo_root="$3"

  local stem=""
  local tag=""

  case "$mode" in
    baseline|soundstream|off)
      printf '\t\n'
      return 0
      ;;
    dac)
      stem="dac"
      ;;
    dac_lstm|daclstm|sequence)
      stem="dac_lstm"
      ;;
    dac_len|daclen)
      stem="dac_len"
      ;;
    dac_lstm_len|daclstm_len|daclstmlen)
      stem="dac_lstm_len"
      ;;
    dac_len_lstm_len|daclen_lstm_len|daclenlstmlen)
      stem="dac_len_lstm_len"
      ;;
    wavtokenizer|wavtokenizer_style|wavtok)
      stem="wavtokenizer"
      ;;
    bscodec|bscodec_style|bs)
      stem="bscodec"
      ;;
    apcodec|apcodec_style|ap)
      stem="apcodec"
      ;;
    spectrostream|spectrostream_style|spectro)
      stem="spectrostream"
      ;;
    *)
      printf 'unknown %s encoder mode: %s\n' "$kind" "$mode" >&2
      return 2
      ;;
  esac

  if [[ "$kind" == "transient" ]]; then
    case "$stem" in
      dac_lstm|dac_len|dac_lstm_len|dac_len_lstm_len)
        printf 'transient encoder mode "%s" is not available. Valid modes: %s\n' \
          "$mode" "$DRUMBLENDER_TRANSIENT_ENCODER_MODE_USAGE" >&2
        return 2
        ;;
    esac
  fi

  case "$kind:$stem" in
    noise:dac) tag="NOISEDAC_" ;;
    noise:dac_lstm) tag="NOISEDACLSTM_" ;;
    noise:dac_len) tag="NOISEDACLEN_" ;;
    noise:dac_lstm_len) tag="NOISEDACLSTMLEN_" ;;
    noise:dac_len_lstm_len) tag="NOISEDACLENLSTMLEN_" ;;
    noise:wavtokenizer) tag="NOISEWAVTOKENIZER_" ;;
    noise:bscodec) tag="NOISEBSCODEC_" ;;
    noise:apcodec) tag="NOISEAPCODEC_" ;;
    noise:spectrostream) tag="NOISESPECTROSTREAM_" ;;
    transient:dac) tag="TRANSDAC_" ;;
    transient:wavtokenizer) tag="TRANSWAVTOKENIZER_" ;;
    transient:bscodec) tag="TRANSBSCODEC_" ;;
    transient:apcodec) tag="TRANSAPCODEC_" ;;
    transient:spectrostream) tag="TRANSSPECTROSTREAM_" ;;
    *)
      printf 'unsupported %s encoder mode: %s\n' "$kind" "$mode" >&2
      return 2
      ;;
  esac

  local cfg_path="$repo_root/cfg/upgrades/encoders/${kind}_${stem}_style.yaml"
  if [[ ! -f "$cfg_path" ]]; then
    printf '%s encoder config not found: %s\n' "$kind" "$cfg_path" >&2
    return 2
  fi

  printf '%s\t%s\n' "$tag" "$cfg_path"
}
