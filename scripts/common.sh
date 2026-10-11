#!/bin/bash
# Shared helpers for the run scripts. Source it; do not run it directly.

# derive_config BASE OUT key=value [key=value ...]
# Copies BASE to OUT without the listed keys, then appends them with the new
# values, so a run never edits the configs in configs/.
derive_config() {
  local base=$1 out=$2
  shift 2
  local keys=""
  for kv in "$@"; do keys="${keys}|${kv%%=*}"; done
  mkdir -p "$(dirname "${out}")"
  grep -v -E "^(${keys#|}):" "${base}" > "${out}"
  {
    echo ""
    echo "# --- overrides written by $(basename "$0") ---"
    for kv in "$@"; do echo "${kv%%=*}: ${kv#*=}"; done
  } >> "${out}"
}

# require_ckpt PATH
# Fine-tuning silently trains from scratch when the pretrain checkpoint is
# missing, so fail early instead.
require_ckpt() {
  if [ -z "${1:-}" ] || [ ! -f "$1" ]; then
    echo "error: PRETRAIN_CKPT='${1:-}' is not a file; download the released checkpoints" \
         "into checkpoints/ or set PRETRAIN_CKPT (see README, Pretrained checkpoints)." >&2
    exit 1
  fi
}

check_dataset() {
  case $1 in
    mabe|calms21|freely) ;;
    *) echo "error: dataset must be mabe, calms21, or freely (got '$1')" >&2; exit 1 ;;
  esac
}
