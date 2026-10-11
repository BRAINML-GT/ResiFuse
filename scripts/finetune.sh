#!/bin/bash
# Supervised behavior classification with residual-guided frame selection.
#
#   [PRETRAIN_CKPT=<pretrain .pth>] bash scripts/finetune.sh <mabe|calms21|freely> <budget> [seed] [select]
#
#   PRETRAIN_CKPT  default checkpoints/resifuse_pretrain_<dataset>.pth (the released weights)
#
#   budget  fraction of frames that receive vision, in [0, 1]
#           (paper, Table 1: freely 0.75, calms21 1.0, mabe 0.75)
#   select  residual (default) | uniform
#
# Outputs: runs/finetune/<dataset>_b<budget>_<select>_s<seed>/{config.yaml,train.log}
# (or OUT_DIR if set); checkpoints go to finetune_save_dir of the config.
# CONFIG_OVERRIDES="key=value ..." changes config values for this run only.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

DATASET=${1:?usage: [PRETRAIN_CKPT=...] bash scripts/finetune.sh <dataset> <budget> [seed] [select]}
BUDGET=${2:?budget required}
SEED=${3:-0}
SELECT=${4:-residual}
check_dataset "${DATASET}"
PRETRAIN_CKPT=${PRETRAIN_CKPT:-checkpoints/resifuse_pretrain_${DATASET}.pth}
require_ckpt "${PRETRAIN_CKPT}"
read -r -a EXTRA <<< "${CONFIG_OVERRIDES:-}"  # optional "key=value ..." for this run only

OUT=${OUT_DIR:-runs/finetune/${DATASET}_b${BUDGET}_${SELECT}_s${SEED}}
derive_config "configs/${DATASET}_finetune.yaml" "${OUT}/config.yaml" \
  "budget_ratio=${BUDGET}" \
  "frame_select_mode=${SELECT}" \
  "pretrain_ckpt=\"${PRETRAIN_CKPT}\"" \
  ${EXTRA[@]+"${EXTRA[@]}"}

python -m resifuse.train --config "${OUT}/config.yaml" --seed "${SEED}" 2>&1 | tee "${OUT}/train.log"
