#!/bin/bash
# Unsupervised behavior segmentation (DEC on the fused representation) with
# residual-guided frame selection.
#
#   [PRETRAIN_CKPT=<pretrain .pth>] bash scripts/unsup.sh <mabe|calms21|freely> <budget> [seed] [select]
#
#   PRETRAIN_CKPT  default checkpoints/resifuse_pretrain_<dataset>.pth (the released weights)
#
#   budget  fraction of frames that receive vision, in [0, 1]
#           (paper, Table 1: freely 0.25, calms21 0.75, mabe 0.75)
#   select  residual (default) | uniform
#
# DEC settings follow the paper's appendix: 30 epochs and K = 50 / 50 / 30
# (mabe / calms21 / freely, from the config); VICReg 0.5 / 1.0 / 0.1;
# per-epoch target distribution on the full training set; empty-cluster
# reseeding; uniform k-means sampling; the visual term centered by a fixed
# training-set mean. The log ends with the test F1 of the epoch with the best
# validation macro F1.
#
# Outputs: runs/unsup/<dataset>_b<budget>_<select>_s<seed>/{config.yaml,train.log}
# (or OUT_DIR if set). CONFIG_OVERRIDES="key=value ..." changes config values
# for this run only.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

DATASET=${1:?usage: [PRETRAIN_CKPT=...] bash scripts/unsup.sh <dataset> <budget> [seed] [select]}
BUDGET=${2:?budget required}
SEED=${3:-0}
SELECT=${4:-residual}
check_dataset "${DATASET}"
PRETRAIN_CKPT=${PRETRAIN_CKPT:-checkpoints/resifuse_pretrain_${DATASET}.pth}
require_ckpt "${PRETRAIN_CKPT}"
read -r -a EXTRA <<< "${CONFIG_OVERRIDES:-}"  # optional "key=value ..." for this run only

VICREG="--vicreg-inv 0.5 --vicreg-var 1.0 --vicreg-cov 0.1"
DEC_FIXES="--dec-freeze-vision-center --dec-kmeans-random-sample --dec-target-update epoch --dec-reseed-empty"
case ${DATASET} in
  mabe)    ARGS="--dec-train-agg per_animal --dec-eval-agg mean" ;;
  calms21) ARGS="--dec-train-agg concat --dec-eval-agg concat --majority-map-mode f1" ;;
  freely)  ARGS="--dec-train-agg concat --dec-eval-agg concat --majority-map-mode f1" ;;
esac

RUN=${DATASET}_b${BUDGET}_${SELECT}_s${SEED}
OUT=${OUT_DIR:-runs/unsup/${RUN}}
derive_config "configs/${DATASET}_finetune.yaml" "${OUT}/config.yaml" \
  "budget_ratio=${BUDGET}" \
  "frame_select_mode=${SELECT}" \
  "pretrain_ckpt=\"${PRETRAIN_CKPT}\"" \
  "vision_center_mean=true" \
  "vision_center_momentum=0.99" \
  "unsup_run_name=${RUN}" \
  ${EXTRA[@]+"${EXTRA[@]}"}

python -m "resifuse.training.unsup_${DATASET}" --config "${OUT}/config.yaml" \
  ${ARGS} ${VICREG} ${DEC_FIXES} --seed "${SEED}" 2>&1 | tee "${OUT}/train.log"
