#!/bin/bash
# Pretrain the motion encoder (masked pose reconstruction, RVQ, cross-modal
# vision prediction).
#
#   [RUN_DIR=<dir>] [SAVE_INTERVAL=<epochs>] [SEED=<int>] [CONFIG_OVERRIDES="key=value ..."] \
#     bash scripts/pretrain.sh <mabe|calms21|openfield> [num_gpus]
#
# Without RUN_DIR, checkpoints go to a new timestamped directory under the
# pretrain_save_dir of configs/<dataset>_pretrain.yaml. With RUN_DIR, they go
# to that directory, and running the same command again resumes from its
# latest.pth (e.g. after a preempted job). SAVE_INTERVAL sets how often a
# periodic checkpoint is written (default from the config: every 25 epochs).
# SEED fixes the RNG seed (default: unseeded). CONFIG_OVERRIDES changes config
# values for this run only, e.g. "batch_size=64"; it needs RUN_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

DATASET=${1:?usage: [RUN_DIR=<dir>] bash scripts/pretrain.sh <mabe|calms21|openfield> [num_gpus]}
NGPU=${2:-1}
check_dataset "${DATASET}"

CFG=configs/${DATASET}_pretrain.yaml
if [ -n "${RUN_DIR:-}" ]; then
  OVERRIDES=("run_dir=${RUN_DIR}")
  if [ -n "${SAVE_INTERVAL:-}" ]; then OVERRIDES+=("pretrain_save_interval=${SAVE_INTERVAL}"); fi
  read -r -a EXTRA <<< "${CONFIG_OVERRIDES:-}"
  derive_config "${CFG}" "${RUN_DIR}/config.yaml" "${OVERRIDES[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
  CFG=${RUN_DIR}/config.yaml
elif [ -n "${CONFIG_OVERRIDES:-}" ]; then
  echo "error: CONFIG_OVERRIDES needs RUN_DIR" >&2
  exit 1
fi
SEED_ARGS=()
if [ -n "${SEED:-}" ]; then SEED_ARGS=(--seed "${SEED}"); fi

torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29500}" \
  -m resifuse.train --config "${CFG}" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"}
