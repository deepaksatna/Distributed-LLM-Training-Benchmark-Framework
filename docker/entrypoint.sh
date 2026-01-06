#!/usr/bin/env bash
set -euo pipefail

echo "=== Distributed Training Entrypoint ==="
date

export STRATEGY="${STRATEGY:-ddp}"             # ddp | fsdp | zero2 | zero3
export WORLD_SIZE="${WORLD_SIZE:-1}"

# Compute RANK from JOB_COMPLETION_INDEX (for Kubernetes Indexed Jobs)
if [ -n "${JOB_COMPLETION_INDEX:-}" ]; then
  export RANK=$((JOB_COMPLETION_INDEX + 1))
  echo "Computed RANK=$RANK from JOB_COMPLETION_INDEX=$JOB_COMPLETION_INDEX"
else
  export RANK="${RANK:-0}"
fi

export LOCAL_RANK="${LOCAL_RANK:-0}"

# For master (rank 0), use POD_IP if available, otherwise use MASTER_ADDR env var
if [ "$RANK" = "0" ] && [ -n "${POD_IP:-}" ]; then
  export MASTER_ADDR="$POD_IP"
  echo "Master using POD_IP: $MASTER_ADDR"
else
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

export MASTER_PORT="${MASTER_PORT:-29500}"
export SEQ_LEN="${SEQ_LEN:-2048}"
export TIER="${TIER:-A}"                       # A | B
export STEPS="${STEPS:-50}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export SYNTHETIC="${SYNTHETIC:-true}"
export RESULTS_DIR="${RESULTS_DIR:-/results}"

echo "Config:"
echo "  STRATEGY=$STRATEGY"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  RANK=$RANK"
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  SEQ_LEN=$SEQ_LEN"
echo "  TIER=$TIER"
echo "  STEPS=$STEPS"
echo "  PER_DEVICE_BATCH=$PER_DEVICE_BATCH"
echo "  GRAD_ACCUM=$GRAD_ACCUM"
echo ""

# GPU check
echo "GPU Status:"
nvidia-smi -L || echo "WARNING: nvidia-smi failed"
echo ""

ARGS="--strategy ${STRATEGY} --world-size ${WORLD_SIZE} --rank ${RANK} --local-rank ${LOCAL_RANK} --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} --seq-len ${SEQ_LEN} --tier ${TIER} --steps ${STEPS} --warmup-steps ${WARMUP_STEPS} --per-device-batch ${PER_DEVICE_BATCH} --grad-accum ${GRAD_ACCUM} --results-dir ${RESULTS_DIR}"
if [[ "${SYNTHETIC}" == "true" ]]; then ARGS="${ARGS} --synthetic"; fi
if [[ "${STRATEGY}" == "zero2" ]]; then ARGS="${ARGS} --deepspeed-config /app/configs/deepspeed/zero2.json"; fi
if [[ "${STRATEGY}" == "zero3" ]]; then ARGS="${ARGS} --deepspeed-config /app/configs/deepspeed/zero3.json"; fi
if [[ "${STRATEGY}" == "fsdp" ]]; then ARGS="${ARGS} --fsdp-config /app/configs/fsdp/fsdp_config.yaml"; fi

echo "=== Launching Training ==="
echo "Command: python -u /app/benchmarking/train_harness.py ${ARGS}"
echo ""

exec python -u /app/benchmarking/train_harness.py ${ARGS}
