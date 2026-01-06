#!/usr/bin/env bash
set -euo pipefail

# Usage: ./launch_multi.sh --strategy ddp --world-size 2 --seq-len 2048 --tier A --image <image>

STRATEGY=""
WORLD_SIZE=""
SEQ_LEN="2048"
TIER="A"
STEPS="100"
PER_DEVICE_BATCH="1"
GRAD_ACCUM="4"
IMAGE="fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1"
NAMESPACE="bench"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --per-device-batch) PER_DEVICE_BATCH="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --synthetic) shift ;;  # Always synthetic
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Validate
if [[ -z "$STRATEGY" || -z "$WORLD_SIZE" ]]; then
  echo "Usage: $0 --strategy <ddp|fsdp|zero2|zero3> --world-size <N> [options]"
  exit 1
fi

if [[ "$WORLD_SIZE" -lt 2 ]]; then
  echo "ERROR: For multi-node, --world-size must be >= 2"
  echo "Use launch_smoke.sh for single-GPU testing"
  exit 1
fi

WORKER_COUNT=$((WORLD_SIZE - 1))

echo "=== Launching Distributed Training ==="
echo "Strategy:      $STRATEGY"
echo "World Size:    $WORLD_SIZE"
echo "Seq Length:    $SEQ_LEN"
echo "Tier:          $TIER"
echo "Steps:         $STEPS"
echo "Image:         $IMAGE"
echo "Namespace:     $NAMESPACE"
echo ""

# Create headless service (idempotent)
kubectl apply -f k8s/service-master.yaml

# Generate master job
cat k8s/job-master.template.yaml | \
  sed "s|{{STRATEGY}}|$STRATEGY|g" | \
  sed "s|{{WORLD_SIZE}}|$WORLD_SIZE|g" | \
  sed "s|{{SEQ_LEN}}|$SEQ_LEN|g" | \
  sed "s|{{TIER}}|$TIER|g" | \
  sed "s|{{STEPS}}|$STEPS|g" | \
  sed "s|{{PER_DEVICE_BATCH}}|$PER_DEVICE_BATCH|g" | \
  sed "s|{{GRAD_ACCUM}}|$GRAD_ACCUM|g" | \
  sed "s|{{IMAGE}}|$IMAGE|g" | \
  kubectl apply -f -

# Generate workers job
cat k8s/job-workers.template.yaml | \
  sed "s|{{STRATEGY}}|$STRATEGY|g" | \
  sed "s|{{WORLD_SIZE}}|$WORLD_SIZE|g" | \
  sed "s|{{WORKER_COUNT}}|$WORKER_COUNT|g" | \
  sed "s|{{SEQ_LEN}}|$SEQ_LEN|g" | \
  sed "s|{{TIER}}|$TIER|g" | \
  sed "s|{{STEPS}}|$STEPS|g" | \
  sed "s|{{PER_DEVICE_BATCH}}|$PER_DEVICE_BATCH|g" | \
  sed "s|{{GRAD_ACCUM}}|$GRAD_ACCUM|g" | \
  sed "s|{{IMAGE}}|$IMAGE|g" | \
  kubectl apply -f -

echo ""
echo "Jobs submitted. Monitor with:"
echo "  kubectl -n $NAMESPACE get jobs"
echo "  kubectl -n $NAMESPACE logs -f job/bench-master-$STRATEGY-ws$WORLD_SIZE-seq$SEQ_LEN"
echo "  kubectl -n $NAMESPACE logs -f job/bench-workers-$STRATEGY-ws$WORLD_SIZE-seq$SEQ_LEN"
