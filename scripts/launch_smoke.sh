#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1}"
NAMESPACE="bench"

echo "=== Launching Smoke Test (1 GPU) ==="
echo "Image: $IMAGE"
echo "Namespace: $NAMESPACE"
echo ""

# Update image in smoke test job
kubectl create -f k8s/job-smoke-1gpu.yaml --dry-run=client -o yaml | \
  sed "s|image:.*|image: $IMAGE|" | \
  kubectl apply -f -

echo ""
echo "Job submitted. Monitor with:"
echo "  kubectl -n $NAMESPACE get jobs"
echo "  kubectl -n $NAMESPACE logs -f job/smoke-1gpu"
