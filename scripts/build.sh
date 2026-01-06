#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1}"

echo "Building Docker image: $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f docker/Dockerfile .

echo "Build complete. Run './scripts/push.sh $IMAGE_TAG' to push to OCIR."
