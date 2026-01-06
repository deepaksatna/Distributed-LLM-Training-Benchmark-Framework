#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1}"

echo "Pushing Docker image: $IMAGE_TAG"
echo "Ensure you are logged in: docker login fra.ocir.io"
echo ""

docker push "$IMAGE_TAG"

echo ""
echo "Push complete. Image available at: $IMAGE_TAG"
