#!/usr/bin/env bash
set -euo pipefail

# Usage: ./collect_results.sh <namespace> <job-name> <output-dir>

NAMESPACE="${1:-bench}"
JOB_NAME="${2:-smoke-1gpu}"
OUTPUT_DIR="${3:-./results}"

mkdir -p "$OUTPUT_DIR"

echo "=== Collecting Results ==="
echo "Namespace:   $NAMESPACE"
echo "Job:         $JOB_NAME"
echo "Output:      $OUTPUT_DIR"
echo ""

# Get pod name from job
POD_NAME=$(kubectl -n "$NAMESPACE" get pods \
  --selector=job-name="$JOB_NAME" \
  --output=jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$POD_NAME" ]]; then
  echo "ERROR: No pod found for job $JOB_NAME"
  exit 1
fi

echo "Pod: $POD_NAME"

# Check pod status
POD_PHASE=$(kubectl -n "$NAMESPACE" get pod "$POD_NAME" -o jsonpath='{.status.phase}')
echo "Pod phase: $POD_PHASE"

if [[ "$POD_PHASE" != "Succeeded" && "$POD_PHASE" != "Running" && "$POD_PHASE" != "Failed" ]]; then
  echo "WARNING: Pod is in $POD_PHASE state"
fi

# Collect logs
echo ""
echo "Collecting logs..."
LOG_FILE="$OUTPUT_DIR/${JOB_NAME}.log"
kubectl -n "$NAMESPACE" logs "$POD_NAME" > "$LOG_FILE" 2>&1 || true

# Extract JSON results from logs
echo "Extracting JSON results from logs..."
JSON_DIR="$OUTPUT_DIR/${JOB_NAME}_results"
mkdir -p "$JSON_DIR"

# Extract content between markers
if grep -q "BENCHMARK_RESULT_JSON_START" "$LOG_FILE"; then
  sed -n '/BENCHMARK_RESULT_JSON_START/,/BENCHMARK_RESULT_JSON_END/p' "$LOG_FILE" | \
    sed '1d;$d' > "$JSON_DIR/result.json"

  if [ -s "$JSON_DIR/result.json" ]; then
    echo "✓ Extracted JSON results to $JSON_DIR/result.json"
  else
    echo "⚠ JSON extraction produced empty file"
    rm -f "$JSON_DIR/result.json"
  fi
else
  echo "⚠ No JSON results found in logs (job may have failed)"
fi

echo ""
echo "Collection complete. Files in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | tail -n +2
