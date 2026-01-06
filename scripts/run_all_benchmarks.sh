#!/usr/bin/env bash
set -euo pipefail

# Comprehensive benchmark suite: Run all configurations and collect results automatically
# This script handles pod termination by collecting results immediately after each job completes

NAMESPACE="bench"
RESULTS_DIR="./results"
IMAGE="fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1"
TIMEOUT=900  # 15 minutes per job

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================================================="
echo "  Distributed Training Benchmark Suite"
echo "=================================================================="
echo "Namespace:    $NAMESPACE"
echo "Results dir:  $RESULTS_DIR"
echo "Timeout:      ${TIMEOUT}s per job"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Benchmark configurations
# Format: "STRATEGY WORLD_SIZE SEQ_LEN TIER STEPS"
BENCHMARKS=(
  # DDP scaling
  "ddp 2 2048 A 100"
  "ddp 4 2048 A 100"

  # FSDP scaling
  "fsdp 2 2048 A 100"
  "fsdp 4 2048 A 100"

  # DeepSpeed ZeRO-2
  "zero2 2 2048 A 100"
  "zero2 4 2048 A 100"

  # DeepSpeed ZeRO-3
  "zero3 2 2048 A 100"
  "zero3 4 2048 A 100"

  # Tier B stress tests (optional - comment out if not needed)
  # "fsdp 4 4096 B 100"
  # "zero3 4 8192 B 100"
)

TOTAL=${#BENCHMARKS[@]}
COMPLETED=0
FAILED=0

echo "Total benchmarks to run: $TOTAL"
echo ""

# Function to run a single benchmark
run_benchmark() {
  local strategy=$1
  local world_size=$2
  local seq_len=$3
  local tier=$4
  local steps=$5

  local job_name="bench-master-${strategy}-ws${world_size}-seq${seq_len}"
  local worker_job_name="bench-workers-${strategy}-ws${world_size}-seq${seq_len}"

  echo "=================================================================="
  echo -e "${BLUE}[$((COMPLETED + 1))/$TOTAL]${NC} Running: ${strategy} | WS=${world_size} | Seq=${seq_len} | Tier=${tier}"
  echo "=================================================================="

  # Launch the job
  echo "Launching jobs..."
  ./scripts/launch_multi.sh \
    --strategy "$strategy" \
    --world-size "$world_size" \
    --seq-len "$seq_len" \
    --tier "$tier" \
    --steps "$steps" \
    --image "$IMAGE"

  echo "Waiting for jobs to complete (timeout: ${TIMEOUT}s)..."

  # Wait for master job to complete
  if kubectl -n "$NAMESPACE" wait \
    --for=condition=complete \
    --timeout="${TIMEOUT}s" \
    "job/$job_name" 2>/dev/null; then

    echo -e "${GREEN}✓${NC} Master job completed successfully"

    # Wait for worker job to complete (if it exists)
    if kubectl -n "$NAMESPACE" get job "$worker_job_name" &>/dev/null; then
      echo "Waiting for worker job to complete..."
      if kubectl -n "$NAMESPACE" wait \
        --for=condition=complete \
        --timeout=60s \
        "job/$worker_job_name" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Worker job completed successfully"
      else
        echo -e "${YELLOW}⚠${NC} Worker job did not complete in time (continuing...)"
      fi
    fi

    # Give pods a moment to finalize
    sleep 5

    # Collect results from master
    echo "Collecting results from master..."
    if ./scripts/collect_results.sh "$NAMESPACE" "$job_name" "$RESULTS_DIR"; then
      echo -e "${GREEN}✓${NC} Results collected from master"
    else
      echo -e "${YELLOW}⚠${NC} Warning: Failed to collect results from master"
    fi

    # Try to collect worker logs too (for debugging)
    if kubectl -n "$NAMESPACE" get job "$worker_job_name" &>/dev/null; then
      echo "Collecting worker logs..."
      ./scripts/collect_results.sh "$NAMESPACE" "$worker_job_name" "$RESULTS_DIR" 2>/dev/null || true
    fi

    COMPLETED=$((COMPLETED + 1))
    echo -e "${GREEN}✓${NC} Benchmark completed successfully"

  else
    # Check if job failed
    if kubectl -n "$NAMESPACE" wait \
      --for=condition=failed \
      --timeout=10s \
      "job/$job_name" 2>/dev/null; then

      echo -e "${RED}✗${NC} Job failed!"

      # Try to get logs from failed pod
      echo "Fetching logs from failed pod..."
      kubectl -n "$NAMESPACE" logs "job/$job_name" --tail=100 || true

      FAILED=$((FAILED + 1))
    else
      echo -e "${RED}✗${NC} Job timed out after ${TIMEOUT}s"
      FAILED=$((FAILED + 1))
    fi
  fi

  # Cleanup jobs to free resources
  echo "Cleaning up jobs..."
  kubectl -n "$NAMESPACE" delete job "$job_name" --ignore-not-found=true
  kubectl -n "$NAMESPACE" delete job "$worker_job_name" --ignore-not-found=true

  echo ""
  sleep 3  # Brief pause between benchmarks
}

# Main execution loop
START_TIME=$(date +%s)

for benchmark in "${BENCHMARKS[@]}"; do
  read -r strategy world_size seq_len tier steps <<< "$benchmark"
  run_benchmark "$strategy" "$world_size" "$seq_len" "$tier" "$steps"
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Summary
echo "=================================================================="
echo "  Benchmark Suite Summary"
echo "=================================================================="
echo -e "Total benchmarks: $TOTAL"
echo -e "${GREEN}Completed:${NC}        $COMPLETED"
echo -e "${RED}Failed:${NC}           $FAILED"
echo -e "Duration:         $((DURATION / 60))m $((DURATION % 60))s"
echo ""

if [ "$FAILED" -gt 0 ]; then
  echo -e "${YELLOW}⚠ Some benchmarks failed. Check logs above.${NC}"
fi

# Generate analysis if we have results
if [ "$COMPLETED" -gt 0 ]; then
  echo "=================================================================="
  echo "  Generating Analysis"
  echo "=================================================================="

  # Parse results to CSV
  echo "Parsing results to CSV..."
  if python3 scripts/parse_metrics.py \
    --results-dir "$RESULTS_DIR" \
    --out "$RESULTS_DIR/summary"; then
    echo -e "${GREEN}✓${NC} Results parsed to CSV"
  else
    echo -e "${YELLOW}⚠${NC} Failed to parse results"
  fi

  # Generate plots
  if [ -f "$RESULTS_DIR/summary/metrics.csv" ]; then
    echo "Generating plots..."
    if python3 scripts/plot.py \
      --results "$RESULTS_DIR/summary/metrics.csv" \
      --out "$RESULTS_DIR/summary/plots"; then
      echo -e "${GREEN}✓${NC} Plots generated"
    else
      echo -e "${YELLOW}⚠${NC} Failed to generate plots"
    fi

    # Generate markdown report
    echo "Generating markdown report..."
    if python3 scripts/make_report.py \
      --csv "$RESULTS_DIR/summary/metrics.csv" \
      --out "$RESULTS_DIR/summary"; then
      echo -e "${GREEN}✓${NC} Report generated"
      echo ""
      echo "📊 Report available at: $RESULTS_DIR/summary/BENCHMARK_REPORT.md"
    else
      echo -e "${YELLOW}⚠${NC} Failed to generate report"
    fi
  fi
fi

echo "=================================================================="
echo -e "${GREEN}✓ Benchmark suite complete!${NC}"
echo "=================================================================="
echo "Results directory: $RESULTS_DIR"
echo ""

if [ -f "$RESULTS_DIR/summary/metrics.csv" ]; then
  echo "Next steps:"
  echo "  1. View report:  cat $RESULTS_DIR/summary/BENCHMARK_REPORT.md"
  echo "  2. View plots:   ls $RESULTS_DIR/summary/plots/"
  echo "  3. View CSV:     cat $RESULTS_DIR/summary/metrics.csv"
fi

exit 0
