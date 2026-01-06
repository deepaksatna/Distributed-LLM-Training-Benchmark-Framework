# Project Structure

This document explains the organization of the LLM Training Benchmark repository.

```
llm-training-benchmark/
│
├── README.md                          # Main documentation (START HERE)
├── LICENSE                            # MIT License
├── PROJECT_STRUCTURE.md               # This file
├── .gitignore                         # Git ignore rules
│
├── docs/                              # Additional documentation
│   ├── ARCHITECTURE.md                # System architecture deep-dive
│   └── TROUBLESHOOTING.md             # Common issues and solutions
│
├── benchmarking/                      # Core training code
│   └── train_harness.py               # Main training script (700+ lines)
│                                      # - TinyGPT model definition
│                                      # - Training loop with metrics
│                                      # - DDP/FSDP/ZeRO wrapper logic
│                                      # - Results export to JSON
│
├── configs/                           # Strategy configurations
│   ├── deepspeed/
│   │   ├── zero2.json                 # DeepSpeed ZeRO-2 config
│   │   └── zero3.json                 # DeepSpeed ZeRO-3 config
│   └── fsdp/
│       └── fsdp_config.yaml           # FSDP sharding strategy config
│
├── docker/                            # Container build files
│   ├── Dockerfile                     # Multi-stage Docker build
│   │                                  # - Base: CUDA 12.1 + cuDNN 8
│   │                                  # - PyTorch 2.1.0 + DeepSpeed
│   │                                  # - All dependencies offline
│   └── entrypoint.sh                  # Container startup script
│                                      # - Computes RANK from K8s index
│                                      # - Sets MASTER_ADDR correctly
│                                      # - Launches train_harness.py
│
├── k8s/                               # Kubernetes manifests
│   ├── namespace.yaml                 # Creates "bench" namespace
│   ├── serviceaccount.yaml            # RBAC for pods
│   ├── service-master.yaml            # ClusterIP for master discovery
│   ├── job-master.template.yaml       # Master (rank 0) job template
│   ├── job-workers.template.yaml      # Workers (rank 1..N) indexed job
│   ├── job-single.tmpl.yaml           # Generic single job template
│   └── job-smoke-1gpu.yaml            # Single-GPU smoke test
│
├── scripts/                           # Automation scripts
│   │
│   ├── build.sh                       # Build Docker image locally
│   ├── push.sh                        # Push image to OCIR
│   ├── verify_offline.sh              # Verify no external downloads
│   │
│   ├── launch_smoke.sh                # Run 1-GPU smoke test
│   ├── launch_multi.sh                # Launch distributed job (master + workers)
│   ├── run_all_benchmarks.sh          # 🚀 MAIN SCRIPT - Run all 8 benchmarks
│   │                                  # - Sequentially runs all configs
│   │                                  # - Collects results after each
│   │                                  # - Generates analysis at end
│   │
│   ├── collect_results.sh             # Extract results from pod logs
│   ├── install_analysis_deps.sh       # Install pandas, matplotlib, numpy
│   │
│   ├── parse_metrics.py               # JSON → CSV aggregation
│   ├── plot.py                        # Generate performance plots
│   ├── make_report.py                 # Generate markdown report
│   │
│   └── check_cluster_gpus.sh          # Verify GPU availability on nodes
│
├── images/                            # Performance visualizations
│   ├── tokens_per_sec_vs_gpu.png      # Throughput comparison
│   ├── step_time_vs_gpu.png           # Latency comparison
│   ├── scaling_efficiency.png         # Scaling analysis
│   └── gbps_vs_gpu.png                # Data transfer rates
│
└── results/                           # Benchmark outputs (gitignored)
    └── example_output/
        └── README.md                  # Example results documentation

    # After running benchmarks, structure will be:
    # results/
    # ├── bench-master-ddp-ws2-seq2048.log
    # ├── bench-master-ddp-ws2-seq2048_results/
    # │   └── result.json
    # ├── bench-master-ddp-ws4-seq2048.log
    # ├── bench-master-ddp-ws4-seq2048_results/
    # │   └── result.json
    # ├── ... (8 configurations total)
    # └── summary/
    #     ├── metrics.csv
    #     ├── BENCHMARK_REPORT.md
    #     └── plots/
    #         ├── tokens_per_sec_vs_gpu.png
    #         ├── step_time_vs_gpu.png
    #         ├── scaling_efficiency.png
    #         └── gbps_vs_gpu.png
```

## File Dependencies

### Build Phase
```
docker/Dockerfile
├── References: benchmarking/train_harness.py
├── References: configs/**/*.{json,yaml}
├── References: docker/entrypoint.sh
└── Produces: Docker image → OCIR
```

### Deploy Phase
```
k8s/namespace.yaml         (must exist)
k8s/serviceaccount.yaml    (must exist)
k8s/service-master.yaml    (must exist for multi-GPU)
│
scripts/launch_multi.sh
├── Reads: k8s/job-master.template.yaml
├── Reads: k8s/job-workers.template.yaml
├── Substitutes: IMAGE, WORLD_SIZE, STRATEGY, etc.
└── Creates: K8s jobs in cluster
```

### Training Phase
```
Pod starts
│
├── docker/entrypoint.sh
│   ├── Computes RANK from JOB_COMPLETION_INDEX
│   ├── Sets environment variables
│   └── Executes: python3 benchmarking/train_harness.py --args...
│
└── benchmarking/train_harness.py
    ├── Loads configs/deepspeed/*.json (if ZeRO)
    ├── Loads configs/fsdp/*.yaml (if FSDP)
    ├── Trains model
    └── Outputs JSON to stdout
```

### Collection Phase
```
scripts/collect_results.sh
├── Input: Namespace, Job name
├── Executes: kubectl logs <pod>
├── Extracts: JSON between markers
└── Writes: results/<job-name>_results/result.json
```

### Analysis Phase
```
scripts/parse_metrics.py
├── Input: results/ directory
├── Finds: All result.json files
├── Aggregates: Into pandas DataFrame
└── Writes: results/summary/metrics.csv

scripts/plot.py
├── Input: metrics.csv
├── Generates: 4 matplotlib plots
└── Writes: results/summary/plots/*.png

scripts/make_report.py
├── Input: metrics.csv
├── Generates: Markdown tables and analysis
└── Writes: results/summary/BENCHMARK_REPORT.md
```

## Key Files Explained

### train_harness.py (700+ lines)
**Purpose:** Core training script with model, strategies, and metrics.

**Sections:**
1. **Imports & Setup** (lines 1-50)
   - PyTorch, DeepSpeed, FSDP, NCCL
   - Argument parsing

2. **TinyGPT Model** (lines 51-150)
   - GPT-2 architecture
   - 117M parameters
   - Configurable layers, heads, embedding size

3. **Strategy Wrappers** (lines 151-280)
   - `wrap_model()` function
   - DDP, FSDP, ZeRO-2, ZeRO-3 logic
   - Fixed DeepSpeed config handling

4. **Training Loop** (lines 281-450)
   - Synthetic data generation
   - Forward/backward passes
   - Metrics collection (VRAM, timing, throughput)

5. **Results Export** (lines 451-500)
   - JSON formatting
   - Stdout output with markers
   - File save (for local runs)

6. **Main Entry Point** (lines 501-700)
   - Process group initialization
   - Argument validation
   - Strategy selection
   - Training execution

### entrypoint.sh
**Purpose:** Compute correct environment for distributed training.

**Critical Logic:**
```bash
# Compute RANK from Kubernetes indexed job
if [ -n "${JOB_COMPLETION_INDEX:-}" ]; then
  export RANK=$((JOB_COMPLETION_INDEX + 1))
fi

# Master uses its own IP, workers use service DNS
if [ "$RANK" = "0" ] && [ -n "${POD_IP:-}" ]; then
  export MASTER_ADDR="$POD_IP"
else
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
```

### run_all_benchmarks.sh
**Purpose:** Orchestrate complete benchmark suite.

**Workflow:**
1. Define 8 configurations (DDP/FSDP/ZeRO-2/ZeRO-3 × 2/4 GPUs)
2. For each config:
   - Launch jobs via `launch_multi.sh`
   - Wait for completion (max 15 min)
   - Collect results via `collect_results.sh`
   - Delete jobs for cleanup
3. After all benchmarks:
   - Parse to CSV
   - Generate plots
   - Create report

### collect_results.sh
**Purpose:** Extract JSON from logs even after pod termination.

**Key Innovation:**
```bash
# Works even after pod terminates!
kubectl logs $POD_NAME > log.txt

# Extract JSON between markers
sed -n '/BENCHMARK_RESULT_JSON_START/,/BENCHMARK_RESULT_JSON_END/p' log.txt | \
  sed '1d;$d' > result.json
```

## Configuration Flow

### Strategy Selection
```
User runs:
./scripts/run_all_benchmarks.sh

└──▶ For each benchmark:
     ./scripts/launch_multi.sh --strategy ddp --world-size 2 ...

     └──▶ Creates job with env:
          STRATEGY=ddp
          WORLD_SIZE=2

          └──▶ entrypoint.sh passes to:
               python3 train_harness.py --strategy ddp --world-size 2

               └──▶ train_harness.py:
                    if args.strategy == "ddp":
                        model = DDP(model, ...)
                    elif args.strategy == "fsdp":
                        model = FSDP(model, ...)
                    elif args.strategy in ["zero2", "zero3"]:
                        model, opt = deepspeed.initialize(...)
```

### Config File Loading
```
DeepSpeed ZeRO-2:
args.strategy = "zero2"
args.deepspeed_config = "/app/configs/deepspeed/zero2.json"
└──▶ train_harness.py loads and modifies JSON
     └──▶ Sets batch sizes as integers (critical fix!)

DeepSpeed ZeRO-3:
args.strategy = "zero3"
args.deepspeed_config = "/app/configs/deepspeed/zero3.json"
└──▶ Same process

FSDP:
Uses in-code configuration (transformer_auto_wrap_policy)
Optional: Can load from configs/fsdp/fsdp_config.yaml
```

## Where to Start

### For Running Benchmarks:
1. Read: `README.md` (quick start section)
2. Edit: `scripts/push.sh` (your OCIR details)
3. Run: `./scripts/build.sh && ./scripts/push.sh`
4. Run: `./scripts/run_all_benchmarks.sh`

### For Understanding Implementation:
1. Read: `docs/ARCHITECTURE.md`
2. Read: `benchmarking/train_harness.py`
3. Read: `docker/entrypoint.sh`
4. Read: `scripts/run_all_benchmarks.sh`

### For Debugging Issues:
1. Read: `docs/TROUBLESHOOTING.md`
2. Check: Pod logs via `kubectl logs`
3. Verify: Environment variables in `entrypoint.sh`

### For Modifying Strategies:
1. Edit: `benchmarking/train_harness.py` (wrap_model function)
2. Edit: `configs/deepspeed/*.json` (ZeRO configs)
3. Rebuild: `./scripts/build.sh && ./scripts/push.sh`

### For Adding New GPUs (A100, H100):
1. Read: `README.md` (Supported GPU Platforms section)
2. Edit: Batch sizes, sequence lengths in configs
3. Test: Run smoke test first `./scripts/launch_smoke.sh`

## Quick Reference

| Task | File | Command |
|------|------|---------|
| Build image | `scripts/build.sh` | `./scripts/build.sh` |
| Push to OCIR | `scripts/push.sh` | `./scripts/push.sh` |
| Run all benchmarks | `scripts/run_all_benchmarks.sh` | `./scripts/run_all_benchmarks.sh` |
| Run smoke test | `scripts/launch_smoke.sh` | `./scripts/launch_smoke.sh` |
| Test single config | `scripts/launch_multi.sh` | `./scripts/launch_multi.sh --strategy ddp --world-size 2` |
| Collect results | `scripts/collect_results.sh` | `./scripts/collect_results.sh bench job-name ./results` |
| Generate plots | `scripts/plot.py` | `python3 scripts/plot.py --results metrics.csv --out plots/` |
| View logs | - | `kubectl logs <pod> -n bench` |
| Debug pod | - | `kubectl exec -it <pod> -n bench -- bash` |

---

**Last Updated:** January 6, 2026
**Maintained By:** Oracle AI CoE
