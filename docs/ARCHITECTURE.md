# Architecture Documentation

## System Architecture

### Overview

This benchmark framework follows a **containerized, Kubernetes-native architecture** designed for distributed LLM training on Oracle Cloud Infrastructure (OCI).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Benchmark Framework Architecture                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Build Phase    │────▶│  Deploy Phase   │────▶│  Execute Phase  │
│  (Non-GPU VM)   │     │  (OKE)          │     │  (GPU Pods)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                         │
        ▼                       ▼                         ▼
    Dockerfile              K8s Jobs               Training Loop
    + Dependencies          + RBAC                 + Metrics
    + Model Code            + Service              + Results
        │                       │                         │
        ▼                       ▼                         ▼
    Push to OCIR           Schedule Pods          Export JSON
                           Assign GPUs            to Logs
                                                         │
                                                         ▼
                                          ┌──────────────────────┐
                                          │  Analysis Phase      │
                                          │  (Python Scripts)    │
                                          └──────────────────────┘
                                                         │
                                                         ▼
                                            CSV + Plots + Report
```

## Component Details

### 1. Container Image

**Purpose:** Package all dependencies, code, and models for offline execution.

**Structure:**
```
Container Image (fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1)
├── Base: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
├── Python 3.10
├── PyTorch 2.1.0 + CUDA 12.1
├── DeepSpeed 0.14.0
├── NCCL 2.18.5
├── Application Code:
│   ├── /app/benchmarking/train_harness.py (main training script)
│   ├── /app/configs/ (strategy configs)
│   └── /app/docker/entrypoint.sh (startup script)
└── No External Dependencies (100% offline)
```

**Build Process:**
```bash
docker build -t mistraltraining:latest -f docker/Dockerfile .
docker tag mistraltraining:latest fra.ocir.io/.../models:mistraltraining-fsdp-zero-v1
docker push fra.ocir.io/.../models:mistraltraining-fsdp-zero-v1
```

### 2. Kubernetes Resources

#### Namespace and RBAC
```yaml
namespace.yaml        # Isolated namespace "bench"
serviceaccount.yaml   # Service account for pod permissions
```

#### Service (for distributed training)
```yaml
service-master.yaml   # ClusterIP service for master discovery
  - Enables workers to connect to master
  - DNS: bench-master.bench.svc.cluster.local
  - Type: ClusterIP (not headless, for better DNS stability)
```

#### Job Templates
```yaml
job-master.template.yaml    # Rank 0 (master) job
job-workers.template.yaml   # Rank 1..N-1 (workers) as indexed job
job-smoke-1gpu.yaml         # Single-GPU smoke test
```

### 3. Distributed Training Coordination

#### Process Group Initialization

```python
# Executed by each pod
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',           # NVIDIA Collective Communications Library
    init_method='env://',     # Use environment variables
    world_size=args.world_size,  # Total number of GPUs
    rank=args.rank            # This pod's rank (0 = master)
)
```

#### Environment Variables

**Master Pod (Rank 0):**
```bash
RANK=0
WORLD_SIZE=4
MASTER_ADDR=$POD_IP          # Master uses its own IP
MASTER_PORT=29500
LOCAL_RANK=0
NCCL_SOCKET_IFNAME=eth0
```

**Worker Pods (Rank 1..N-1):**
```bash
RANK=$((JOB_COMPLETION_INDEX + 1))  # Computed in entrypoint.sh
WORLD_SIZE=4
MASTER_ADDR=<master-pod-ip>         # From service discovery
MASTER_PORT=29500
LOCAL_RANK=0                         # Each pod has 1 GPU
NCCL_SOCKET_IFNAME=eth0
```

### 4. Training Workflow

#### Step-by-Step Execution

```
1. Pod Startup
   ├── entrypoint.sh executes
   ├── Compute RANK from JOB_COMPLETION_INDEX
   ├── Set MASTER_ADDR (master uses POD_IP, workers use service DNS)
   └── Execute train_harness.py

2. Process Group Init
   ├── All pods call dist.init_process_group()
   ├── NCCL establishes communication channels
   ├── Synchronization barrier
   └── Ready for training

3. Model Setup
   ├── Load TinyGPT model definition (in code, no downloads)
   ├── Move model to GPU
   ├── Wrap with strategy (DDP/FSDP/DeepSpeed)
   └── Create optimizer

4. Training Loop
   ├── Generate synthetic data (no I/O)
   ├── Forward pass
   ├── Loss computation
   ├── Backward pass (gradients computed)
   ├── Optimizer step (with gradient synchronization)
   └── Collect metrics (time, VRAM, throughput)

5. Results Export
   ├── Rank 0 aggregates metrics
   ├── Calculate tokens/sec, mean step time, peak VRAM
   ├── Export to JSON
   ├── Print JSON to stdout with markers
   └── Pod completes successfully

6. Log Collection
   ├── collect_results.sh fetches logs
   ├── Extract JSON from logs using sed
   ├── Save to results/<job-name>_results/result.json
   └── Cleanup K8s job
```

### 5. Strategy Implementations

#### DDP (Distributed Data Parallel)
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
)

# Behavior:
# - Full model replica on each GPU
# - AllReduce gradients after backward pass
# - Synchronous updates
# - No memory savings
```

#### FSDP (Fully Sharded Data Parallel)
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    auto_wrap_policy=transformer_auto_wrap_policy
)

# Behavior:
# - Shards parameters, gradients, optimizer states
# - Gathers parameters for forward/backward
# - Memory: O(model_size / world_size)
# - Higher communication than DDP
```

#### DeepSpeed ZeRO-2
```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": micro * accum * world_size,
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": accum,
        "zero_optimization": {"stage": 2}
    }
)

# Behavior:
# - Shards optimizer states + gradients
# - Parameters replicated (like DDP)
# - Lower communication than ZeRO-3
# - Memory savings from optimizer sharding
```

#### DeepSpeed ZeRO-3
```python
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_max_live_parameters": 1e9
        }
    }
)

# Behavior:
# - Shards parameters + gradients + optimizer states
# - Maximum memory efficiency
# - Higher communication (parameter gathering)
# - Enables largest models
```

### 6. Networking Architecture

#### Pod Networking

```
┌────────────────────────────────────────────────────────────┐
│  Node 1 (10.0.10.101)                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pod: bench-master (hostNetwork=true)                │  │
│  │  IP: 10.0.10.101 (uses host IP)                      │  │
│  │  RANK: 0                                             │  │
│  │  GPU: 0                                              │  │
│  │  MASTER_ADDR: $POD_IP (10.0.10.101)                 │  │
│  │  Listening on: 0.0.0.0:29500                        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                          ▲
                          │ NCCL Communication
                          │ (RDMA if available)
                          ▼
┌────────────────────────────────────────────────────────────┐
│  Node 2 (10.0.10.102)                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pod: bench-workers-0 (hostNetwork=true)             │  │
│  │  IP: 10.0.10.102                                     │  │
│  │  RANK: 1 (from JOB_COMPLETION_INDEX=0)              │  │
│  │  GPU: 0                                              │  │
│  │  MASTER_ADDR: 10.0.10.101                           │  │
│  │  Connects to: 10.0.10.101:29500                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**Why hostNetwork mode:**
- Direct pod-to-pod communication (no overlay network overhead)
- Lower latency for NCCL operations
- Better RDMA support
- Avoids IPv6 DNS resolution issues

#### NCCL Communication Patterns

**DDP & ZeRO-2:**
```
Step 1: Forward pass (no communication)
Step 2: Backward pass (compute gradients)
Step 3: AllReduce gradients across all ranks
        ┌─────────┐
        │ Rank 0  │──┐
        └─────────┘  │
                     ├──▶ AllReduce ──▶ Synchronized Gradients
        ┌─────────┐  │
        │ Rank 1  │──┘
        └─────────┘
Step 4: Optimizer step (locally)
```

**FSDP & ZeRO-3:**
```
Forward Pass:
  ┌─────────┐       ┌─────────┐
  │ Rank 0  │◀─────▶│ Rank 1  │  AllGather parameters
  └─────────┘       └─────────┘

Backward Pass:
  ┌─────────┐       ┌─────────┐
  │ Rank 0  │◀─────▶│ Rank 1  │  AllGather params + ReduceScatter gradients
  └─────────┘       └─────────┘

Optimizer Step:
  Each rank updates its shard locally
```

### 7. Results Collection Architecture

#### Challenge: Ephemeral Pod Storage

```
Problem:
┌────────────────────┐
│  Pod Running       │
│  ┌──────────────┐  │
│  │ EmptyDir:    │  │
│  │ /results/    │  │  ← Results saved here
│  │ result.json  │  │
│  └──────────────┘  │
└────────────────────┘
         │
         │ Pod Completes
         ▼
┌────────────────────┐
│  Pod Terminated    │
│  EmptyDir: GONE!   │  ← All files lost!
└────────────────────┘
```

#### Solution: Log-Based Collection

```
New Approach:
┌────────────────────┐
│  Pod Running       │
│  1. Save JSON      │
│  2. Print to stdout│  ← JSON goes to logs
│     with markers   │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Kubernetes Logs   │  ← Persist after pod termination
│  (Retained)        │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  collect_results.sh│
│  kubectl logs ...  │  ← Extract JSON from logs
│  sed -n '/START/,  │
│  /END/p'          │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│  Local filesystem  │
│  result.json       │  ← Permanent storage
└────────────────────┘
```

### 8. Automation Architecture

#### run_all_benchmarks.sh Workflow

```
┌──────────────────────────────────────────────────────────┐
│  Benchmark Suite Orchestration                           │
└──────────────────────────────────────────────────────────┘

for each configuration in [ddp-2gpu, ddp-4gpu, fsdp-2gpu, ...]:
    │
    ├─▶ 1. launch_multi.sh
    │   ├── Create K8s jobs (master + workers)
    │   └── Jobs start pods
    │
    ├─▶ 2. kubectl wait --for=condition=complete
    │   └── Block until jobs finish
    │
    ├─▶ 3. collect_results.sh
    │   ├── kubectl logs <master-pod> > log.txt
    │   ├── Extract JSON from logs
    │   └── Save to results/<job-name>_results/result.json
    │
    ├─▶ 4. kubectl delete job <job-name>
    │   └── Cleanup for next iteration
    │
    └─▶ Next configuration

After all benchmarks:
    │
    ├─▶ 5. parse_metrics.py
    │   ├── Find all result.json files
    │   ├── Aggregate to pandas DataFrame
    │   └── Export to metrics.csv
    │
    ├─▶ 6. plot.py
    │   ├── Load metrics.csv
    │   ├── Generate 4 plots (matplotlib)
    │   └── Save to summary/plots/
    │
    └─▶ 7. make_report.py
        ├── Load metrics.csv
        ├── Generate markdown tables
        └── Save BENCHMARK_REPORT.md
```

## Scalability Considerations

### Horizontal Scaling (More GPUs)

**Current:** 4 GPUs (4 nodes × 1 GPU)
**Tested:** 2 GPUs, 4 GPUs
**Extendable to:** 8, 16, 32+ GPUs

**Changes needed for 8+ GPUs:**
```python
# Increase timeout in run_all_benchmarks.sh
TIMEOUT=1800  # 30 minutes for larger runs

# Adjust batch size to maintain constant global batch
per_device_batch = global_batch // world_size // grad_accum

# Consider gradient compression
"compression_training": {
    "enabled": true,
    "method": "1-bit-adam"
}
```

### Vertical Scaling (Larger Models)

**Current:** 117M parameters (TinyGPT)
**Extendable to:** 7B, 13B, 30B, 70B+

**Changes needed for 7B model:**
```python
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = 32      # (was 12)
        self.n_head = 32       # (was 12)
        self.n_embd = 4096     # (was 768)
        # Result: ~7B parameters

# Use ZeRO-3 for memory efficiency
strategy = "zero3"
per_device_batch = 1  # May need to decrease
enable_cpu_offload = True  # For 70B+ models
```

## Performance Optimization

### NCCL Tuning

**For A10 VMs (Current Configuration):**
```bash
# Optimal settings for A10 VMs without RDMA
export NCCL_SOCKET_IFNAME=eth0     # Standard Ethernet interface
export NCCL_IB_DISABLE=1           # Disable InfiniBand (not available)
export NCCL_P2P_LEVEL=SYS          # System-level P2P
export NCCL_ALGO=Ring              # Ring algorithm for AllReduce
export NCCL_DEBUG=WARN             # Logging level
```

**For A100/H100/H200 with RDMA (Advanced Configuration):**
```bash
# Optimal settings for bare metal shapes with RDMA
export NCCL_SOCKET_IFNAME=ib0      # InfiniBand interface
export NCCL_IB_DISABLE=0           # Enable InfiniBand/RDMA
export NCCL_NET_GDR_LEVEL=5        # GPU Direct RDMA
export NCCL_P2P_LEVEL=SYS          # System-level P2P
export NCCL_ALGO=Ring              # Ring for small clusters, Tree for large
export NCCL_NVLINK_ENABLE=1        # Enable NVLink (H100/H200)
export NCCL_DEBUG=INFO             # More verbose for RDMA debugging

# Additional tuning for H100/H200
export NCCL_COLLNET_ENABLE=1       # Enable SHARP (if available)
export NCCL_TOPO_FILE=/etc/nccl/topology.xml  # Custom topology
```

**⚠️ CRITICAL:** Using the wrong NCCL configuration can cause:
- Severe performance degradation (10x slower)
- Communication timeouts
- Training failures

Always verify your GPU shape capabilities before setting NCCL parameters!

### Memory Optimization

```python
# Enable activation checkpointing
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Trade compute for memory
    return checkpoint(self.layer, x)

# Enable gradient checkpointing in configs
"activation_checkpointing": {
    "enabled": true,
    "partition_activations": true,
    "cpu_checkpointing": false  # Keep on GPU if fits
}
```

### Computation Optimization

```python
# PyTorch 2.0+ compilation
model = torch.compile(model, mode="max-autotune")

# Flash Attention for long sequences
from flash_attn import flash_attn_func

# FP8 on H100
from transformer_engine.pytorch import Linear as te_Linear
```

## Monitoring and Observability

### Metrics Collected

```python
metrics = {
    "tokens_per_sec": <float>,
    "mean_step_time_sec": <float>,
    "peak_vram_gb": <float>,
    "h2d_gbps_per_gpu": <float>,
    "mean_loss": <float>,
    "scaling_efficiency_pct": <float>
}
```

### Recommended Additions

```python
# GPU utilization
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)

# Communication time breakdown
import torch.distributed as dist
with dist.profile():
    # Training code
    pass

# Memory timeline
torch.cuda.memory._record_memory_history()
```

---

**Last Updated:** January 6, 2026
**Maintained By:** Oracle AI CoE
