# Example Benchmark Results

This directory contains example outputs from a successful benchmark run.

## Example Result JSON

```json
{
  "strategy": "ddp",
  "world_size": 2,
  "rank": 0,
  "seq_len": 2048,
  "tier": "A",
  "steps": 100,
  "per_device_batch": 1,
  "grad_accum": 4,
  "tokens_per_sec": 8369.455699192402,
  "mean_step_time_sec": 0.4893986117156026,
  "mean_loss": 6.133937082792583,
  "peak_vram_gb": 13.96808448,
  "h2d_gbps_per_gpu": 1.6738911398384803e-05
}
```

## Field Descriptions

| Field | Description | Unit |
|-------|-------------|------|
| `strategy` | Training strategy (ddp, fsdp, zero2, zero3) | - |
| `world_size` | Total number of GPUs used | - |
| `rank` | This process's rank (0 = master) | - |
| `seq_len` | Sequence length | tokens |
| `tier` | Resource tier (A = standard, B = high-memory) | - |
| `steps` | Number of training steps completed | - |
| `per_device_batch` | Batch size per GPU | - |
| `grad_accum` | Gradient accumulation steps | - |
| `tokens_per_sec` | **Throughput** | tokens/second |
| `mean_step_time_sec` | **Average step duration** | seconds |
| `mean_loss` | Training loss (for convergence check) | - |
| `peak_vram_gb` | **Maximum GPU memory used** | GB |
| `h2d_gbps_per_gpu` | Host-to-device transfer rate | GB/s |

## Expected Output Structure

After running `./scripts/run_all_benchmarks.sh`, you'll get:

```
results/
├── bench-master-ddp-ws2-seq2048.log
├── bench-master-ddp-ws2-seq2048_results/
│   └── result.json
├── bench-master-ddp-ws4-seq2048.log
├── bench-master-ddp-ws4-seq2048_results/
│   └── result.json
├── bench-master-fsdp-ws2-seq2048.log
├── bench-master-fsdp-ws2-seq2048_results/
│   └── result.json
├── bench-master-fsdp-ws4-seq2048.log
├── bench-master-fsdp-ws4-seq2048_results/
│   └── result.json
├── bench-master-zero2-ws2-seq2048.log
├── bench-master-zero2-ws2-seq2048_results/
│   └── result.json
├── bench-master-zero2-ws4-seq2048.log
├── bench-master-zero2-ws4-seq2048_results/
│   └── result.json
├── bench-master-zero3-ws2-seq2048.log
├── bench-master-zero3-ws2-seq2048_results/
│   └── result.json
├── bench-master-zero3-ws4-seq2048.log
├── bench-master-zero3-ws4-seq2048_results/
│   └── result.json
└── summary/
    ├── metrics.csv
    ├── BENCHMARK_REPORT.md
    └── plots/
        ├── tokens_per_sec_vs_gpu.png
        ├── step_time_vs_gpu.png
        ├── scaling_efficiency.png
        └── gbps_vs_gpu.png
```

## Sample CSV Output

`summary/metrics.csv` will contain:

```csv
strategy,world_size,rank,seq_len,tier,steps,per_device_batch,grad_accum,tokens_per_sec,mean_step_time_sec,mean_loss,peak_vram_gb,h2d_gbps_per_gpu,scaling_efficiency_pct
ddp,2,0,2048,A,100,1,4,8369.455699192402,0.4893986117156026,6.133937082792583,13.96808448,1.6738911398384803e-05,50.0
ddp,4,0,2048,A,100,1,4,12220.341463838564,0.6703577002525746,5.424101734161377,13.96808448,1.2220341463838563e-05,36.50279630794195
...
```

## How to Interpret Results

### Tokens per Second (Higher is Better)
- Measures training throughput
- Includes forward + backward + optimizer step
- DDP baseline: ~8,000-12,000 tokens/sec
- ZeRO-2 best: ~18,000 tokens/sec (48% improvement)

### Mean Step Time (Lower is Better)
- Time for one complete training iteration
- Includes all communication overhead
- Target: <0.5s for medium models
- DeepSpeed achieves ~0.37s vs DDP's ~0.49s

### Peak VRAM (Lower is Better)
- Maximum GPU memory during training
- Crucial for fitting larger models
- DDP: 13.97 GB (no sharding)
- ZeRO-3: 9.67 GB (31% reduction)

### Scaling Efficiency
- Measures how well performance scales with more GPUs
- Ideal: 100% (linear scaling)
- Realistic: 35-45% for 4 GPUs
- ZeRO-2 achieves 41.2% (best in class)

## Validation Checks

When reviewing results, verify:

✅ **Throughput is reasonable:**
- 2 GPUs: 6,000-11,000 tokens/sec (depending on strategy)
- 4 GPUs: 9,000-18,000 tokens/sec

✅ **Step time is consistent:**
- Should not vary by more than 10% between runs
- Watch for outliers

✅ **VRAM usage is sane:**
- Should be <24 GB (GPU limit)
- ZeRO strategies use less than DDP
- FSDP uses less than DDP but more than ZeRO-3

✅ **Loss is decreasing:**
- Not critical for benchmark (synthetic data)
- But should generally trend downward
- Typical range: 8-6 over 100 steps

✅ **Scaling efficiency makes sense:**
- 50% at 2 GPUs (baseline)
- 35-45% at 4 GPUs (good)
- <30% at 4 GPUs (investigate issues)

## Common Result Patterns

### Pattern 1: High Throughput, High VRAM
**Example:** DDP at 4 GPUs
- Tokens/sec: 12,220
- VRAM: 13.97 GB

**Interpretation:** Fast but memory-hungry. Good for small models that fit.

### Pattern 2: Medium Throughput, Low VRAM
**Example:** ZeRO-3 at 4 GPUs
- Tokens/sec: 15,977
- VRAM: 9.67 GB

**Interpretation:** Memory-efficient with good speed. Best for large models.

### Pattern 3: Highest Throughput, Medium VRAM
**Example:** ZeRO-2 at 4 GPUs
- Tokens/sec: 18,147
- VRAM: 10.47 GB

**Interpretation:** Sweet spot for production. Best all-around performance.

### Pattern 4: Low Throughput, Poor Scaling
**Example:** FSDP at 4 GPUs
- Tokens/sec: 9,424
- Scaling: 34.8%

**Interpretation:** High overhead, avoid for medium models. Better for 30B+ params.
