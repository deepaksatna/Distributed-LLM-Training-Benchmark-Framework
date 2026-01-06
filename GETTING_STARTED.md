# Getting Started with LLM Training Benchmark

This quick start guide will get you running benchmarks in under 30 minutes.

## Prerequisites Checklist

- [ ] OKE cluster with GPU nodes (A10, A100, or H100)
- [ ] `kubectl` configured to access your cluster
- [ ] Docker installed on build machine
- [ ] OCIR credentials (tenancy, username, auth token)
- [ ] Python 3.10+ for analysis scripts

## Step-by-Step Setup

### 1. Clone and Configure (5 minutes)

```bash
# Clone repository
git clone https://github.com/deepaksatna/Distributed-LLM-Training-Benchmark-Framework.git
cd Distributed-LLM-Training-Benchmark-Framework

# Update OCIR settings
vim scripts/push.sh
# Change: REGISTRY, NAMESPACE to match your OCI tenancy

vim scripts/run_all_benchmarks.sh
# Change: IMAGE variable to your OCIR path
```

### 2. Build and Push Image (10 minutes)

```bash
# Build Docker image
./scripts/build.sh

# Login to OCIR
docker login fra.ocir.io
# Username: <tenancy-namespace>/<username>
# Password: <auth-token>

# Push to registry
./scripts/push.sh

# Verify
docker pull fra.ocir.io/your-namespace/models:mistraltraining-fsdp-zero-v1
```

### 3. Setup Kubernetes (2 minutes)

```bash
# Verify cluster access
kubectl get nodes

# Create resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/service-master.yaml

# Verify
kubectl get ns bench
kubectl get sa -n bench
```

### 4. Run Smoke Test (2 minutes)

```bash
# Test single GPU
./scripts/launch_smoke.sh

# Watch progress
kubectl get pods -n bench -w

# Check logs
kubectl logs job/smoke-1gpu -n bench

# Clean up
kubectl delete job smoke-1gpu -n bench
```

### 5. Install Analysis Tools (1 minute)

```bash
# On your local machine or OKE VM
pip3 install --user pandas matplotlib numpy

# Or use provided script
./scripts/install_analysis_deps.sh
```

### 6. Run Full Benchmark Suite (12-15 minutes)

```bash
# Run all 8 configurations
./scripts/run_all_benchmarks.sh

# Monitor progress
tail -f benchmark_run.log

# Or in foreground (recommended for first run)
./scripts/run_all_benchmarks.sh
```

### 7. View Results (1 minute)

```bash
# View report
cat results/summary/BENCHMARK_REPORT.md

# View CSV
cat results/summary/metrics.csv

# View plots
open results/summary/plots/tokens_per_sec_vs_gpu.png  # macOS
xdg-open results/summary/plots/tokens_per_sec_vs_gpu.png  # Linux
```

## Expected Output

### Console Output During Benchmark

```
==================================================================
  Distributed Training Benchmark Suite
==================================================================
Namespace:    bench
Results dir:  ./results
Timeout:      900s per job

Total benchmarks to run: 8

==================================================================
[1/8] Running: ddp | WS=2 | Seq=2048 | Tier=A
==================================================================
Launching jobs...
Waiting for jobs to complete (timeout: 900s)...
✓ Master job completed successfully
✓ Worker job completed successfully
Collecting results from master...
✓ Results collected from master
✓ Benchmark completed successfully

==================================================================
[2/8] Running: ddp | WS=4 | Seq=2048 | Tier=A
==================================================================
...

==================================================================
  Benchmark Suite Summary
==================================================================
Total benchmarks: 8
Completed:        8
Failed:           0
Duration:         12m 39s

==================================================================
  Generating Analysis
==================================================================
Parsing results to CSV...
✓ Results parsed to CSV
Generating plots...
✓ Plots generated
Generating markdown report...
✓ Report generated

📊 Report available at: ./results/summary/BENCHMARK_REPORT.md

==================================================================
✓ Benchmark suite complete!
==================================================================
```

### Generated Files

```
results/
├── bench-master-ddp-ws2-seq2048.log
├── bench-master-ddp-ws2-seq2048_results/result.json
├── bench-master-ddp-ws4-seq2048.log
├── bench-master-ddp-ws4-seq2048_results/result.json
├── bench-master-fsdp-ws2-seq2048.log
├── bench-master-fsdp-ws2-seq2048_results/result.json
├── bench-master-fsdp-ws4-seq2048.log
├── bench-master-fsdp-ws4-seq2048_results/result.json
├── bench-master-zero2-ws2-seq2048.log
├── bench-master-zero2-ws2-seq2048_results/result.json
├── bench-master-zero2-ws4-seq2048.log
├── bench-master-zero2-ws4-seq2048_results/result.json
├── bench-master-zero3-ws2-seq2048.log
├── bench-master-zero3-ws2-seq2048_results/result.json
├── bench-master-zero3-ws4-seq2048.log
├── bench-master-zero3-ws4-seq2048_results/result.json
└── summary/
    ├── metrics.csv
    ├── BENCHMARK_REPORT.md
    └── plots/
        ├── tokens_per_sec_vs_gpu.png
        ├── step_time_vs_gpu.png
        ├── scaling_efficiency.png
        └── gbps_vs_gpu.png
```

## What to Expect: Performance Baselines

### On NVIDIA A10 (24GB)

| Strategy | 2 GPU | 4 GPU | Notes |
|----------|-------|-------|-------|
| DDP      | ~8,400 tok/s | ~12,200 tok/s | Baseline, reliable |
| FSDP     | ~6,800 tok/s | ~9,400 tok/s | Poor scaling |
| ZERO2    | ~11,000 tok/s | ~18,100 tok/s | **Best throughput** |
| ZERO3    | ~10,600 tok/s | ~16,000 tok/s | Best memory |

### On NVIDIA A100 (40GB or 80GB)

Expected improvements:
- 2-3x higher throughput
- Can use batch_size 2-4 instead of 1
- Longer sequences (4096, 8192)

### On NVIDIA H100 (80GB)

Expected improvements:
- 4-6x higher throughput vs A10
- FP8 support (Transformer Engine)
- Sequence lengths up to 16K+

## Common First-Run Issues

### Issue: ImagePullBackOff

```bash
# Check image exists
docker images | grep mistraltraining

# Verify OCIR path
kubectl describe pod <pod> -n bench | grep Image

# Create image pull secret if needed
kubectl create secret docker-registry ocir-secret \
  --docker-server=fra.ocir.io \
  --docker-username='<tenancy>/<user>' \
  --docker-password='<auth-token>' \
  -n bench
```

### Issue: Pods Pending (No GPUs)

```bash
# Check GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# Check if GPUs are allocated to other pods
kubectl get pods -A -o json | \
  jq '.items[] | select(.spec.containers[].resources.requests."nvidia.com/gpu" != null)'

# Clean up old jobs
kubectl delete jobs --all -n bench
```

### Issue: NCCL Timeout

```bash
# Check network connectivity
kubectl exec -it <worker-pod> -n bench -- ping <master-ip>

# Verify hostNetwork mode is enabled
kubectl get pod <pod> -n bench -o yaml | grep hostNetwork

# Should output: hostNetwork: true
```

## Next Steps After First Run

1. **Compare Strategies**
   - Review `results/summary/BENCHMARK_REPORT.md`
   - Identify best strategy for your use case
   - Note VRAM usage for future model sizing

2. **Optimize Configuration**
   - Adjust batch sizes based on VRAM headroom
   - Test longer sequences if memory allows
   - Enable activation checkpointing for larger models

3. **Test with Real Models**
   - Replace TinyGPT with your model
   - Load from HuggingFace or local checkpoint
   - Maintain same benchmark infrastructure

4. **Scale Up**
   - Test with 8, 16, or 32 GPUs
   - Benchmark A100 or H100 if available
   - Compare cost-effectiveness across SKUs

5. **Production Deployment**
   - Use winning strategy (likely ZeRO-2)
   - Add checkpointing for fault tolerance
   - Integrate with MLOps tools (W&B, TensorBoard)

## Customization Examples

### Change Model Size

```python
# In benchmarking/train_harness.py
class TinyGPT(nn.Module):
    def __init__(self):
        self.n_layer = 24     # Instead of 12
        self.n_embd = 1024    # Instead of 768
        # Results in ~350M params instead of 117M
```

### Change Sequence Length

```bash
# In scripts/run_all_benchmarks.sh
"ddp 2 4096 A 100"  # seq_len=4096 instead of 2048
```

### Add New Benchmark Config

```bash
# In scripts/run_all_benchmarks.sh, add to BENCHMARKS array:
"zero2 8 2048 A 100"  # Test with 8 GPUs
"zero3 4 8192 B 100"  # Long sequence, tier B
```

### Test Different Batch Sizes

```bash
./scripts/launch_multi.sh \
  --strategy zero2 \
  --world-size 4 \
  --per-device-batch 4 \  # Instead of 1
  --grad-accum 2          # Instead of 4
```

## Performance Tuning Tips

1. **For Maximum Throughput**
   - Use DeepSpeed ZeRO-2
   - Increase batch size to fill VRAM
   - Use BF16 or FP8 (H100)
   - Enable Flash Attention for long sequences

2. **For Minimum Memory**
   - Use DeepSpeed ZeRO-3
   - Enable activation checkpointing
   - Consider CPU offloading for 70B+ models
   - Reduce batch size

3. **For Best Scaling**
   - Use ZeRO-2 or ZeRO-3
   - Ensure RDMA is enabled
   - Use hostNetwork mode
   - Optimize NCCL settings

4. **For Debugging**
   - Start with single GPU smoke test
   - Then 2 GPUs, then 4
   - Enable NCCL_DEBUG=INFO
   - Check logs frequently

## Support Resources

- **Documentation:** See `README.md`, `docs/ARCHITECTURE.md`
- **Troubleshooting:** See `docs/TROUBLESHOOTING.md`
- **Example Results:** See `results/example_output/README.md`
- **Issues:** GitHub Issues (when repo is public)
- **Contact:** deep.soni@oracle.com

---

**Ready to benchmark? Start with Step 1 above!**

Last Updated: January 6, 2026
