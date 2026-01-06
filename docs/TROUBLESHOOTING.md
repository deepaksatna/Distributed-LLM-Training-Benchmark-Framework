# Troubleshooting Guide

This document covers common issues encountered during benchmarking and their solutions.

## Table of Contents

- [Build Issues](#build-issues)
- [Deployment Issues](#deployment-issues)
- [Training Issues](#training-issues)
- [Results Collection Issues](#results-collection-issues)
- [Performance Issues](#performance-issues)

---

## Build Issues

### Issue 1: Docker Build Fails on Non-GPU VM

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch+cu121
```

**Cause:** Trying to install GPU version of PyTorch without proper index URL.

**Solution:**
```dockerfile
# In Dockerfile, use the correct index URL
RUN pip3 install --no-cache-dir torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: Image Size Too Large

**Symptoms:**
```
Image size: 15GB+
Push to OCIR takes 30+ minutes
```

**Solution:**
```dockerfile
# Use --no-cache-dir to reduce size
RUN pip3 install --no-cache-dir <package>

# Multi-stage build
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
```

### Issue 3: CUDA Version Mismatch

**Symptoms:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**
```bash
# Check CUDA version on GPU nodes
kubectl exec -it <pod> -- nvidia-smi

# Use matching CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04  # For CUDA 11.8
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04  # For CUDA 12.1
```

---

## Deployment Issues

### Issue 1: ImagePullBackOff

**Symptoms:**
```
kubectl get pods
NAME                     READY   STATUS             RESTARTS   AGE
bench-master-xxx         0/1     ImagePullBackOff   0          2m
```

**Diagnosis:**
```bash
kubectl describe pod bench-master-xxx | grep -A 10 Events
```

**Solutions:**

**A. Image Not Found**
```bash
# Verify image exists in OCIR
docker pull fra.ocir.io/frntrd2vyxvi/models:mistraltraining-fsdp-zero-v1

# Check image name in job template
grep "image:" k8s/job-master.template.yaml
```

**B. Missing Image Pull Secret**
```bash
# Create OCIR secret
kubectl create secret docker-registry ocir-secret \
  --docker-server=fra.ocir.io \
  --docker-username='<tenancy>/<username>' \
  --docker-password='<auth-token>' \
  -n bench

# Add to job template
spec:
  template:
    spec:
      imagePullSecrets:
        - name: ocir-secret
```

**C. Wrong Registry Region**
```bash
# Make sure registry region matches
IMAGE="fra.ocir.io/..."    # Frankfurt
IMAGE="iad.ocir.io/..."    # Ashburn
IMAGE="phx.ocir.io/..."    # Phoenix
```

### Issue 2: CrashLoopBackOff

**Symptoms:**
```
bench-master-xxx   0/1     CrashLoopBackOff   3          5m
```

**Diagnosis:**
```bash
# Check pod logs
kubectl logs bench-master-xxx -n bench

# Check previous crash
kubectl logs bench-master-xxx -n bench --previous

# Check pod events
kubectl describe pod bench-master-xxx -n bench
```

**Common Causes:**

**A. Missing Dependencies**
```python
ModuleNotFoundError: No module named 'deepspeed'
```
→ Rebuild image with all dependencies

**B. Entrypoint Script Errors**
```bash
/bin/bash: /app/docker/entrypoint.sh: No such file or directory
```
→ Check COPY commands in Dockerfile

**C. Python Syntax Errors**
```python
SyntaxError: invalid syntax
```
→ Test code locally before building image

### Issue 3: Insufficient GPU Resources

**Symptoms:**
```
0/4 nodes are available: 4 Insufficient nvidia.com/gpu
```

**Diagnosis:**
```bash
# Check GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# Check existing GPU allocations
kubectl get pods -A -o json | \
  jq '.items[] | select(.spec.containers[].resources.requests."nvidia.com/gpu" != null) | .metadata.name'
```

**Solution:**
```bash
# Delete old pods/jobs using GPUs
kubectl delete jobs --all -n bench

# Or request fewer GPUs
world_size=2  # Instead of 4
```

---

## Training Issues

### Issue 1: DeepSpeed TypeError (String vs Int)

**Symptoms:**
```python
TypeError: '>' not supported between instances of 'str' and 'int'
File "deepspeed/runtime/zero/stage_1_and_2.py", line 1567, in _batch_assertion
```

**Cause:** JSON configs have `"auto"` strings instead of integers.

**Solution:**

**Step 1:** Update `train_harness.py`
```python
# Remove "auto" strings
ds_config.pop('train_batch_size', None)
ds_config.pop('train_micro_batch_size_per_gpu', None)
ds_config.pop('gradient_accumulation_steps', None)

# Set explicit integers
ds_config['train_micro_batch_size_per_gpu'] = int(args.per_device_batch)
ds_config['gradient_accumulation_steps'] = int(args.grad_accum)
ds_config['train_batch_size'] = int(args.per_device_batch) * int(args.grad_accum) * args.world_size
```

**Step 2:** Clean JSON configs
```json
// Remove these lines from zero2.json and zero3.json
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"gradient_accumulation_steps": "auto",
```

**Step 3:** Rebuild and push image

### Issue 2: Process Group Not Initialized

**Symptoms:**
```python
RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
```

**Cause:** `dist.init_process_group()` not called or failed.

**Diagnosis:**
```bash
# Check environment variables
kubectl logs <pod> -n bench | grep -E "RANK|WORLD_SIZE|MASTER"

# Should see:
# RANK=0
# WORLD_SIZE=4
# MASTER_ADDR=10.0.10.101
# MASTER_PORT=29500
```

**Solutions:**

**A. Missing Environment Variables**
```yaml
# In job template, ensure all required env vars:
env:
  - name: RANK
  - name: WORLD_SIZE
  - name: MASTER_ADDR
  - name: MASTER_PORT
  - name: LOCAL_RANK
```

**B. Master Not Reachable**
```bash
# From worker pod, test connectivity
kubectl exec -it bench-workers-0 -n bench -- \
  nc -zv <master-ip> 29500

# Should output: Connection to <master-ip> 29500 port [tcp/*] succeeded!
```

**C. Firewall/Network Policy**
```yaml
# Ensure hostNetwork mode is enabled
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
```

### Issue 3: IPv6 Warnings and Hangs

**Symptoms:**
```
[W socket.cpp:663] [c10d] IPv6 address not available for eth0
<Process hangs at init_process_group>
```

**Solution:**
```yaml
# Use hostNetwork mode to bypass IPv6 issues
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet

# Set NCCL to use IPv4 interface
env:
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
  - name: NCCL_DEBUG
    value: "INFO"
```

### Issue 4: CUDA Out of Memory (OOM)

**Symptoms:**
```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

**A. Reduce Batch Size**
```bash
# In launch_multi.sh
--per-device-batch 1  # Instead of 2
--grad-accum 8        # Instead of 4 (maintain same global batch)
```

**B. Use More Memory-Efficient Strategy**
```bash
# Switch from DDP to ZeRO-3
--strategy zero3
```

**C. Enable Activation Checkpointing**
```python
# In train_harness.py
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.layer, x, use_reentrant=False)
```

**D. Reduce Sequence Length**
```bash
--seq-len 1024  # Instead of 2048
```

### Issue 5: NCCL Timeout

**Symptoms:**
```
[E ProcessGroupNCCL.cpp:563] [Rank 1] Watchdog caught collective operation timeout
```

**Cause:** Network issues, slow GPUs, or deadlock.

**Diagnosis:**
```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO

# Check for network issues
kubectl exec -it <worker-pod> -- ping <master-ip>

# Check GPU utilization
kubectl exec -it <pod> -- nvidia-smi dmon -s u -c 10
```

**Solutions:**

**A. Increase Timeout**
```python
# In train_harness.py
dist.init_process_group(
    backend='nccl',
    timeout=timedelta(minutes=30)  # Default is 10 minutes
)
```

**B. Check for Deadlocks**
```python
# Ensure all ranks call collective operations in same order
# BAD:
if rank == 0:
    dist.all_reduce(tensor)  # Only rank 0 calls - DEADLOCK!

# GOOD:
dist.all_reduce(tensor)  # All ranks call
```

---

## Results Collection Issues

### Issue 1: Cannot Exec into Completed Pod

**Symptoms:**
```bash
kubectl cp bench-master-xxx:/results/result.json ./result.json
Error: cannot exec into a container in a completed pod
```

**Cause:** Pod has terminated, EmptyDir volumes destroyed.

**Solution:** Use log-based collection (already implemented)
```bash
# This works even after pod termination
./scripts/collect_results.sh bench <job-name> ./results
```

### Issue 2: No JSON Found in Logs

**Symptoms:**
```
⚠ No JSON results found in logs (job may have failed)
```

**Diagnosis:**
```bash
# Check if training completed successfully
kubectl logs <job-name> -n bench | grep "Results saved"

# Check for errors
kubectl logs <job-name> -n bench | grep -i error

# Look for JSON markers
kubectl logs <job-name> -n bench | grep BENCHMARK_RESULT_JSON
```

**Solutions:**

**A. Job Failed Before Completion**
```bash
# Check why job failed
kubectl describe job <job-name> -n bench
kubectl logs <job-name> -n bench --tail=100
```

**B. JSON Not Printed to Stdout**
```python
# Verify in train_harness.py
print("\nBENCHMARK_RESULT_JSON_START")
print(json.dumps(result, indent=2))
print("BENCHMARK_RESULT_JSON_END\n")
```

### Issue 3: Missing Pandas for Analysis

**Symptoms:**
```python
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# On OKE VM (not in container)
pip3 install --user pandas matplotlib numpy

# Or use the install script
./scripts/install_analysis_deps.sh
```

---

## Performance Issues

### Issue 1: Low Throughput

**Symptoms:** Tokens/sec significantly lower than expected.

**Diagnosis:**
```bash
# Check GPU utilization
kubectl exec -it <pod> -n bench -- nvidia-smi dmon -s u

# Should be >80% during training
# If low (<50%), indicates CPU bottleneck or I/O wait
```

**Solutions:**

**A. CPU Bottleneck (Data Loading)**
```python
# Increase num_workers in DataLoader
train_loader = DataLoader(
    dataset,
    num_workers=4,  # More workers
    pin_memory=True
)
```

**B. Small Batch Size**
```bash
# Increase batch size (if memory allows)
--per-device-batch 4  # Instead of 1
```

**C. Suboptimal Strategy**
```bash
# Try DeepSpeed ZeRO-2 for best throughput
--strategy zero2
```

### Issue 2: Poor Scaling Efficiency

**Symptoms:** 4 GPUs only 1.2x faster than 2 GPUs (should be ~1.8x).

**Diagnosis:**
```bash
# Check network latency between nodes
kubectl exec -it <worker-pod> -- ping <master-ip>

# Check NCCL all-reduce bandwidth
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2
```

**Solutions:**

**A. Network Bottleneck**
```bash
# Ensure RDMA is enabled (if available)
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

**B. Gradient Accumulation Too High**
```bash
# Reduce gradient accumulation for better parallelism
--grad-accum 2  # Instead of 8
```

**C. Wrong NCCL Algorithm**
```bash
export NCCL_ALGO=Ring    # Good for small clusters
export NCCL_ALGO=Tree    # Good for large clusters (8+ GPUs)
```

### Issue 3: High Memory Usage

**Symptoms:** Peak VRAM higher than expected.

**Diagnosis:**
```bash
# Monitor memory during training
kubectl exec -it <pod> -- nvidia-smi dmon -s m -c 100
```

**Solutions:**

**A. Switch to Memory-Efficient Strategy**
```bash
--strategy zero3  # Most memory-efficient
```

**B. Enable Activation Checkpointing**
```python
# Trade 20% speed for 50% memory reduction
use_checkpoint = True
```

**C. Reduce Model Size**
```python
# For testing, use smaller model
n_layer = 6    # Instead of 12
n_embd = 512   # Instead of 768
```

---

## Quick Reference

### Useful Debug Commands

```bash
# Check pod status
kubectl get pods -n bench -o wide

# View pod logs
kubectl logs <pod-name> -n bench --tail=100 -f

# Execute commands in pod
kubectl exec -it <pod-name> -n bench -- bash

# Check GPU availability
kubectl exec -it <pod-name> -n bench -- nvidia-smi

# Check network connectivity
kubectl exec -it <worker-pod> -n bench -- nc -zv <master-ip> 29500

# View job status
kubectl describe job <job-name> -n bench

# Get events
kubectl get events -n bench --sort-by='.lastTimestamp'

# Force delete stuck pod
kubectl delete pod <pod-name> -n bench --force --grace-period=0

# Clean up everything
kubectl delete namespace bench
kubectl create -f k8s/namespace.yaml
```

### Enable Maximum Debugging

```yaml
# In job template
env:
  - name: NCCL_DEBUG
    value: "INFO"
  - name: NCCL_DEBUG_SUBSYS
    value: "ALL"
  - name: TORCH_DISTRIBUTED_DEBUG
    value: "DETAIL"
  - name: PYTHONFAULTHANDLER
    value: "1"
```

```bash
# In scripts
set -x  # Print all commands
set -e  # Exit on error
set -u  # Error on undefined variables
```

---

**Need More Help?**

1. Check [Architecture Documentation](ARCHITECTURE.md)
2. Review [main README](../README.md)
3. Check GitHub issues
4. Contact: deep.soni@oracle.com

---

**Last Updated:** January 6, 2026
