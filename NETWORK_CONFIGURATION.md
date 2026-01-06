# Network Configuration Guide

## ⚠️ CRITICAL: GPU Shape-Specific Network Settings

This benchmark framework currently uses **NVIDIA A10 VMs** which have **NO RDMA support**. If you're using A100, H100, or H200 GPUs, you **MUST** update the NCCL configuration.

---

## Current Configuration (A10 VMs)

### Hardware
- **GPU Shape:** VM.GPU.A10.1
- **Network Interface:** eth0 (Standard Ethernet)
- **RDMA:** Not available
- **Bandwidth:** 24.6 Gbps

### NCCL Configuration (Current in Job Templates)

```yaml
env:
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"              # Standard Ethernet
  - name: NCCL_IB_DISABLE
    value: "1"                 # MUST be 1 (no InfiniBand)
  - name: NCCL_DEBUG
    value: "WARN"
```

**Location:** `k8s/job-master.template.yaml` and `k8s/job-workers.template.yaml`

---

## Required Changes for A100/H100/H200

### A100 Bare Metal (BM.GPU.A100-v2.8)

**Hardware:**
- Network Interface: ib0 (InfiniBand)
- RDMA: Available
- NVLink: 600 GB/s (GPU-to-GPU)

**NCCL Configuration:**

```yaml
env:
  - name: NCCL_SOCKET_IFNAME
    value: "ib0"               # InfiniBand interface (verify with 'ifconfig')
  - name: NCCL_IB_DISABLE
    value: "0"                 # MUST be 0 (enable InfiniBand)
  - name: NCCL_NET_GDR_LEVEL
    value: "5"                 # Enable GPU Direct RDMA
  - name: NCCL_NVLINK_ENABLE
    value: "1"                 # Enable NVLink (A100 has NVLink)
  - name: NCCL_DEBUG
    value: "INFO"              # More verbose for debugging RDMA
  - name: NCCL_IB_HCA
    value: "mlx5"              # Mellanox HCA (verify your hardware)
```

**Verification Commands:**
```bash
# Check InfiniBand interface
ifconfig ib0

# Test RDMA bandwidth
ib_write_bw

# Test NCCL communication
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2
```

### H100 Bare Metal (BM.GPU4.8)

**Hardware:**
- Network Interface: ib0 (InfiniBand) + NVLink 4.0
- RDMA: Required for optimal performance
- NVLink: 900 GB/s

**NCCL Configuration:**

```yaml
env:
  - name: NCCL_SOCKET_IFNAME
    value: "ib0"
  - name: NCCL_IB_DISABLE
    value: "0"
  - name: NCCL_NET_GDR_LEVEL
    value: "5"
  - name: NCCL_NVLINK_ENABLE
    value: "1"                 # CRITICAL for H100
  - name: NCCL_COLLNET_ENABLE
    value: "1"                 # Enable SHARP (if available)
  - name: NCCL_DEBUG
    value: "INFO"
  - name: NCCL_IB_HCA
    value: "mlx5"
  - name: NCCL_TOPO_FILE
    value: "/etc/nccl/topology.xml"  # Optional: custom topology
```

**Additional Setup:**
```bash
# Enable NVLink
nvidia-smi nvlink --status

# Check topology
nvidia-smi topo -m

# Create custom topology file if needed
nvidia-smi topo -m --xml > /etc/nccl/topology.xml
```

### H200 Bare Metal

**Same configuration as H100** - uses identical networking setup.

---

## How to Update Configuration

### Step 1: Identify Your GPU Shape

```bash
# On OKE nodes
nvidia-smi

# Check network interfaces
ifconfig | grep -E "eth|ib"

# Check RDMA capability
ls /dev/infiniband/
```

### Step 2: Update Job Templates

Edit both:
- `k8s/job-master.template.yaml`
- `k8s/job-workers.template.yaml`

Find the NCCL environment variables section and replace based on your GPU shape.

### Step 3: Rebuild Image (if needed)

If you've hardcoded any NCCL settings in the Docker image, rebuild:

```bash
./scripts/build.sh
./scripts/push.sh
```

### Step 4: Test Connectivity

Before running full benchmarks:

```bash
# Single GPU smoke test
./scripts/launch_smoke.sh

# 2-GPU test
./scripts/launch_multi.sh --strategy ddp --world-size 2 --steps 10

# Check logs for NCCL initialization
kubectl logs <pod> -n bench | grep NCCL
```

---

## Verification Checklist

Before running benchmarks on A100/H100/H200:

- [ ] Verified GPU shape with `nvidia-smi`
- [ ] Identified network interface (ib0 vs eth0) with `ifconfig`
- [ ] Checked RDMA availability with `ls /dev/infiniband/`
- [ ] Updated NCCL_IB_DISABLE (0 for RDMA, 1 for no RDMA)
- [ ] Updated NCCL_SOCKET_IFNAME (ib0 for RDMA, eth0 for standard)
- [ ] Added NCCL_NET_GDR_LEVEL=5 (if RDMA available)
- [ ] Added NCCL_NVLINK_ENABLE=1 (for H100/H200)
- [ ] Tested with nccl-tests
- [ ] Ran smoke test successfully
- [ ] Checked logs for NCCL warnings/errors

---

## Expected Performance Impact

### With Correct Configuration

| GPU Shape | Expected Tokens/sec (4 GPU, ZeRO-2) |
|-----------|-------------------------------------|
| A10 (eth0, no RDMA) | 18,000 | ✅ Current validated |
| A100 (ib0, RDMA) | 45,000 - 54,000 | Expected 2.5-3x vs A10 |
| H100 (ib0, RDMA, NVLink) | 90,000 - 108,000 | Expected 5-6x vs A10 |
| H200 (same as H100) | 95,000 - 115,000 | Expected 5.5-6.5x vs A10 |

### With WRONG Configuration

- Using NCCL_IB_DISABLE=0 on A10: **Training will fail** (no InfiniBand device)
- Using NCCL_IB_DISABLE=1 on A100/H100: **10x performance degradation** (TCP/IP instead of RDMA)
- Wrong interface name: **Communication timeouts**

---

## Common Issues

### Issue: NCCL initialization hangs

**Cause:** Wrong interface name or RDMA settings

**Solution:**
```bash
# Check available interfaces
kubectl exec -it <pod> -n bench -- ifconfig

# Test NCCL with debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Issue: "No such device" error

**Cause:** NCCL_IB_DISABLE=0 but no InfiniBand hardware

**Solution:** Set NCCL_IB_DISABLE=1 (you're on A10 or standard VMs)

### Issue: Poor scaling efficiency (<20%)

**Cause:** Not using RDMA on capable hardware

**Solution:** Verify RDMA is enabled and working:
```bash
# Test RDMA
ibv_devinfo

# Test NCCL bandwidth
nccl-tests/build/all_reduce_perf -b 8 -e 128M
```

---

## Summary

| GPU Shape | Interface | NCCL_IB_DISABLE | NCCL_NET_GDR_LEVEL | NCCL_NVLINK_ENABLE |
|-----------|-----------|-----------------|--------------------|--------------------|
| **A10 VM** | eth0 | 1 (disable) | Not set | Not set |
| **A100 BM** | ib0 | 0 (enable) | 5 | 1 (enable) |
| **H100 BM** | ib0 | 0 (enable) | 5 | 1 (enable) |
| **H200 BM** | ib0 | 0 (enable) | 5 | 1 (enable) |

**Always test your configuration before running full benchmarks!**

---

**Last Updated:** January 6, 2026
**Maintained By:** Oracle AI CoE
