#!/usr/bin/env bash
set -euo pipefail

echo "==================================================================="
echo "  OKE CLUSTER GPU VERIFICATION"
echo "==================================================================="
echo ""

# Check kubectl connectivity
echo "Step 1: Checking kubectl connectivity..."
if ! kubectl cluster-info &>/dev/null; then
    echo "❌ ERROR: Cannot connect to Kubernetes cluster"
    echo "   Please configure kubectl first:"
    echo "   oci ce cluster create-kubeconfig --cluster-id <cluster-ocid>"
    exit 1
fi
echo "✅ Connected to cluster: $(kubectl config current-context)"
echo ""

# Check NVIDIA device plugin
echo "Step 2: Checking NVIDIA device plugin..."
NVIDIA_PODS=$(kubectl -n kube-system get pods -l name=nvidia-device-plugin-ds -o name 2>/dev/null | wc -l)
if [ "$NVIDIA_PODS" -eq 0 ]; then
    echo "⚠️  WARNING: NVIDIA device plugin not found"
    echo "   Looking for alternative GPU operator..."
    GPU_OPERATOR=$(kubectl -n gpu-operator get pods 2>/dev/null | wc -l)
    if [ "$GPU_OPERATOR" -eq 0 ]; then
        echo "❌ ERROR: No GPU device plugin found"
        echo "   Install NVIDIA device plugin or GPU operator first"
        exit 1
    else
        echo "✅ Found GPU operator running"
    fi
else
    echo "✅ NVIDIA device plugin running ($NVIDIA_PODS pods)"
    kubectl -n kube-system get pods -l name=nvidia-device-plugin-ds -o wide
fi
echo ""

# Check nodes with GPU capacity
echo "Step 3: Checking nodes with GPU capacity..."
GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity["nvidia.com/gpu"] != null) | .metadata.name')

if [ -z "$GPU_NODES" ]; then
    echo "❌ ERROR: No nodes with GPU capacity found"
    echo "   Please ensure your OKE cluster has GPU worker nodes"
    exit 1
fi

echo "✅ Found GPU nodes:"
echo ""
echo "Node Name                          GPU Capacity    GPU Allocatable    GPU Type"
echo "--------------------------------   ------------    ---------------    --------"

for node in $GPU_NODES; do
    GPU_CAPACITY=$(kubectl get node "$node" -o jsonpath='{.status.capacity.nvidia\.com/gpu}')
    GPU_ALLOCATABLE=$(kubectl get node "$node" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
    GPU_TYPE=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' || echo "unknown")
    printf "%-34s %-15s %-18s %s\n" "$node" "$GPU_CAPACITY" "$GPU_ALLOCATABLE" "$GPU_TYPE"
done
echo ""

# Count total GPUs
TOTAL_GPU_CAPACITY=$(kubectl get nodes -o json | jq '[.items[] | select(.status.capacity["nvidia.com/gpu"] != null) | .status.capacity["nvidia.com/gpu"] | tonumber] | add // 0')
TOTAL_GPU_ALLOCATABLE=$(kubectl get nodes -o json | jq '[.items[] | select(.status.allocatable["nvidia.com/gpu"] != null) | .status.allocatable["nvidia.com/gpu"] | tonumber] | add // 0')

echo "Total GPU Capacity:     $TOTAL_GPU_CAPACITY"
echo "Total GPU Allocatable:  $TOTAL_GPU_ALLOCATABLE"
echo ""

# Check for GPU workloads already running
echo "Step 4: Checking current GPU usage..."
USED_GPUS=$(kubectl get pods --all-namespaces -o json | jq '[.items[] | .spec.containers[]? | .resources.requests["nvidia.com/gpu"] // 0 | tonumber] | add // 0')
echo "GPUs currently in use:  $USED_GPUS"
echo "GPUs available:         $((TOTAL_GPU_ALLOCATABLE - USED_GPUS))"
echo ""

# Check bench namespace
echo "Step 5: Checking 'bench' namespace..."
if kubectl get namespace bench &>/dev/null; then
    echo "✅ Namespace 'bench' exists"

    # Check OCIR secret
    if kubectl -n bench get secret ocir-secret &>/dev/null; then
        echo "✅ OCIR secret 'ocir-secret' exists"
    else
        echo "⚠️  WARNING: OCIR secret 'ocir-secret' not found"
        echo "   Create it with:"
        echo "   kubectl create secret docker-registry ocir-secret \\"
        echo "     --namespace bench \\"
        echo "     --docker-server=fra.ocir.io \\"
        echo "     --docker-username='frntrd2vyxvi/<your-username>' \\"
        echo "     --docker-password='<your-auth-token>'"
    fi
else
    echo "⚠️  Namespace 'bench' not found"
    echo "   Create it with: kubectl apply -f k8s/namespace.yaml"
fi
echo ""

# Summary
echo "==================================================================="
echo "  CLUSTER STATUS SUMMARY"
echo "==================================================================="
if [ "$TOTAL_GPU_ALLOCATABLE" -ge 4 ]; then
    echo "✅ Cluster has sufficient GPUs for benchmarking"
    echo ""
    echo "Recommended test matrix:"
    echo "  - 1 GPU:  Smoke test + baseline"
    echo "  - 2 GPUs: DDP, FSDP, ZeRO-2, ZeRO-3 scaling"
    echo "  - 4 GPUs: Full distributed benchmark"
    echo ""
    echo "Next steps:"
    echo "  1. kubectl apply -f k8s/namespace.yaml"
    echo "  2. kubectl apply -f k8s/serviceaccount.yaml"
    echo "  3. ./scripts/launch_smoke.sh"
elif [ "$TOTAL_GPU_ALLOCATABLE" -ge 2 ]; then
    echo "⚠️  Cluster has $TOTAL_GPU_ALLOCATABLE GPUs (recommended: 4)"
    echo "   You can still run:"
    echo "   - 1 GPU tests (smoke test, baseline)"
    echo "   - 2 GPU tests (limited scaling tests)"
    echo ""
    echo "Next steps:"
    echo "  1. kubectl apply -f k8s/namespace.yaml"
    echo "  2. kubectl apply -f k8s/serviceaccount.yaml"
    echo "  3. ./scripts/launch_smoke.sh"
else
    echo "⚠️  WARNING: Only $TOTAL_GPU_ALLOCATABLE GPU(s) available"
    echo "   Recommended: At least 2 GPUs for distributed training tests"
fi
echo "==================================================================="
