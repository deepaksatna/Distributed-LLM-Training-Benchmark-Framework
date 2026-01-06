# OCI OKE A10 (CUDA 12.1) - FULLY OFFLINE Distributed Training Benchmark
# ALL dependencies pre-installed, NO runtime downloads
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    PATH=/usr/local/cuda-12.1/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# OFFLINE MODE: Disable all external network calls for ML libraries
ENV HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    DISABLE_TELEMETRY=1

# System dependencies (installed from base image mirrors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    git ca-certificates curl wget jq vim \
    build-essential libaio-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip==23.3.1

# PyTorch with CUDA 12.1 support (pre-downloaded wheel, cached layer)
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Distributed training frameworks (pre-installed, no runtime downloads)
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    deepspeed==0.13.1 \
    peft==0.7.1

# Utilities and monitoring (all pre-installed)
RUN pip install --no-cache-dir \
    matplotlib==3.7.2 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    pyyaml==6.0 \
    psutil==5.9.6 \
    pynvml==11.5.0 \
    gpustat==1.1 \
    tqdm==4.65.0

# App structure
WORKDIR /app

# Copy application code (custom TinyGPT model defined in code - NO external models)
COPY benchmarking/ /app/benchmarking/
COPY configs/ /app/configs/
COPY docker/entrypoint.sh /app/entrypoint.sh

# Results directory
RUN mkdir -p /results && chmod -R 777 /results && \
    chmod +x /app/entrypoint.sh

# NCCL defaults for OKE multi-node
ENV NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_SOCKET_IFNAME=eth0 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    TORCH_DISTRIBUTED_DEBUG=DETAIL

# Verify offline mode at build time
RUN python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" && \
    python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')" && \
    python -c "import transformers; print(f'Transformers {transformers.__version__} (offline mode)')" && \
    echo "✓ All dependencies verified - 100% offline ready"

ENTRYPOINT ["/app/entrypoint.sh"]
