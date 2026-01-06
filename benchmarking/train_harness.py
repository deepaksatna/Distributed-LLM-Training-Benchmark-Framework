#!/usr/bin/env python3
"""
Unified distributed training benchmark: DDP, FSDP, DeepSpeed ZeRO-2/3
For OKE with A10 GPUs - Synthetic data mode for reproducible benchmarks
"""
import argparse
import json
import os
import time
import yaml
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.cuda.amp as amp

# Conditional DeepSpeed import
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("WARNING: DeepSpeed not available")


# ============================================================================
# Model Definition: TinyGPT for synthetic benchmarking
# ============================================================================

class TinyGPT(nn.Module):
    """Minimal GPT-style model for benchmarking distributed strategies"""

    def __init__(
        self,
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        block_size: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.transformer['wte'].weight = self.lm_head.weight

        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence {t} exceeds block size {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer['wte'](idx)
        pos_emb = self.transformer['wpe'](pos)
        x = self.transformer['drop'](tok_emb + pos_emb)

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss


class TransformerBlock(nn.Module):
    """Standard transformer block with MHA + FFN"""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        # FFN with residual
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================================
# Synthetic Dataset
# ============================================================================

class SyntheticDataset(Dataset):
    """Fixed synthetic dataset for reproducible benchmarks"""

    def __init__(self, vocab_size: int = 32000, seq_len: int = 2048, size: int = 1000, seed: int = 42):
        torch.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        print(f"SyntheticDataset: {size} samples, seq_len={seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Model Configuration by Tier
# ============================================================================

def get_model_config(tier: str, seq_len: int) -> Dict[str, int]:
    """
    Tier A: ~1-3B params - fast iteration
    Tier B: ~7-13B params - stress test (scaled up)
    """
    if tier == "A":
        return {
            'vocab_size': 32000,
            'n_embd': 1024,
            'n_head': 16,
            'n_layer': 16,
            'block_size': seq_len,
        }
    elif tier == "B":
        return {
            'vocab_size': 32000,
            'n_embd': 2048,
            'n_head': 32,
            'n_layer': 32,
            'block_size': seq_len,
        }
    else:
        raise ValueError(f"Unknown tier: {tier}")


# ============================================================================
# Main Training Logic
# ============================================================================

def setup_distributed(args):
    """Initialize distributed training backend"""
    if args.world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{args.master_addr}:{args.master_port}',
                world_size=args.world_size,
                rank=args.rank
            )
        print(f"[Rank {args.rank}/{args.world_size}] Distributed initialized")
    else:
        print("Single-GPU mode (no distributed)")


def cleanup_distributed():
    """Cleanup distributed resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model, args):
    """Wrap model based on strategy: DDP, FSDP, or DeepSpeed"""

    if args.strategy == "ddp":
        if args.world_size == 1:
            # Single GPU - no DDP wrapping needed
            print(f"[Rank {args.rank}] Single-GPU mode, skipping DDP wrapper")
            return model, None
        else:
            print(f"[Rank {args.rank}] Wrapping with DDP")
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False
            )
            return model, None

    elif args.strategy == "fsdp":
        print(f"[Rank {args.rank}] Wrapping with FSDP")

        # FSDP auto-wrap policy (wrap layers > 100M params)
        auto_wrap_policy = size_based_auto_wrap_policy

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=None,  # We handle AMP separately
            device_id=args.local_rank,
        )
        return model, None

    elif args.strategy in ["zero2", "zero3"]:
        if not DEEPSPEED_AVAILABLE:
            raise RuntimeError("DeepSpeed not available but strategy is zero2/zero3")

        print(f"[Rank {args.rank}] Initializing DeepSpeed {args.strategy.upper()}")

        # Load DeepSpeed config
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)

        # Remove "auto" strings that cause TypeError in DeepSpeed's batch assertion
        ds_config.pop('train_batch_size', None)
        ds_config.pop('train_micro_batch_size_per_gpu', None)
        ds_config.pop('gradient_accumulation_steps', None)

        # Explicitly calculate and set all batch-related fields as integers
        micro_batch = int(args.per_device_batch)
        grad_accum = int(args.grad_accum)
        train_batch_size = micro_batch * grad_accum * args.world_size

        ds_config['train_micro_batch_size_per_gpu'] = micro_batch
        ds_config['gradient_accumulation_steps'] = grad_accum
        ds_config['train_batch_size'] = train_batch_size

        print(f"[Rank {args.rank}] DeepSpeed batch config: micro={micro_batch}, accum={grad_accum}, total={train_batch_size}")

        # DeepSpeed engine initialization
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        return model_engine, optimizer

    else:
        # Single-GPU (no wrapping)
        return model, None


def train(args):
    """Main training loop with metrics collection"""

    # Setup
    setup_distributed(args)
    torch.manual_seed(42 + args.rank)
    torch.cuda.manual_seed(42 + args.rank)

    device = torch.device(f'cuda:{args.local_rank}')
    torch.cuda.set_device(device)

    is_main = args.rank == 0

    if is_main:
        print("\n" + "="*80)
        print(f"Benchmark Config: {args.strategy.upper()} | Tier {args.tier} | WS={args.world_size} | SeqLen={args.seq_len}")
        print("="*80 + "\n")

    # Model
    model_config = get_model_config(args.tier, args.seq_len)
    model = TinyGPT(**model_config)
    model = model.to(device)

    # Wrap model based on strategy
    model, optimizer_from_ds = wrap_model(model, args)

    # Dataset & DataLoader
    dataset = SyntheticDataset(
        vocab_size=model_config['vocab_size'],
        seq_len=args.seq_len,
        size=1000,
        seed=42
    )

    sampler = None
    if args.world_size > 1 and args.strategy not in ["zero2", "zero3"]:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer (unless DeepSpeed provides it)
    if optimizer_from_ds is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    else:
        optimizer = optimizer_from_ds

    # AMP scaler (not used with DeepSpeed as it has built-in AMP)
    use_amp = args.strategy not in ["zero2", "zero3"]
    scaler = amp.GradScaler() if use_amp else None

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        print(f"Starting training: {args.steps} steps, warmup={args.warmup_steps}")
        print(f"Per-device batch: {args.per_device_batch}, Grad accum: {args.grad_accum}")
        print("")

    model.train()
    step_times = []
    losses = []

    data_iter = iter(dataloader)

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device, non_blocking=True)
        targets = batch.clone()

        step_start = time.perf_counter()

        # Forward + Backward
        if args.strategy in ["zero2", "zero3"]:
            # DeepSpeed path
            _, loss = model(batch, targets)
            model.backward(loss)
            model.step()
        else:
            # DDP/FSDP/single-GPU path
            if use_amp:
                with amp.autocast():
                    _, loss = model(batch, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(batch, targets)
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

        step_end = time.perf_counter()
        step_time = step_end - step_start

        # Collect metrics after warmup
        if step >= args.warmup_steps:
            step_times.append(step_time)
            losses.append(loss.item())

        if is_main and step % 10 == 0:
            print(f"[Step {step:04d}] Loss: {loss.item():.4f}, Time: {step_time:.3f}s")

    # Final metrics
    if dist.is_initialized():
        dist.barrier()

    mean_step_time = sum(step_times) / len(step_times) if step_times else 0.0
    mean_loss = sum(losses) / len(losses) if losses else 0.0

    # Tokens/sec (global)
    tokens_per_step = args.per_device_batch * args.seq_len * args.world_size
    tokens_per_sec = tokens_per_step / mean_step_time if mean_step_time > 0 else 0.0

    # GPU memory
    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    peak_vram_gb = peak_vram_bytes / 1e9

    # GB/sec per GPU: data transfer rate (H2D equivalent calculation)
    # Approximation: (batch_tokens * 4 bytes per token for FP32 equivalent) / step_time
    bytes_per_step = args.per_device_batch * args.seq_len * 4  # 4 bytes/token (FP32)
    h2d_gbps_per_gpu = (bytes_per_step / mean_step_time) / 1e9 if mean_step_time > 0 else 0.0

    result = {
        'strategy': args.strategy,
        'world_size': args.world_size,
        'rank': args.rank,
        'seq_len': args.seq_len,
        'tier': args.tier,
        'steps': args.steps,
        'per_device_batch': args.per_device_batch,
        'grad_accum': args.grad_accum,
        'tokens_per_sec': tokens_per_sec,
        'mean_step_time_sec': mean_step_time,
        'mean_loss': mean_loss,
        'peak_vram_gb': peak_vram_gb,
        'h2d_gbps_per_gpu': h2d_gbps_per_gpu,
    }

    if is_main:
        print("\n" + "="*80)
        print("Benchmark Results:")
        print(f"  Tokens/sec:       {tokens_per_sec:,.0f}")
        print(f"  Mean step time:   {mean_step_time:.4f}s")
        print(f"  Peak VRAM/GPU:    {peak_vram_gb:.2f} GB")
        print(f"  H2D GB/s/GPU:     {h2d_gbps_per_gpu:.3f}")
        print(f"  Mean loss:        {mean_loss:.4f}")
        print("="*80 + "\n")

        # Save results
        os.makedirs(args.results_dir, exist_ok=True)
        result_file = os.path.join(
            args.results_dir,
            f"result_{args.strategy}_ws{args.world_size}_seq{args.seq_len}_tier{args.tier}.json"
        )
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {result_file}")

        # Also output JSON to stdout for log-based collection
        print("\n" + "="*80)
        print("BENCHMARK_RESULT_JSON_START")
        print(json.dumps(result, indent=2))
        print("BENCHMARK_RESULT_JSON_END")
        print("="*80 + "\n")

    cleanup_distributed()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Distributed Training Benchmark")

    # Strategy
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['ddp', 'fsdp', 'zero2', 'zero3'],
                        help='Distributed strategy')

    # Distributed
    parser.add_argument('--world-size', type=int, required=True, help='Total number of GPUs')
    parser.add_argument('--rank', type=int, required=True, help='Global rank')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master address')
    parser.add_argument('--master-port', type=int, default=29500, help='Master port')

    # Model & Data
    parser.add_argument('--tier', type=str, required=True, choices=['A', 'B'], help='Model tier')
    parser.add_argument('--seq-len', type=int, required=True, help='Sequence length')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')

    # Training
    parser.add_argument('--steps', type=int, required=True, help='Training steps')
    parser.add_argument('--warmup-steps', type=int, default=5, help='Warmup steps')
    parser.add_argument('--per-device-batch', type=int, required=True, help='Batch size per GPU')
    parser.add_argument('--grad-accum', type=int, required=True, help='Gradient accumulation steps')

    # Configs
    parser.add_argument('--deepspeed-config', type=str, help='DeepSpeed config JSON')
    parser.add_argument('--fsdp-config', type=str, help='FSDP config YAML')

    # Output
    parser.add_argument('--results-dir', type=str, required=True, help='Results directory')

    args = parser.parse_args()

    # Validate
    if args.strategy in ['zero2', 'zero3'] and not args.deepspeed_config:
        raise ValueError("DeepSpeed strategy requires --deepspeed-config")

    train(args)


if __name__ == '__main__':
    main()
