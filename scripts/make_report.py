#!/usr/bin/env python3
"""
Generate Markdown benchmark summary report from CSV
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_report(csv_path: str, output_dir: str):
    """Generate Markdown report from metrics CSV"""
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / "BENCHMARK_REPORT.md"

    with open(report_file, 'w') as f:
        # Header
        f.write("# Distributed Training Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Platform:** Oracle Kubernetes Engine (OKE) with NVIDIA A10 GPUs (24GB)\n\n")
        f.write("---\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Strategy | World Size | Seq Len | Tier | Tokens/sec | Step Time (s) | Peak VRAM (GB) | Scaling Eff (%) |\n")
        f.write("|----------|-----------|---------|------|------------|---------------|----------------|------------------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['strategy'].upper():8s} | {row['world_size']:9d} | "
                    f"{row['seq_len']:7d} | {row['tier']:4s} | "
                    f"{row['tokens_per_sec']:10,.0f} | {row['mean_step_time_sec']:13.4f} | "
                    f"{row['peak_vram_gb']:14.2f} | {row['scaling_efficiency_pct']:16.1f} |\n")

        f.write("\n---\n\n")

        # Per-strategy breakdown
        f.write("## Strategy Comparison\n\n")

        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy].sort_values(['world_size', 'seq_len'])
            f.write(f"### {strategy.upper()}\n\n")

            if len(subset) == 0:
                f.write("No data available.\n\n")
                continue

            f.write("| World Size | Seq Len | Tokens/sec | Step Time (s) | Peak VRAM (GB) | H2D GB/s/GPU | Scaling Eff (%) |\n")
            f.write("|-----------|---------|------------|---------------|----------------|--------------|------------------|\n")

            for _, row in subset.iterrows():
                f.write(f"| {row['world_size']:9d} | {row['seq_len']:7d} | "
                        f"{row['tokens_per_sec']:10,.0f} | {row['mean_step_time_sec']:13.4f} | "
                        f"{row['peak_vram_gb']:14.2f} | {row['h2d_gbps_per_gpu']:12.3f} | "
                        f"{row['scaling_efficiency_pct']:16.1f} |\n")

            f.write("\n")

        f.write("---\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Best throughput
        best_tps = df.loc[df['tokens_per_sec'].idxmax()]
        f.write(f"- **Best Throughput:** {best_tps['tokens_per_sec']:,.0f} tokens/sec "
                f"({best_tps['strategy'].upper()}, WS={best_tps['world_size']}, "
                f"SeqLen={best_tps['seq_len']})\n")

        # Best scaling efficiency
        best_eff = df.loc[df['scaling_efficiency_pct'].idxmax()]
        f.write(f"- **Best Scaling Efficiency:** {best_eff['scaling_efficiency_pct']:.1f}% "
                f"({best_eff['strategy'].upper()}, WS={best_eff['world_size']})\n")

        # Lowest VRAM
        lowest_vram = df.loc[df['peak_vram_gb'].idxmin()]
        f.write(f"- **Lowest Peak VRAM:** {lowest_vram['peak_vram_gb']:.2f} GB "
                f"({lowest_vram['strategy'].upper()}, WS={lowest_vram['world_size']})\n")

        f.write("\n---\n\n")

        # Insights
        f.write("## Strategy Trade-offs\n\n")
        f.write("### DDP (Distributed Data Parallel)\n")
        f.write("- Replicates entire model on each GPU\n")
        f.write("- Best for small-to-medium models that fit in single GPU memory\n")
        f.write("- Minimal communication overhead, highest throughput when applicable\n")
        f.write("- Memory: O(model_size) per GPU\n\n")

        f.write("### FSDP (Fully Sharded Data Parallel)\n")
        f.write("- Shards model parameters, gradients, and optimizer states across GPUs\n")
        f.write("- Enables training of larger models than DDP\n")
        f.write("- PyTorch-native, good balance of memory and speed\n")
        f.write("- Memory: O(model_size / world_size) per GPU\n\n")

        f.write("### DeepSpeed ZeRO-2\n")
        f.write("- Shards optimizer states and gradients only (parameters replicated)\n")
        f.write("- Lower communication overhead than ZeRO-3\n")
        f.write("- Good when model parameters fit but optimizer states don't\n")
        f.write("- Memory: Between DDP and FSDP\n\n")

        f.write("### DeepSpeed ZeRO-3\n")
        f.write("- Shards parameters, gradients, AND optimizer states\n")
        f.write("- Maximum memory efficiency, enables largest models\n")
        f.write("- Higher communication overhead (parameter gathering)\n")
        f.write("- Memory: O(model_size / world_size) per GPU\n\n")

        f.write("---\n\n")

        # Plots
        f.write("## Visualizations\n\n")
        f.write("![Tokens per second](plots/tokens_per_sec_vs_gpu.png)\n\n")
        f.write("![Step time](plots/step_time_vs_gpu.png)\n\n")
        f.write("![Scaling efficiency](plots/scaling_efficiency.png)\n\n")
        f.write("![Peak VRAM](plots/vram_vs_seqlen.png)\n\n")
        f.write("![Data transfer rate](plots/gbps_vs_gpu.png)\n\n")

        f.write("---\n\n")
        f.write("**Generated by:** Oracle AI CoE Distributed Training Benchmark Suite\n")

    print(f"Report generated: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument('--csv', required=True, help='Path to metrics.csv')
    parser.add_argument('--out', required=True, help='Output directory for report')
    args = parser.parse_args()

    generate_report(args.csv, args.out)


if __name__ == '__main__':
    main()
