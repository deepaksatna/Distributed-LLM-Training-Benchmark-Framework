#!/usr/bin/env python3
"""
Generate plots from benchmark metrics CSV
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(csv_path: str, output_dir: str):
    """Generate benchmark plots"""
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} records from {csv_path}")

    # Plot 1: Tokens/sec vs GPU count (per strategy)
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy].sort_values('world_size')
        ax.plot(subset['world_size'], subset['tokens_per_sec'],
                marker='o', label=strategy.upper(), linewidth=2)
    ax.set_xlabel('World Size (GPUs)', fontsize=12)
    ax.set_ylabel('Tokens/sec', fontsize=12)
    ax.set_title('Tokens/sec vs GPU Count', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot1 = output_path / "tokens_per_sec_vs_gpu.png"
    plt.savefig(plot1, dpi=150)
    plt.close()
    print(f"Plot saved: {plot1}")

    # Plot 2: Step time vs GPU count
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy].sort_values('world_size')
        ax.plot(subset['world_size'], subset['mean_step_time_sec'],
                marker='o', label=strategy.upper(), linewidth=2)
    ax.set_xlabel('World Size (GPUs)', fontsize=12)
    ax.set_ylabel('Mean Step Time (sec)', fontsize=12)
    ax.set_title('Step Time vs GPU Count', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2 = output_path / "step_time_vs_gpu.png"
    plt.savefig(plot2, dpi=150)
    plt.close()
    print(f"Plot saved: {plot2}")

    # Plot 3: Peak VRAM vs sequence length (per strategy)
    if len(df['seq_len'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy].sort_values('seq_len')
            ax.plot(subset['seq_len'], subset['peak_vram_gb'],
                    marker='o', label=strategy.upper(), linewidth=2)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Peak VRAM (GB)', fontsize=12)
        ax.set_title('Peak VRAM vs Sequence Length', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot3 = output_path / "vram_vs_seqlen.png"
        plt.savefig(plot3, dpi=150)
        plt.close()
        print(f"Plot saved: {plot3}")

    # Plot 4: Scaling efficiency (per strategy)
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy].sort_values('world_size')
        ax.plot(subset['world_size'], subset['scaling_efficiency_pct'],
                marker='o', label=strategy.upper(), linewidth=2)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax.set_xlabel('World Size (GPUs)', fontsize=12)
    ax.set_ylabel('Scaling Efficiency (%)', fontsize=12)
    ax.set_title('Scaling Efficiency vs GPU Count', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plot4 = output_path / "scaling_efficiency.png"
    plt.savefig(plot4, dpi=150)
    plt.close()
    print(f"Plot saved: {plot4}")

    # Plot 5: GB/sec per GPU (per strategy)
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in df['strategy'].unique():
        subset = df[df['strategy'] == strategy].sort_values('world_size')
        ax.plot(subset['world_size'], subset['h2d_gbps_per_gpu'],
                marker='o', label=strategy.upper(), linewidth=2)
    ax.set_xlabel('World Size (GPUs)', fontsize=12)
    ax.set_ylabel('H2D GB/s per GPU', fontsize=12)
    ax.set_title('Data Transfer Rate vs GPU Count', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot5 = output_path / "gbps_vs_gpu.png"
    plt.savefig(plot5, dpi=150)
    plt.close()
    print(f"Plot saved: {plot5}")

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument('--results', required=True, help='Path to metrics.csv')
    parser.add_argument('--out', required=True, help='Output directory for plots')
    args = parser.parse_args()

    plot_metrics(args.results, args.out)


if __name__ == '__main__':
    main()
