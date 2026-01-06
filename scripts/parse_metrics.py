#!/usr/bin/env python3
"""
Parse benchmark result JSON files into a consolidated CSV
"""
import argparse
import json
import os
import pandas as pd
from pathlib import Path


def parse_results(results_dir: str, output_dir: str):
    """
    Scan results_dir for result_*.json files and aggregate into CSV
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON result files (in subdirectories like bench-master-*_results/result.json)
    json_files = list(results_path.rglob("result.json"))

    if not json_files:
        print(f"WARNING: No result.json files found in {results_dir}")
        print(f"Expected files like: bench-master-*_results/result.json")
        return

    print(f"Found {len(json_files)} result files")

    records = []
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"WARNING: Failed to parse {jf}: {e}")

    if not records:
        print("ERROR: No valid records parsed")
        return

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Sort by strategy, world_size, seq_len
    df = df.sort_values(by=['strategy', 'world_size', 'seq_len'])

    # Calculate scaling efficiency (vs WS=1 per strategy/seq_len)
    df['scaling_efficiency_pct'] = 100.0
    for strategy in df['strategy'].unique():
        for seq_len in df['seq_len'].unique():
            mask = (df['strategy'] == strategy) & (df['seq_len'] == seq_len)
            subset = df[mask].copy()
            if len(subset) > 1:
                baseline = subset[subset['world_size'] == subset['world_size'].min()]
                if not baseline.empty:
                    baseline_tps = baseline.iloc[0]['tokens_per_sec']
                    for idx, row in subset.iterrows():
                        ideal_tps = baseline_tps * row['world_size']
                        actual_tps = row['tokens_per_sec']
                        efficiency = (actual_tps / ideal_tps * 100.0) if ideal_tps > 0 else 0.0
                        df.loc[idx, 'scaling_efficiency_pct'] = efficiency

    # Save CSV
    csv_file = output_path / "metrics.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nMetrics saved to: {csv_file}")

    # Print summary
    print("\n=== Summary ===")
    print(df[['strategy', 'world_size', 'seq_len', 'tier', 'tokens_per_sec',
              'mean_step_time_sec', 'peak_vram_gb', 'h2d_gbps_per_gpu',
              'scaling_efficiency_pct']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark results to CSV")
    parser.add_argument('--results-dir', required=True, help='Directory containing result JSON files')
    parser.add_argument('--out', required=True, help='Output directory for metrics.csv')
    args = parser.parse_args()

    parse_results(args.results_dir, args.out)


if __name__ == '__main__':
    main()
