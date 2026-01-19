#!/usr/bin/env python3
"""
Compare Rust and Julia matrix multiplication benchmarks.
Generates separate plots for each algorithm type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Set up paths
RESULTS_DIR = "../results"
OUTPUT_DIR = "../results"

# Find CSV files
julia_files = glob.glob(os.path.join(RESULTS_DIR, "julia_benchmark_*.csv"))
rust_files = glob.glob(os.path.join(RESULTS_DIR, "rust_benchmark_*.csv"))

if not julia_files or not rust_files:
    print("Error: Could not find benchmark CSV files")
    exit(1)

# Read the most recent files
julia_df = pd.read_csv(julia_files[0])
rust_df = pd.read_csv(rust_files[0])

# Extract thread counts from filenames
julia_threads = re.search(r'_(\d+)t_', julia_files[0])
rust_threads = re.search(r'_(\d+)t_', rust_files[0])
julia_thread_count = julia_threads.group(1) if julia_threads else "unknown"
rust_thread_count = rust_threads.group(1) if rust_threads else "unknown"

# Add language column
julia_df['language'] = 'Julia'
rust_df['language'] = 'Rust'

# Get unique algorithms (excluding Julia-Builtin which is Julia-only)
algorithms = ['Iterative', 'Divide-Conquer', 'Strassen']

# Create plots for each algorithm
for algorithm in algorithms:
    # Time comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Add thread info to title for Divide-Conquer
    if algorithm == 'Iterative':
        title = f'{algorithm} algorithm'
    else:
        title = f'{algorithm} algorithm ({julia_thread_count} threads)'

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Filter data for this algorithm
    julia_data = julia_df[julia_df['algorithm'] == algorithm]
    rust_data = rust_df[rust_df['algorithm'] == algorithm]

    # Plot 1: Linear Scale (Time)
    julia_label = 'Julia'
    rust_label = 'Rust'

    ax1.plot(julia_data['size'], julia_data['time_ms'],
             marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
    ax1.plot(rust_data['size'], rust_data['time_ms'],
             marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Execution Time (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(julia_data['size'])
    ax1.set_xticklabels(julia_data['size'])

    # Plot 2: Log Scale (Time)
    ax2.plot(julia_data['size'], julia_data['time_ms'],
             marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
    ax2.plot(rust_data['size'], rust_data['time_ms'],
             marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Execution Time (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xticks(julia_data['size'])
    ax2.set_xticklabels(julia_data['size'])

    plt.tight_layout()

    # Save time comparison plot
    output_file = os.path.join(OUTPUT_DIR, f'{algorithm.lower().replace("-", "_")}_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Memory comparison plot (separate) with both linear and log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{algorithm} algorithm', fontsize=16, fontweight='bold')

    # Plot 1: Linear Scale (Memory)
    ax1.plot(julia_data['size'], julia_data['memory_mb'],
            marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
    ax1.plot(rust_data['size'], rust_data['memory_mb'],
            marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
    ax1.set_xlabel('Matrix Size', fontsize=12)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.set_title('Memory Usage (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(julia_data['size'])
    ax1.set_xticklabels(julia_data['size'])

    # Plot 2: Log Scale (Memory)
    ax2.plot(julia_data['size'], julia_data['memory_mb'],
            marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
    ax2.plot(rust_data['size'], rust_data['memory_mb'],
            marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
    ax2.set_xlabel('Matrix Size', fontsize=12)
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.set_title('Memory Usage (Log Scale)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xticks(julia_data['size'])
    ax2.set_xticklabels(julia_data['size'])

    plt.tight_layout()

    # Save memory plot
    output_file = os.path.join(OUTPUT_DIR, f'{algorithm.lower().replace("-", "_")}_memory.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

# Create a combined time comparison plot with Julia builtin
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('All Algorithms Comparison', fontsize=16, fontweight='bold')

# Plot 1: Linear Scale (Time)
for algorithm in ['Iterative', 'Divide-Conquer', 'Strassen']:
    julia_data = julia_df[julia_df['algorithm'] == algorithm]
    rust_data = rust_df[rust_df['algorithm'] == algorithm]

    ax1.plot(julia_data['size'], julia_data['time_ms'],
             marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
    ax1.plot(rust_data['size'], rust_data['time_ms'],
             marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

# Add Julia builtin to linear plot
julia_builtin = julia_df[julia_df['algorithm'] == 'Julia-Builtin']
ax1.plot(julia_builtin['size'], julia_builtin['time_ms'],
         marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
         color='green', linestyle='-')

ax1.set_xlabel('Matrix Size', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('Execution Time (Linear Scale)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Log Scale (Time)
for algorithm in ['Iterative', 'Divide-Conquer', 'Strassen']:
    julia_data = julia_df[julia_df['algorithm'] == algorithm]
    rust_data = rust_df[rust_df['algorithm'] == algorithm]

    ax2.plot(julia_data['size'], julia_data['time_ms'],
             marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
    ax2.plot(rust_data['size'], rust_data['time_ms'],
             marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

# Add Julia builtin to log plot
ax2.plot(julia_builtin['size'], julia_builtin['time_ms'],
         marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
         color='green', linestyle='-')

ax2.set_xlabel('Matrix Size', fontsize=12)
ax2.set_ylabel('Time (ms)', fontsize=12)
ax2.set_title('Execution Time (Log Scale)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xticks(julia_builtin['size'])
ax2.set_xticklabels(julia_builtin['size'])

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, 'all_algorithms_comparison.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Create a combined memory comparison plot with both linear and log scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('All Algorithms Memory Usage', fontsize=16, fontweight='bold')

# Plot 1: Linear Scale
for algorithm in ['Iterative', 'Divide-Conquer', 'Strassen']:
    julia_data = julia_df[julia_df['algorithm'] == algorithm]
    rust_data = rust_df[rust_df['algorithm'] == algorithm]

    ax1.plot(julia_data['size'], julia_data['memory_mb'],
            marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
    ax1.plot(rust_data['size'], rust_data['memory_mb'],
            marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

# Add Julia builtin memory to linear plot
ax1.plot(julia_builtin['size'], julia_builtin['memory_mb'],
        marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
        color='green', linestyle='-')

ax1.set_xlabel('Matrix Size', fontsize=12)
ax1.set_ylabel('Memory (MB)', fontsize=12)
ax1.set_title('Memory Usage (Linear Scale)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Log Scale
for algorithm in ['Iterative', 'Divide-Conquer', 'Strassen']:
    julia_data = julia_df[julia_df['algorithm'] == algorithm]
    rust_data = rust_df[rust_df['algorithm'] == algorithm]

    ax2.plot(julia_data['size'], julia_data['memory_mb'],
            marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
    ax2.plot(rust_data['size'], rust_data['memory_mb'],
            marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

# Add Julia builtin memory to log plot
ax2.plot(julia_builtin['size'], julia_builtin['memory_mb'],
        marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
        color='green', linestyle='-')

ax2.set_xlabel('Matrix Size', fontsize=12)
ax2.set_ylabel('Memory (MB)', fontsize=12)
ax2.set_title('Memory Usage (Log Scale)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xticks(julia_builtin['size'])
ax2.set_xticklabels(julia_builtin['size'])

plt.tight_layout()
output_file = os.path.join(OUTPUT_DIR, 'all_algorithms_memory.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

print("\nAll plots generated successfully!")
