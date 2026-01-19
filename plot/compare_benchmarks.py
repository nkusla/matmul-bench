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
RESULTS_DIR = "../results/data"
OUTPUT_DIR = "../results/plots/"

def load_benchmark_data(results_dir):
	"""Load benchmark CSV files for Julia and Rust."""
	julia_files = glob.glob(os.path.join(results_dir, "julia_benchmark_*.csv"))
	rust_files = glob.glob(os.path.join(results_dir, "rust_benchmark_*.csv"))

	if not julia_files or not rust_files:
		print("Error: Could not find benchmark CSV files")
		exit(1)

	julia_df = pd.read_csv(julia_files[0])
	rust_df = pd.read_csv(rust_files[0])

	julia_df['language'] = 'Julia'
	rust_df['language'] = 'Rust'

	return julia_df, rust_df, julia_files[0], rust_files[0]

def extract_thread_count(filename):
	"""Extract thread count from benchmark filename."""
	threads = re.search(r'_(\d+)t_', filename)
	return threads.group(1) if threads else "unknown"

def create_algorithm_time_plot(algorithm, julia_data, rust_data, julia_threads, output_dir):
	"""Create time comparison plot for a specific algorithm."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

	if algorithm == 'Iterative':
		title = f'{algorithm} algorithm'
	else:
		title = f'{algorithm} algorithm ({julia_threads} threads)'

	fig.suptitle(title, fontsize=16, fontweight='bold')

	julia_label = 'Julia'
	rust_label = 'Rust'

	# Plot 1: Linear Scale
	ax1.plot(julia_data['size'], julia_data['time_ms'],
			 marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
	ax1.plot(rust_data['size'], rust_data['time_ms'],
			 marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
	ax1.set_xlabel('Matrix Size', fontsize=12)
	ax1.set_ylabel('Time (ms)', fontsize=12)
	ax1.set_title('Execution Time (Linear Scale)', fontsize=13, fontweight='bold')
	ax1.legend(fontsize=11)
	ax1.grid(True, alpha=0.3)
	# ax1.set_xticks(julia_data['size'])
	# ax1.set_xticklabels(julia_data['size'])

	# Plot 2: Log Scale
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
	# ax2.set_xticks(julia_data['size'])
	# ax2.set_xticklabels(julia_data['size'])

	plt.tight_layout()

	output_file = os.path.join(output_dir, f'{algorithm.lower().replace("-", "_")}_comparison.png')
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"Saved: {output_file}")
	plt.close()

def create_algorithm_memory_plot(algorithm, julia_data, rust_data, output_dir):
	"""Create memory comparison plot for a specific algorithm."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	fig.suptitle(f'{algorithm} algorithm', fontsize=16, fontweight='bold')

	julia_label = 'Julia'
	rust_label = 'Rust'

	# Plot 1: Linear Scale
	ax1.plot(julia_data['size'], julia_data['memory_mb'],
			marker='o', linewidth=2, markersize=8, label=julia_label, color='#9558B2')
	ax1.plot(rust_data['size'], rust_data['memory_mb'],
			marker='s', linewidth=2, markersize=8, label=rust_label, color='#CE422B')
	ax1.set_xlabel('Matrix Size', fontsize=12)
	ax1.set_ylabel('Memory (MB)', fontsize=12)
	ax1.set_title('Memory Usage (Linear Scale)', fontsize=13, fontweight='bold')
	ax1.legend(fontsize=11)
	ax1.grid(True, alpha=0.3)
	# ax1.set_xticks(julia_data['size'])
	# ax1.set_xticklabels(julia_data['size'])

	# Plot 2: Log Scale
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
	# ax2.set_xticks(julia_data['size'])
	# ax2.set_xticklabels(julia_data['size'])

	plt.tight_layout()

	output_file = os.path.join(output_dir, f'{algorithm.lower().replace("-", "_")}_memory.png')
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"Saved: {output_file}")
	plt.close()

def create_individual_algorithm_plots(julia_df, rust_df, julia_threads, output_dir):
	"""Create individual plots for each algorithm."""
	algorithms = ['Iterative', 'Divide-Conquer', 'Strassen']

	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm]
		rust_data = rust_df[rust_df['algorithm'] == algorithm]

		create_algorithm_time_plot(algorithm, julia_data, rust_data, julia_threads, output_dir)
		create_algorithm_memory_plot(algorithm, julia_data, rust_data, output_dir)

def create_combined_time_plot(julia_df, rust_df, output_dir):
	"""Create combined time comparison plot for all algorithms."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	fig.suptitle('All Algorithms Comparison', fontsize=16, fontweight='bold')

	algorithms = ['Iterative', 'Divide-Conquer', 'Strassen']
	julia_builtin = julia_df[julia_df['algorithm'] == 'Julia-Builtin']

	# Plot 1: Linear Scale
	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm]
		rust_data = rust_df[rust_df['algorithm'] == algorithm]

		ax1.plot(julia_data['size'], julia_data['time_ms'],
				 marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
		ax1.plot(rust_data['size'], rust_data['time_ms'],
				 marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

	ax1.plot(julia_builtin['size'], julia_builtin['time_ms'],
			 marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
			 color='green', linestyle='-')

	ax1.set_xlabel('Matrix Size', fontsize=12)
	ax1.set_ylabel('Time (ms)', fontsize=12)
	ax1.set_title('Execution Time (Linear Scale)', fontsize=13, fontweight='bold')
	ax1.legend(fontsize=9)
	ax1.grid(True, alpha=0.3)

	# Plot 2: Log Scale
	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm]
		rust_data = rust_df[rust_df['algorithm'] == algorithm]

		ax2.plot(julia_data['size'], julia_data['time_ms'],
				 marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
		ax2.plot(rust_data['size'], rust_data['time_ms'],
				 marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

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
	# ax2.set_xticks(julia_builtin['size'])
	# ax2.set_xticklabels(julia_builtin['size'])

	plt.tight_layout()
	output_file = os.path.join(output_dir, 'all_algorithms_comparison.png')
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"Saved: {output_file}")
	plt.close()

def create_combined_memory_plot(julia_df, rust_df, output_dir):
	"""Create combined memory comparison plot for all algorithms."""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	fig.suptitle('All Algorithms Memory Usage', fontsize=16, fontweight='bold')

	algorithms = ['Iterative', 'Divide-Conquer', 'Strassen']
	julia_builtin = julia_df[julia_df['algorithm'] == 'Julia-Builtin']

	# Plot 1: Linear Scale
	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm]
		rust_data = rust_df[rust_df['algorithm'] == algorithm]

		ax1.plot(julia_data['size'], julia_data['memory_mb'],
				marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
		ax1.plot(rust_data['size'], rust_data['memory_mb'],
				marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

	ax1.plot(julia_builtin['size'], julia_builtin['memory_mb'],
			marker='^', linewidth=2, markersize=6, label='Julia Builtin (BLAS)',
			color='green', linestyle='-')

	ax1.set_xlabel('Matrix Size', fontsize=12)
	ax1.set_ylabel('Memory (MB)', fontsize=12)
	ax1.set_title('Memory Usage (Linear Scale)', fontsize=13, fontweight='bold')
	ax1.legend(fontsize=9)
	ax1.grid(True, alpha=0.3)

	# Plot 2: Log Scale
	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm]
		rust_data = rust_df[rust_df['algorithm'] == algorithm]

		ax2.plot(julia_data['size'], julia_data['memory_mb'],
				marker='o', linewidth=2, markersize=6, label=f'Julia {algorithm}', linestyle='-')
		ax2.plot(rust_data['size'], rust_data['memory_mb'],
				marker='s', linewidth=2, markersize=6, label=f'Rust {algorithm}', linestyle='--')

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
	# ax2.set_xticks(julia_builtin['size'])
	# ax2.set_xticklabels(julia_builtin['size'])

	plt.tight_layout()
	output_file = os.path.join(output_dir, 'all_algorithms_memory.png')
	plt.savefig(output_file, dpi=300, bbox_inches='tight')
	print(f"Saved: {output_file}")
	plt.close()

def create_algorithm_comparison_table(algorithm, julia_data, rust_data, output_dir):
	"""Create comparison table for a specific algorithm."""
	# Merge data by size
	merged = pd.merge(
		julia_data[['size', 'time_ms', 'memory_mb']],
		rust_data[['size', 'time_ms', 'memory_mb']],
		on='size',
		suffixes=('_julia', '_rust')
	)

	# Calculate speedup (Julia time / Rust time)
	merged['speedup'] = merged['time_ms_julia'] / merged['time_ms_rust']

	# Calculate memory ratio (Julia memory / Rust memory)
	merged['memory_ratio'] = merged['memory_mb_julia'] / merged['memory_mb_rust']

	# Create table data with proper column names
	table_data = pd.DataFrame({
		'Matrix size': merged['size'].astype(int).astype(str),
		'Julia Time (ms)': merged['time_ms_julia'].round(2),
		'Rust Time (ms)': merged['time_ms_rust'].round(2),
		'Speedup (J/R)': merged['speedup'].round(2),
		'Julia Memory (MB)': merged['memory_mb_julia'].round(2),
		'Rust Memory (MB)': merged['memory_mb_rust'].round(2),
		'Memory Ratio (J/R)': merged['memory_ratio'].round(2)
	})

	# Create figure with no axes - larger size
	fig, ax = plt.subplots(figsize=(20, len(table_data) * 1.4 + 1.5))
	ax.axis('off')

	# Create table - using values and columns to exclude row index
	table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
					 loc='center', cellLoc='center', colWidths=[0.16] * len(table_data.columns))

	# Style the table
	table.auto_set_font_size(False)
	table.set_fontsize(16)
	table.scale(1.0, 3.5)

	# Color header row
	for i in range(len(table_data.columns)):
		cell = table[(0, i)]
		cell.set_facecolor('#4472C4')
		cell.set_text_props(weight='bold', color='white')

	# Alternate row colors
	for i in range(1, len(table_data) + 1):
		for j in range(len(table_data.columns)):
			cell = table[(i, j)]
			if i % 2 == 0:
				cell.set_facecolor('#E7E6E6')
			else:
				cell.set_facecolor('#FFFFFF')

	# Add title
	plt.title(f'{algorithm} Algorithm - Performance Comparison',
			  fontsize=18, fontweight='bold', pad=15)

	plt.tight_layout(pad=0.5)
	output_file = os.path.join(output_dir, f'{algorithm.lower().replace("-", "_")}_table.png')
	plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
	print(f"Saved: {output_file}")
	plt.close()

def create_comparison_tables(julia_df, rust_df, output_dir):
	"""Create comparison tables for each algorithm."""
	algorithms = ['Iterative', 'Divide-Conquer', 'Strassen']

	for algorithm in algorithms:
		julia_data = julia_df[julia_df['algorithm'] == algorithm].copy()
		rust_data = rust_df[rust_df['algorithm'] == algorithm].copy()

		if len(julia_data) > 0 and len(rust_data) > 0:
			create_algorithm_comparison_table(algorithm, julia_data, rust_data, output_dir)

def main():
	"""Main function to orchestrate the benchmark comparison."""
	# Load data
	julia_df, rust_df, julia_file, rust_file = load_benchmark_data(RESULTS_DIR)

	# Extract thread counts
	julia_threads = extract_thread_count(julia_file)
	rust_threads = extract_thread_count(rust_file)

	# Generate plots
	create_individual_algorithm_plots(julia_df, rust_df, julia_threads, OUTPUT_DIR)
	create_combined_time_plot(julia_df, rust_df, OUTPUT_DIR)
	create_combined_memory_plot(julia_df, rust_df, OUTPUT_DIR)

	# Generate comparison tables
	create_comparison_tables(julia_df, rust_df, OUTPUT_DIR)

	print("\nAll plots and tables generated successfully!")

if __name__ == "__main__":
	main()
