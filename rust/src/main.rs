mod benchmark;
mod iterative_matmul;
mod divide_conquer_matmul;
mod strassen_matmul;
mod matrix;

use benchmark::{print_results_table, run_benchmarks, save_results_csv};
use std::time::Instant;
use stats_alloc::{StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Main function to run matrix multiplication benchmarks
fn main() {
	println!("{}", "=".repeat(80));
	println!("Matrix Multiplication Benchmark - Rust Implementation");
	println!("{}", "=".repeat(80));

	// Get number of threads (rayon uses all available cores by default)
	let num_threads = rayon::current_num_threads();
	println!("Number of threads: {}", num_threads);
	println!();

	// Define matrix sizes to test
	// Start small and scale up
	let sizes = vec![
		64, 128, 256, 512, 1024, 2048,
	];

	println!("Testing sizes: {:?}", sizes);
	println!();

	// Run benchmarks
	let start_time = Instant::now();
	let results = run_benchmarks(&sizes);
	let elapsed_time = start_time.elapsed();

	// Display results
	print_results_table(&results);

	// Save results to CSV
	let timestamp = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap()
		.as_secs();

	let os = std::env::consts::OS;
	let arch = std::env::consts::ARCH;
	let filename = format!("../results/data/rust_benchmark_{os}_{arch}_{num_threads}t_{timestamp}.csv");

	if let Err(e) = save_results_csv(&results, &filename) {
		eprintln!("Error saving results: {}", e);
	}

	println!("\nBenchmark completed!");
	println!(
		"Total elapsed time: {:.2} seconds",
		elapsed_time.as_secs_f64()
	);
}
