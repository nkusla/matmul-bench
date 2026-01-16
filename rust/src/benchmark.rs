use crate::iterative_matmul::iterative_matmul;
use crate::divide_conquer_matmul::divide_conquer_matmul;
use crate::strassen_matmul::strassen_matmul;
use crate::matrix::Matrix;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use stats_alloc::{Region, INSTRUMENTED_SYSTEM};

/// Structure to store benchmark results
#[derive(Clone)]
pub struct BenchmarkResult {
	pub size: usize,
	pub algorithm: String,
	pub time: f64,   // in milliseconds
	pub memory: f64, // in megabytes
}

/// Calculate the mean of a vector of f64 values.
fn calculate_mean(values: &[f64]) -> f64 {
	values.iter().sum::<f64>() / values.len() as f64
}

/// Benchmark a single algorithm with given matrices.
/// Returns the mean time in milliseconds and mean memory allocated in megabytes.
fn benchmark_algorithm<F>(algorithm_fn: F, name: &str, a: &Matrix, b: &Matrix) -> (f64, f64)
where
	F: Fn(&Matrix, &Matrix) -> Matrix,
{
	println!("  Benchmarking {}...", name);

	// Warmup run
	let _ = algorithm_fn(a, b);

	// Benchmark runs
	let samples = 10;
	let mut times = Vec::with_capacity(samples);
	let mut memory_allocations = Vec::with_capacity(samples);

	for _ in 0..samples {
		let reg: Region<'_, std::alloc::System> = Region::new(&INSTRUMENTED_SYSTEM);
		let start = Instant::now();
		let _ = algorithm_fn(a, b);
		let duration = start.elapsed();
		let stats = reg.change();

		times.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
		memory_allocations.push(stats.bytes_allocated as f64 / 1e6); // Convert to megabytes
	}

	let mean_time = calculate_mean(&times);
	let mean_memory = calculate_mean(&memory_allocations);

	(mean_time, mean_memory)
}

/// Run benchmarks for different matrix sizes and algorithms.
/// Returns a vector of BenchmarkResult objects.
pub fn run_benchmarks(sizes: &Vec<usize>) -> Vec<BenchmarkResult> {
	let mut results = Vec::new();

	for &n in sizes {
		println!("\nTesting matrix size: {}x{}", n, n);
		println!("{}", "=".repeat(50));

		// Generate random matrices
		let a = Matrix::random(n, n);
		let b = Matrix::random(n, n);

		// Benchmark iterative algorithm
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let (time_iterative, memory_iterative) = benchmark_algorithm(|a, b| iterative_matmul(a, b), "Iterative", &a, &b);
			results.push(BenchmarkResult {
				size: n,
				algorithm: "Iterative".to_string(),
				time: time_iterative,
				memory: memory_iterative,
			});
			println!(
				"    Time: {:.2} ms, Memory: {:.2} MB",
				time_iterative, memory_iterative
			);
		})) {
			Err(e) => println!("    Error: {:?}", e),
			Ok(_) => {}
		}

		// Benchmark divide-and-conquer (always parallel)
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let (time_dc, memory_dc) = benchmark_algorithm(
				|a, b| divide_conquer_matmul(a, b, 64, true),
				"Divide-Conquer",
				&a,
				&b,
			);
			results.push(BenchmarkResult {
				size: n,
				algorithm: "Divide-Conquer".to_string(),
				time: time_dc,
				memory: memory_dc,
			});
			println!(
				"    Time: {:.2} ms, Memory: {:.2} MB",
				time_dc, memory_dc
			);
		})) {
			Err(e) => println!("    Error: {:?}", e),
			Ok(_) => {}
		}

		// Benchmark Strassen (always parallel)
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let (time_strassen, memory_strassen) = benchmark_algorithm(
				|a, b| strassen_matmul(a, b, 64, true),
				"Strassen",
				&a,
				&b,
			);
			results.push(BenchmarkResult {
				size: n,
				algorithm: "Strassen".to_string(),
				time: time_strassen,
				memory: memory_strassen,
			});
			println!(
				"    Time: {:.2} ms, Memory: {:.2} MB",
				time_strassen, memory_strassen
			);
		})) {
			Err(e) => println!("    Error: {:?}", e),
			Ok(_) => {}
		}
	}

	results
}

/// Print benchmark results in a formatted table
pub fn print_results_table(results: &Vec<BenchmarkResult>) {
	println!("\n{}", "=".repeat(80));
	println!("BENCHMARK RESULTS SUMMARY");
	println!("{}", "=".repeat(80));
	println!(
		"{:<10} {:<25} {:>12} {:>15}",
		"Size", "Algorithm", "Time (ms)", "Memory (MB)"
	);
	println!("{}", "-".repeat(80));

	for result in results {
		println!(
			"{:<10} {:<25} {:>12.2} {:>15.2}",
			result.size, result.algorithm, result.time, result.memory
		);
	}

	println!("{}", "=".repeat(80));
}

/// Save benchmark results to a CSV file
pub fn save_results_csv(results: &Vec<BenchmarkResult>, filename: &str) -> std::io::Result<()> {
	let mut file = File::create(filename)?;
	writeln!(file, "size,algorithm,time_ms,memory_mb")?;

	for result in results {
		writeln!(
			file,
			"{},{},{},{}",
			result.size, result.algorithm, result.time, result.memory
		)?;
	}

	println!("\nResults saved to: {}", filename);
	Ok(())
}
