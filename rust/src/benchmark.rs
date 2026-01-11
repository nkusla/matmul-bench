use crate::classic_matmul::classic_matmul;
use crate::divide_conquer_matmul::divide_conquer_matmul;
use crate::matrix::Matrix;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Structure to store benchmark results
#[derive(Clone)]
pub struct BenchmarkResult {
	pub size: usize,
	pub algorithm: String,
	pub time: f64,   // in milliseconds
	pub gflops: f64, // Giga FLOPS
}

/// Benchmark a single algorithm with given matrices.
/// Returns the median time in milliseconds.
fn benchmark_algorithm<F>(algorithm_fn: F, name: &str, a: &Matrix, b: &Matrix) -> f64
where
	F: Fn(&Matrix, &Matrix) -> Matrix,
{
	println!("  Benchmarking {}...", name);

	// Warmup run
	let _ = algorithm_fn(a, b);

	// Benchmark runs
	let samples = 10;
	let mut times = Vec::with_capacity(samples);

	for _ in 0..samples {
		let start = Instant::now();
		let _ = algorithm_fn(a, b);
		let duration = start.elapsed();
		times.push(duration.as_secs_f64() * 1000.0); // Convert to milliseconds
	}

	// Calculate median
	times.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let median_time = if times.len() % 2 == 0 {
		(times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
	} else {
		times[times.len() / 2]
	};

	median_time
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

		// Calculate FLOPS for this size (2n³ for matrix multiplication)
		let flops = 2.0 * (n as f64).powi(3);

		// Benchmark classic algorithm
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let time_classic = benchmark_algorithm(|a, b| classic_matmul(a, b), "Classic", &a, &b);
			let gflops_classic = flops / (time_classic / 1e3) / 1e9; // Convert ms to s for GFLOPS
			results.push(BenchmarkResult {
				size: n,
				algorithm: "Classic".to_string(),
				time: time_classic,
				gflops: gflops_classic,
			});
			println!(
				"    Time: {:.2} ms, Performance: {:.2} GFLOPS",
				time_classic, gflops_classic
			);
		})) {
			Err(e) => println!("    Error: {:?}", e),
			Ok(_) => {}
		}

		// Benchmark divide-and-conquer (always parallel)
		match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
			let time_dc = benchmark_algorithm(
				|a, b| divide_conquer_matmul(a, b, 64, true),
				"Divide-Conquer",
				&a,
				&b,
			);
			let gflops_dc = flops / (time_dc / 1e3) / 1e9; // Convert ms to s for GFLOPS
			results.push(BenchmarkResult {
				size: n,
				algorithm: "Divide-Conquer".to_string(),
				time: time_dc,
				gflops: gflops_dc,
			});
			println!(
				"    Time: {:.2} ms, Performance: {:.2} GFLOPS",
				time_dc, gflops_dc
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
		"{:<10} {:<25} {:>12} {:>12}",
		"Size", "Algorithm", "Time (ms)", "GFLOPS"
	);
	println!("{}", "-".repeat(80));

	for result in results {
		println!(
			"{:<10} {:<25} {:>12.2} {:>12.2}",
			result.size, result.algorithm, result.time, result.gflops
		);
	}

	println!("{}", "=".repeat(80));
}

/// Save benchmark results to a CSV file
pub fn save_results_csv(results: &Vec<BenchmarkResult>, filename: &str) -> std::io::Result<()> {
	let mut file = File::create(filename)?;
	writeln!(file, "size,algorithm,time_ms,gflops")?;

	for result in results {
		writeln!(
			file,
			"{},{},{},{}",
			result.size, result.algorithm, result.time, result.gflops
		)?;
	}

	println!("\nResults saved to: {}", filename);
	Ok(())
}
