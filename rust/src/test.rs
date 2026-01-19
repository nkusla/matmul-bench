mod iterative_matmul;
mod divide_conquer_matmul;
mod strassen_matmul;
mod matrix;

use iterative_matmul::iterative_matmul;
use divide_conquer_matmul::divide_conquer_matmul;
use strassen_matmul::strassen_matmul;
use matrix::Matrix;

/// Calculate Frobenius norm of difference with another matrix
fn matrix_error(a: &Matrix, b: &Matrix) -> f64 {
	assert_eq!(a.rows, b.rows);
	assert_eq!(a.cols, b.cols);

	let sum: f64 = a
		.data
		.iter()
		.zip(b.data.iter())
		.map(|(a, b)| {
			let diff = a - b;
			diff * diff
		})
		.sum();

	sum.sqrt()
}

/// Naive reference matrix multiplication for testing
fn reference_matmul(a: &Matrix, b: &Matrix) -> Matrix {
	let m = a.rows;
	let n = a.cols;
	let p = b.cols;
	let mut c = Matrix::new(m, p);

	for i in 0..m {
		for j in 0..p {
			for k in 0..n {
				c[(i, j)] += a[(i, k)] * b[(k, j)];
			}
		}
	}

	c
}

fn main() {
	println!("Running correctness tests...");
	println!("{}", "=".repeat(50));

	// Test with small matrices (including odd sizes)
	let sizes_to_test = vec![50, 64, 100, 128, 150];

	for n in sizes_to_test {
		println!("\nTesting size: {}x{}", n, n);

		// Create random test matrices
		let a = Matrix::random(n, n);
		let b = Matrix::random(n, n);

		// Compute reference result
		let c_reference = reference_matmul(&a, &b);

		// Test iterative multiplication
		let c_iterative = iterative_matmul(&a, &b);
		let error_iterative = matrix_error(&c_reference, &c_iterative);
		assert!(
			error_iterative < 1e-10,
			"Iterative multiplication error too large"
		);
		println!("  ✓ Iterative: error = {}", error_iterative);

		// Test divide-and-conquer
		let c_dc_par = divide_conquer_matmul(&a, &b, 4);
		let error_dc_par = matrix_error(&c_reference, &c_dc_par);
		assert!(
			error_dc_par < 1e-10,
			"Divide-conquer multiplication error too large"
		);
		println!("  ✓ Divide-Conquer: error = {}", error_dc_par);

		// Test Strassen
		let c_strassen = strassen_matmul(&a, &b, 4);
		let error_strassen = matrix_error(&c_reference, &c_strassen);
		assert!(
			error_strassen < 1e-10,
			"Strassen multiplication error too large"
		);
		println!("  ✓ Strassen: error = {}", error_strassen);
	}

	println!("\n{}", "=".repeat(50));
	println!("All tests passed! ✓");
}
