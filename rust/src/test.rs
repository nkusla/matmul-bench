mod classic_matmul;
mod divide_conquer_matmul;
mod matrix;

use classic_matmul::classic_matmul;
use divide_conquer_matmul::divide_conquer_matmul;
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

	// Test with small matrices
	let sizes_to_test = vec![4, 8, 16, 32, 64];

	for n in sizes_to_test {
		println!("\nTesting size: {}x{}", n, n);

		// Create random test matrices
		let a = Matrix::random(n, n);
		let b = Matrix::random(n, n);

		// Compute reference result
		let c_reference = reference_matmul(&a, &b);

		// Test classic multiplication
		let c_classic = classic_matmul(&a, &b);
		let error_classic = matrix_error(&c_reference, &c_classic);
		assert!(
			error_classic < 1e-10,
			"Classic multiplication error too large"
		);
		println!("  ✓ Classic: error = {}", error_classic);

		// Test divide-and-conquer
		let c_dc_par = divide_conquer_matmul(&a, &b, 4, true);
		let error_dc_par = matrix_error(&c_reference, &c_dc_par);
		assert!(
			error_dc_par < 1e-10,
			"Divide-conquer multiplication error too large"
		);
		println!("  ✓ Divide-Conquer: error = {}", error_dc_par);
	}

	println!("\n{}", "=".repeat(50));
	println!("All tests passed! ✓");
}
