use crate::iterative_matmul::iterative_matmul;
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Strassen matrix multiplication algorithm with optional parallelization.
/// Uses 7 recursive multiplications instead of 8 by computing intermediate
/// products and combining them cleverly.
///
/// # Arguments
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `threshold` - Minimum size to switch to standard multiplication
/// * `parallel` - Whether to use parallel execution
///
/// # Returns
/// Result matrix
pub fn strassen_matmul(a: &Matrix, b: &Matrix, threshold: usize, parallel: bool) -> Matrix {
	let m = a.rows;
	let n = a.cols;
	let q = b.rows;
	let p = b.cols;

	if n != q {
		panic!(
			"Matrix dimensions must agree: A is {}x{}, B is {}x{}",
			m, n, q, p
		);
	}

	strassen_recursive(a, b, threshold, parallel)
}

/// Internal recursive function for Strassen multiplication.
fn strassen_recursive(a: &Matrix, b: &Matrix, threshold: usize, parallel: bool) -> Matrix {
	let m = a.rows;
	let n = a.cols;
	let p = b.cols;

	// Base case: use iterative implementation for fair comparison
	if m <= threshold || n <= threshold || p <= threshold {
		return iterative_matmul(a, b);
	}

	// Divide matrices into quadrants
	let m_half = m / 2;
	let n_half = n / 2;
	let p_half = p / 2;

	// Divide A into quadrants
	let a11 = a.submatrix(0, m_half, 0, n_half);
	let a12 = a.submatrix(0, m_half, n_half, n);
	let a21 = a.submatrix(m_half, m, 0, n_half);
	let a22 = a.submatrix(m_half, m, n_half, n);

	// Divide B into quadrants
	let b11 = b.submatrix(0, n_half, 0, p_half);
	let b12 = b.submatrix(0, n_half, p_half, p);
	let b21 = b.submatrix(n_half, n, 0, p_half);
	let b22 = b.submatrix(n_half, n, p_half, p);

	// Compute the 7 Strassen products
	// M1 = (A11 + A22) * (B11 + B22)
	// M2 = (A21 + A22) * B11
	// M3 = A11 * (B12 - B22)
	// M4 = A22 * (B21 - B11)
	// M5 = (A11 + A12) * B22
	// M6 = (A21 - A11) * (B11 + B12)
	// M7 = (A12 - A22) * (B21 + B22)

	let (m1, m2, m3, m4, m5, m6, m7) = if parallel {
		// Prepare all inputs for parallel computation
		let inputs: Vec<(Matrix, Matrix)> = vec![
			(a11.add(&a22), b11.add(&b22)), // M1
			(a21.add(&a22), b11.clone()),   // M2
			(a11.clone(), b12.sub(&b22)),   // M3
			(a22.clone(), b21.sub(&b11)),   // M4
			(a11.add(&a12), b22.clone()),   // M5
			(a21.sub(&a11), b11.add(&b12)), // M6
			(a12.sub(&a22), b21.add(&b22)), // M7
		];

		// Parallel computation using rayon
		let results: Vec<Matrix> = inputs
			.into_par_iter()
			.map(|(a_sub, b_sub)| strassen_recursive(&a_sub, &b_sub, threshold, parallel))
			.collect();

		(
			results[0].clone(),
			results[1].clone(),
			results[2].clone(),
			results[3].clone(),
			results[4].clone(),
			results[5].clone(),
			results[6].clone(),
		)
	} else {
		// Sequential computation
		let m1 = strassen_recursive(&a11.add(&a22), &b11.add(&b22), threshold, parallel);
		let m2 = strassen_recursive(&a21.add(&a22), &b11, threshold, parallel);
		let m3 = strassen_recursive(&a11, &b12.sub(&b22), threshold, parallel);
		let m4 = strassen_recursive(&a22, &b21.sub(&b11), threshold, parallel);
		let m5 = strassen_recursive(&a11.add(&a12), &b22, threshold, parallel);
		let m6 = strassen_recursive(&a21.sub(&a11), &b11.add(&b12), threshold, parallel);
		let m7 = strassen_recursive(&a12.sub(&a22), &b21.add(&b22), threshold, parallel);

		(m1, m2, m3, m4, m5, m6, m7)
	};

	// Combine the products to get result quadrants
	// C11 = M1 + M4 - M5 + M7
	// C12 = M3 + M5
	// C21 = M2 + M4
	// C22 = M1 - M2 + M3 + M6

	let c11 = m1.add(&m4).sub(&m5).add(&m7);
	let c12 = m3.add(&m5);
	let c21 = m2.add(&m4);
	let c22 = m1.sub(&m2).add(&m3).add(&m6);

	// Combine quadrants
	Matrix::combine_quadrants(&c11, &c12, &c21, &c22)
}
