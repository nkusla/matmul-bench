use crate::classic_matmul::classic_matmul;
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Divide and conquer matrix multiplication algorithm with optional parallelization.
/// Recursively divides matrices into quadrants until reaching the threshold,
/// then uses standard multiplication.
///
/// # Arguments
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `threshold` - Minimum size to switch to standard multiplication
/// * `parallel` - Whether to use parallel execution
///
/// # Returns
/// Result matrix
pub fn divide_conquer_matmul(a: &Matrix, b: &Matrix, threshold: usize, parallel: bool) -> Matrix {
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

	divide_conquer_recursive(a, b, threshold, parallel)
}

/// Internal recursive function for divide and conquer multiplication.
fn divide_conquer_recursive(a: &Matrix, b: &Matrix, threshold: usize, parallel: bool) -> Matrix {
	let m = a.rows;
	let n = a.cols;
	let p = b.cols;

	// Base case: use classic implementation for fair comparison
	if m <= threshold || n <= threshold || p <= threshold {
		return classic_matmul(a, b);
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

	// Compute the 8 products needed (C = A * B)
	// C11 = A11*B11 + A12*B21
	// C12 = A11*B12 + A12*B22
	// C21 = A21*B11 + A22*B21
	// C22 = A21*B12 + A22*B22

	let (c11, c12, c21, c22) = if parallel {
		// Parallel computation using rayon
		let results: Vec<Matrix> = vec![
			(&a11, &b11),
			(&a12, &b21),
			(&a11, &b12),
			(&a12, &b22),
			(&a21, &b11),
			(&a22, &b21),
			(&a21, &b12),
			(&a22, &b22),
		]
		.into_par_iter()
		.map(|(a_sub, b_sub)| divide_conquer_recursive(a_sub, b_sub, threshold, parallel))
		.collect();

		let c11 = results[0].add(&results[1]);
		let c12 = results[2].add(&results[3]);
		let c21 = results[4].add(&results[5]);
		let c22 = results[6].add(&results[7]);

		(c11, c12, c21, c22)
	} else {
		// Sequential computation
		let p1 = divide_conquer_recursive(&a11, &b11, threshold, parallel);
		let p2 = divide_conquer_recursive(&a12, &b21, threshold, parallel);
		let c11 = p1.add(&p2);

		let p3 = divide_conquer_recursive(&a11, &b12, threshold, parallel);
		let p4 = divide_conquer_recursive(&a12, &b22, threshold, parallel);
		let c12 = p3.add(&p4);

		let p5 = divide_conquer_recursive(&a21, &b11, threshold, parallel);
		let p6 = divide_conquer_recursive(&a22, &b21, threshold, parallel);
		let c21 = p5.add(&p6);

		let p7 = divide_conquer_recursive(&a21, &b12, threshold, parallel);
		let p8 = divide_conquer_recursive(&a22, &b22, threshold, parallel);
		let c22 = p7.add(&p8);

		(c11, c12, c21, c22)
	};

	// Combine quadrants
	Matrix::combine_quadrants(&c11, &c12, &c21, &c22)
}
