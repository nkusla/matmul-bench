use crate::iterative_matmul::iterative_matmul;
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Divide and conquer matrix multiplication algorithm with parallelization.
/// Recursively divides matrices into quadrants until reaching the threshold,
/// then uses standard multiplication.
///
/// # Arguments
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `threshold` - Minimum size to switch to standard multiplication
///
/// # Returns
/// Result matrix
pub fn divide_conquer_matmul(a: &Matrix, b: &Matrix, threshold: usize) -> Matrix {
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

	if rayon::current_num_threads() <= 1 {
		panic!("divide_conquer_matmul requires multiple threads to run");
	}

	divide_conquer_recursive(a, b, threshold)
}

/// Internal recursive function for divide and conquer multiplication.
fn divide_conquer_recursive(a: &Matrix, b: &Matrix, threshold: usize) -> Matrix {
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

	// Compute the 8 products needed (C = A * B)
	// C11 = A11*B11 + A12*B21
	// C12 = A11*B12 + A12*B22
	// C21 = A21*B11 + A22*B21
	// C22 = A21*B12 + A22*B22

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
	.map(|(a_sub, b_sub)| divide_conquer_recursive(a_sub, b_sub, threshold))
	.collect();

	let [mut c11,
		r1,
		mut c12,
		r3,
		mut c21,
		r5,
		mut c22,
		r7]: [Matrix; 8] =
		results.try_into().unwrap();

	c11.add(&r1);
	c12.add(&r3);
	c21.add(&r5);
	c22.add(&r7);

	// Combine quadrants
	Matrix::combine_quadrants(&c11, &c12, &c21, &c22)
}
