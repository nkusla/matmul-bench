use crate::iterative_matmul::iterative_matmul;
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Strassen matrix multiplication algorithm with parallelization.
/// Uses 7 recursive multiplications instead of 8 by computing intermediate
/// products and combining them cleverly.
///
/// # Arguments
/// * `a` - First input matrix
/// * `b` - Second input matrix
/// * `threshold` - Minimum size to switch to standard multiplication
///
/// # Returns
/// Result matrix
pub fn strassen_matmul(a: &Matrix, b: &Matrix, threshold: usize) -> Matrix {
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
		panic!("strassen_matmul requires multiple threads to run");
	}

	strassen_recursive(a, b, threshold)
}

/// Internal recursive function for Strassen multiplication.
fn strassen_recursive(a: &Matrix, b: &Matrix, threshold: usize) -> Matrix {
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

	let mut a11_a22 = a11.clone();
	a11_a22.add(&a22);

	let mut b11_b22 = b11.clone();
	b11_b22.add(&b22);

	let mut a21_a22 = a21.clone();
	a21_a22.add(&a22);

	let mut b12_b22 = b12.clone();
	b12_b22.sub(&b22);

	let mut b21_b11 = b21.clone();
	b21_b11.sub(&b11);

	let mut a11_a12 = a11.clone();
	a11_a12.add(&a12);

	let mut a21_a11 = a21.clone();
	a21_a11.sub(&a11);

	let mut b11_b12 = b11.clone();
	b11_b12.add(&b12);

	let mut a12_a22 = a12.clone();
	a12_a22.sub(&a22);

	let mut b21_b22 = b21.clone();
	b21_b22.add(&b22);

	// Prepare all inputs for parallel computation
	let inputs: Vec<(&Matrix, &Matrix)> = vec![
		(&a11_a22, &b11_b22), // M1
		(&a21_a22, &b11),     // M2
		(&a11, &b12_b22),     // M3
		(&a22, &b21_b11),     // M4
		(&a11_a12, &b22),     // M5
		(&a21_a11, &b11_b12), // M6
		(&a12_a22, &b21_b22), // M7
	];

	// Parallel computation using rayon
	let results: Vec<Matrix> = inputs
		.into_par_iter()
		.map(|(a_sub, b_sub)| strassen_recursive(&a_sub, &b_sub, threshold))
		.collect();

	let [m1, m2, m3, mut m4, mut m5, mut m6, mut m7]: [Matrix; 7] = results.try_into().unwrap();

	// Combine the products to get result quadrants
	// C11 = M1 + M4 - M5 + M7
	// C12 = M3 + M5
	// C21 = M2 + M4
	// C22 = M1 - M2 + M3 + M6

	m7.add(&m1).add(&m4).sub(&m5);
	let c11 = m7;

	m5.add(&m3);
	let c12 = m5;

	m4.add(&m2);
	let c21 = m4;

	m6.add(&m1).sub(&m2).add(&m3);
	let c22 = m6;

	// Combine quadrants
	Matrix::combine_quadrants(&c11, &c12, &c21, &c22)
}
