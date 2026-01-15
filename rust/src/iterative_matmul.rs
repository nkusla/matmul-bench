use crate::matrix::Matrix;

/// Standard iterative matrix multiplication algorithm.
/// Time complexity: O(n³)
///
/// # Arguments
/// * `a` - First input matrix (m x n)
/// * `b` - Second input matrix (n x p)
///
/// # Returns
/// Result matrix (m x p)
///
/// # Panics
/// Panics if matrix dimensions don't match (columns of A != rows of B)
pub fn iterative_matmul(a: &Matrix, b: &Matrix) -> Matrix {
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

	let mut c = Matrix::new(m, p);

	// Column major computation (optimized for contiguous memory layout)
	// Access memory sequentially by iterating columns in the innermost loop
	for j in 0..p {
		for k in 0..n {
			for i in 0..m {
				c[(i, j)] += a[(i, k)] * b[(k, j)];
			}
		}
	}

	c
}
