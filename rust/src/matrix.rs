use std::ops::{Index, IndexMut};
use rand::Rng;

/// Matrix stored in contiguous memory (column-major order)
#[derive(Clone, Debug)]
pub struct Matrix {
	pub data: Vec<f64>,
	pub rows: usize,
	pub cols: usize,
}

impl Matrix {
	/// Create a new matrix with given dimensions, initialized to zero
	pub fn new(rows: usize, cols: usize) -> Self {
		Matrix {
			data: vec![0.0; rows * cols],
			rows,
			cols,
		}
	}

	/// Create a random matrix with values in [0, 1)
	pub fn random(rows: usize, cols: usize) -> Self {
		let mut rng = rand::thread_rng();
		let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen::<f64>()).collect();
		Matrix { data, rows, cols }
	}

	/// Extract a submatrix (creates a copy)
	pub fn submatrix(
		&self,
		row_start: usize,
		row_end: usize,
		col_start: usize,
		col_end: usize,
	) -> Matrix {
		let new_rows = row_end - row_start;
		let new_cols = col_end - col_start;
		let mut result = Matrix::new(new_rows, new_cols);

		for i in 0..new_rows {
			for j in 0..new_cols {
				result[(i, j)] = self[(row_start + i, col_start + j)];
			}
		}

		result
	}

	/// Add another matrix in-place (modifies self)
	pub fn add(&mut self, other: &Matrix) -> &mut Self {
		assert_eq!(self.rows, other.rows);
		assert_eq!(self.cols, other.cols);

		for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
			*a += b;
		}
		self
	}

	/// Subtract another matrix in-place (modifies self)
	pub fn sub(&mut self, other: &Matrix) -> &mut Self {
		assert_eq!(self.rows, other.rows);
		assert_eq!(self.cols, other.cols);

		for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
			*a -= b;
		}
		self
	}

	/// Combine four quadrant matrices into a single matrix
	pub fn combine_quadrants(c11: &Matrix, c12: &Matrix, c21: &Matrix, c22: &Matrix) -> Matrix {
		let rows_top = c11.rows;
		let rows_bottom = c21.rows;
		let cols_left = c11.cols;
		let cols_right = c12.cols;

		let mut result = Matrix::new(rows_top + rows_bottom, cols_left + cols_right);

		// Copy C11
		for i in 0..c11.rows {
			for j in 0..c11.cols {
				result[(i, j)] = c11[(i, j)];
			}
		}

		// Copy C12
		for i in 0..c12.rows {
			for j in 0..c12.cols {
				result[(i, cols_left + j)] = c12[(i, j)];
			}
		}

		// Copy C21
		for i in 0..c21.rows {
			for j in 0..c21.cols {
				result[(rows_top + i, j)] = c21[(i, j)];
			}
		}

		// Copy C22
		for i in 0..c22.rows {
			for j in 0..c22.cols {
				result[(rows_top + i, cols_left + j)] = c22[(i, j)];
			}
		}

		result
	}
}

// Implement indexing: matrix[(row, col)]
impl Index<(usize, usize)> for Matrix {
	type Output = f64;

	#[inline]
	fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
		&self.data[col * self.rows + row]
	}
}

// Implement mutable indexing: matrix[(row, col)] = value
impl IndexMut<(usize, usize)> for Matrix {
	#[inline]
	fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
		&mut self.data[col * self.rows + row]
	}
}
