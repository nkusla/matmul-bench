using Base.Threads

include("iterative_matmul.jl")

"""
    strassen_matmul(A::Matrix{T}, B::Matrix{T}; threshold::Int=64, parallel::Bool=true) where T

Strassen's matrix multiplication algorithm with optional parallelization.
Recursively divides matrices into quadrants and computes 7 products instead of 8,
reducing complexity from O(n³) to O(n^2.807).

Arguments:
- A, B: Input matrices to multiply
- threshold: Minimum size to switch to standard multiplication (default: 64)
- parallel: Whether to use parallel execution (default: true)
"""
function strassen_matmul(A::Matrix{T}, B::Matrix{T}; threshold::Int=64, parallel::Bool=true) where T
  m, n = size(A)
  q, p = size(B)

  if n != q
    throw(DimensionMismatch("Matrix dimensions must agree: A is $(m)x$(n), B is $(q)x$(p)"))
  end

  return _strassen_recursive(A, B, threshold, parallel)
end

"""
    _strassen_recursive(A, B, threshold, parallel)

Internal recursive function for Strassen multiplication.
"""
function _strassen_recursive(A::Matrix{T}, B::Matrix{T}, threshold::Int, parallel::Bool) where T
  m, n = size(A)
  q, p = size(B)

  # Base case: use iterative multiplication for small matrices
  if m <= threshold || n <= threshold || p <= threshold
    return iterative_matmul(A, B)
  end

  # Pad matrices to even dimensions if necessary
  m_even = m + (m % 2)
  n_even = n + (n % 2)
  p_even = p + (p % 2)

  # Pad if necessary
  if m != m_even || n != n_even
    A_padded = zeros(T, m_even, n_even)
    A_padded[1:m, 1:n] = A
    A = A_padded
  end

  if n != n_even || p != p_even
    B_padded = zeros(T, n_even, p_even)
    B_padded[1:n, 1:p] = B
    B = B_padded
  end

  # Update dimensions after padding
  m, n = m_even, n_even
  p = p_even

  # Calculate split points
  m_half = m ÷ 2
  n_half = n ÷ 2
  p_half = p ÷ 2

  # Divide A into quadrants
  A11 = A[1:m_half, 1:n_half]
  A12 = A[1:m_half, n_half+1:n]
  A21 = A[m_half+1:m, 1:n_half]
  A22 = A[m_half+1:m, n_half+1:n]

  # Divide B into quadrants
  B11 = B[1:n_half, 1:p_half]
  B12 = B[1:n_half, p_half+1:p]
  B21 = B[n_half+1:n, 1:p_half]
  B22 = B[n_half+1:n, p_half+1:p]

  # Strassen's algorithm: compute 7 products instead of 8
  # M1 = (A11 + A22) * (B11 + B22)
  # M2 = (A21 + A22) * B11
  # M3 = A11 * (B12 - B22)
  # M4 = A22 * (B21 - B11)
  # M5 = (A11 + A12) * B22
  # M6 = (A21 - A11) * (B11 + B12)
  # M7 = (A12 - A22) * (B21 + B22)
  #
  # C11 = M1 + M4 - M5 + M7
  # C12 = M3 + M5
  # C21 = M2 + M4
  # C22 = M1 - M2 + M3 + M6

  if parallel && Threads.nthreads() > 1
    # Parallel computation using threads
    # Spawn 7 tasks for the 7 Strassen products
    t1 = Threads.@spawn _strassen_recursive(A11 + A22, B11 + B22, threshold, parallel)
    t2 = Threads.@spawn _strassen_recursive(A21 + A22, B11, threshold, parallel)
    t3 = Threads.@spawn _strassen_recursive(A11, B12 - B22, threshold, parallel)
    t4 = Threads.@spawn _strassen_recursive(A22, B21 - B11, threshold, parallel)
    t5 = Threads.@spawn _strassen_recursive(A11 + A12, B22, threshold, parallel)
    t6 = Threads.@spawn _strassen_recursive(A21 - A11, B11 + B12, threshold, parallel)
    t7 = Threads.@spawn _strassen_recursive(A12 - A22, B21 + B22, threshold, parallel)

    # Fetch results
    M1 = fetch(t1)
    M2 = fetch(t2)
    M3 = fetch(t3)
    M4 = fetch(t4)
    M5 = fetch(t5)
    M6 = fetch(t6)
    M7 = fetch(t7)

    # Combine results to form C
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
  else
    # Sequential computation
    M1 = _strassen_recursive(A11 + A22, B11 + B22, threshold, parallel)
    M2 = _strassen_recursive(A21 + A22, B11, threshold, parallel)
    M3 = _strassen_recursive(A11, B12 - B22, threshold, parallel)
    M4 = _strassen_recursive(A22, B21 - B11, threshold, parallel)
    M5 = _strassen_recursive(A11 + A12, B22, threshold, parallel)
    M6 = _strassen_recursive(A21 - A11, B11 + B12, threshold, parallel)
    M7 = _strassen_recursive(A12 - A22, B21 + B22, threshold, parallel)

    # Combine results to form C
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
  end

  # Combine quadrants
  C = [C11 C12; C21 C22]

  return C
end
