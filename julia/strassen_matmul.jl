using Base.Threads

include("iterative_matmul.jl")

"""
    next_power_of_2(n::Int)

Find the next power of 2 greater than or equal to n.
"""
function next_power_of_2(n::Int)
  n <= 0 && return 1
  power = 1
  while power < n
    power *= 2
  end
  return power
end

"""
    strassen_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}; threshold::Int=32) where T

Strassen's matrix multiplication algorithm with optional parallelization.
Recursively divides matrices into quadrants and computes 7 products instead of 8,
reducing complexity from O(n³) to O(n^2.807).

Arguments:
- A, B: Input matrices to multiply
- threshold: Minimum size to switch to standard multiplication (default: 32)
"""
function strassen_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}; threshold::Int=32) where T
  m, n = size(A)
  q, p = size(B)

  if n != q
    throw(DimensionMismatch("Matrix dimensions must agree: A is $(m)x$(n), B is $(q)x$(p)"))
  end

  if Threads.nthreads() <= 1
    throw(ArgumentError("strassen_matmul requires multiple threads to run"))
  end

  # Pad matrices to the nearest power of 2 for optimal Strassen algorithm
  padded_m = next_power_of_2(m)
  padded_n = next_power_of_2(n)
  padded_p = next_power_of_2(p)

  # Pad if necessary
  if m != padded_m || n != padded_n
    A_padded = zeros(T, padded_m, padded_n)
    A_padded[1:m, 1:n] = A
    A = A_padded
  end

  if n != padded_n || p != padded_p
    B_padded = zeros(T, padded_n, padded_p)
    B_padded[1:q, 1:p] = B
    B = B_padded
  end

  result = _strassen_recursive(A, B, threshold)

  # Extract the original size result
  return result[1:m, 1:p]
end

"""
    _strassen_recursive(A, B, threshold)

Internal recursive function for Strassen multiplication.
"""
function _strassen_recursive(A::AbstractMatrix{T}, B::AbstractMatrix{T}, threshold::Int) where T
  m, n = size(A)
  q, p = size(B)

  # Base case: use iterative multiplication for small matrices
  if m <= threshold || n <= threshold || p <= threshold
    return iterative_matmul(A, B)
  end

  # Calculate split points (matrices are already padded to power of 2)
  m_half = m ÷ 2
  n_half = n ÷ 2
  p_half = p ÷ 2

  # Divide A into quadrants
  A11 = @view A[1:m_half, 1:n_half]
  A12 = @view A[1:m_half, n_half+1:n]
  A21 = @view A[m_half+1:m, 1:n_half]
  A22 = @view A[m_half+1:m, n_half+1:n]

  # Divide B into quadrants
  B11 = @view B[1:n_half, 1:p_half]
  B12 = @view B[1:n_half, p_half+1:p]
  B21 = @view B[n_half+1:n, 1:p_half]
  B22 = @view B[n_half+1:n, p_half+1:p]

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

  # Parallel computation using threads
  # Spawn 7 tasks for the 7 Strassen products
  t1 = Threads.@spawn _strassen_recursive(A11 + A22, B11 + B22, threshold)
  t2 = Threads.@spawn _strassen_recursive(A21 + A22, B11, threshold)
  t3 = Threads.@spawn _strassen_recursive(A11, B12 - B22, threshold)
  t4 = Threads.@spawn _strassen_recursive(A22, B21 - B11, threshold)
  t5 = Threads.@spawn _strassen_recursive(A11 + A12, B22, threshold)
  t6 = Threads.@spawn _strassen_recursive(A21 - A11, B11 + B12, threshold)
  t7 = Threads.@spawn _strassen_recursive(A12 - A22, B21 + B22, threshold)

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

  # Combine quadrants
  C = [C11 C12; C21 C22]

  return C
end
