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
    divide_conquer_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}; threshold::Int=32) where T

Divide and conquer matrix multiplication algorithm with parallelization.
Recursively divides matrices into quadrants until reaching the threshold,
then uses standard multiplication.

Arguments:
- A, B: Input matrices to multiply
- threshold: Minimum size to switch to standard multiplication (default: 32)
"""
function divide_conquer_matmul(A::AbstractMatrix{T}, B::AbstractMatrix{T}; threshold::Int=32) where T
  m, n = size(A)
  q, p = size(B)

  if n != q
    throw(DimensionMismatch("Matrix dimensions must agree: A is $(m)x$(n), B is $(q)x$(p)"))
  end

  if Threads.nthreads() <= 1
    throw(ArgumentError("divide_conquer_matmul requires multiple threads to run"))
  end

  # Pad matrices to the nearest power of 2 for optimal divide and conquer
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

  result = _divide_conquer_recursive(A, B, threshold)

  # Extract the original size result
  return result[1:m, 1:p]
end

"""
    _divide_conquer_recursive(A, B, threshold)

Internal recursive function for divide and conquer multiplication.
"""
function _divide_conquer_recursive(A::AbstractMatrix{T}, B::AbstractMatrix{T}, threshold::Int) where T
  m, n = size(A)
  q, p = size(B)

  # Base case: use our own iterative implementation for fair comparison
  if m <= threshold || n <= threshold || p <= threshold
    return iterative_matmul(A, B)
  end

  # Ensure matrices can be divided evenly (pad if necessary)
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

  # Compute the 8 products needed (C = A * B)
  # C11 = A11*B11 + A12*B21
  # C12 = A11*B12 + A12*B22
  # C21 = A21*B11 + A22*B21
  # C22 = A21*B12 + A22*B22

  # Parallel computation using threads
  # Spawn 8 tasks for the 8 required products
  t1 = Threads.@spawn _divide_conquer_recursive(A11, B11, threshold)
  t2 = Threads.@spawn _divide_conquer_recursive(A12, B21, threshold)
  t3 = Threads.@spawn _divide_conquer_recursive(A11, B12, threshold)
  t4 = Threads.@spawn _divide_conquer_recursive(A12, B22, threshold)
  t5 = Threads.@spawn _divide_conquer_recursive(A21, B11, threshold)
  t6 = Threads.@spawn _divide_conquer_recursive(A22, B21, threshold)
  t7 = Threads.@spawn _divide_conquer_recursive(A21, B12, threshold)
  t8 = Threads.@spawn _divide_conquer_recursive(A22, B22, threshold)

  # Fetch results and combine
  C11 = fetch(t1) + fetch(t2)
  C12 = fetch(t3) + fetch(t4)
  C21 = fetch(t5) + fetch(t6)
  C22 = fetch(t7) + fetch(t8)

  # Combine quadrants
  C = [C11 C12; C21 C22]

  return C
end
