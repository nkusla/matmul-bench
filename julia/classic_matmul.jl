"""
    classic_matmul(A::Matrix{T}, B::Matrix{T}) where T

Standard iterative matrix multiplication algorithm.
Time complexity: O(n³)
"""
function classic_matmul(A::Matrix{T}, B::Matrix{T}) where T
  m, n = size(A)
  n2, p = size(B)

  if n != n2
    throw(DimensionMismatch("Matrix dimensions must agree: A is $(m)x$(n), B is $(n2)x$(p)"))
  end

  C = zeros(T, m, p)

  for i in 1:m
    for j in 1:p
      for k in 1:n
        C[i, j] += A[i, k] * B[k, j]
      end
    end
  end

  return C
end
