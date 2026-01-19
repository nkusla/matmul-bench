# Test script to verify correctness of implementations
using Test
using LinearAlgebra

include("iterative_matmul.jl")
include("divide_conquer_matmul.jl")
include("strassen_matmul.jl")

println("Running correctness tests...")
println("="^50)

# Test with small matrices
sizes_to_test = [50, 64, 100, 128, 150]

for n in sizes_to_test
  println("\nTesting size: $(n)x$(n)")

  # Create random test matrices
  A = rand(Float64, n, n)
  B = rand(Float64, n, n)

  # Compute reference result
  C_reference = A * B

  # Test iterative multiplication
  C_iterative = iterative_matmul(A, B)
  error_iterative = norm(C_reference - C_iterative)
  @test error_iterative < 1e-10
  println("  ✓ Iterative: error = $(error_iterative)")

  # Test divide-and-conquer
  C_dc_par = divide_conquer_matmul(A, B; threshold=4)
  error_dc_par = norm(C_reference - C_dc_par)
  @test error_dc_par < 1e-10
  println("  ✓ Divide-Conquer: error = $(error_dc_par)")

  # Test Strassen
  C_strassen = strassen_matmul(A, B; threshold=4)
  error_strassen = norm(C_reference - C_strassen)
  @test error_strassen < 1e-10
  println("  ✓ Strassen: error = $(error_strassen)")
end

println("\n" * "="^50)
println("All tests passed! ✓")
