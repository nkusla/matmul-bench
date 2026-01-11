# Test script to verify correctness of implementations
using Test
using LinearAlgebra

include("classic_matmul.jl")
include("divide_conquer_matmul.jl")

println("Running correctness tests...")
println("="^50)

# Test with small matrices
sizes_to_test = [4, 8, 16, 32, 64]

for n in sizes_to_test
  println("\nTesting size: $(n)x$(n)")

  # Create random test matrices
  A = rand(Float64, n, n)
  B = rand(Float64, n, n)

  # Compute reference result
  C_reference = A * B

  # Test classic multiplication
  C_classic = classic_matmul(A, B)
  error_classic = norm(C_reference - C_classic)
  @test error_classic < 1e-10
  println("  ✓ Classic: error = $(error_classic)")

  # Test divide-and-conquer
  C_dc_par = divide_conquer_matmul(A, B; threshold=4, parallel=true)
  error_dc_par = norm(C_reference - C_dc_par)
  @test error_dc_par < 1e-10
  println("  ✓ Divide-Conquer: error = $(error_dc_par)")
end

println("\n" * "="^50)
println("All tests passed! ✓")
