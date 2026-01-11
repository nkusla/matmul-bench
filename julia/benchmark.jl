using BenchmarkTools
using Statistics
using Printf

"""
		benchmark_algorithm(algorithm_fn, name, A, B; kwargs...)

Benchmark a single algorithm with given matrices.
Returns the median time in seconds.
"""
function benchmark_algorithm(algorithm_fn, name::String, A::Matrix{T}, B::Matrix{T}; kwargs...) where T
  println("  Benchmarking $name...")

  # Actual benchmark
  trial = @benchmark $algorithm_fn($A, $B; $(kwargs)...) samples = 10 evals = 1

  median_time = median(trial.times) / 1e6  # Convert to milliseconds

  return median_time
end

"""
		BenchmarkResult

Structure to store benchmark results.
"""
struct BenchmarkResult
  size::Int
  algorithm::String
  time::Float64  # in milliseconds
  gflops::Float64  # Giga FLOPS
end

"""
		run_benchmarks(sizes::Vector{Int})

Run benchmarks for different matrix sizes and algorithms.
Returns a vector of BenchmarkResult objects.

Arguments:
- sizes: Vector of matrix sizes to test
"""
function run_benchmarks(sizes::Vector{Int})
  results = BenchmarkResult[]

  for n in sizes
    println("\nTesting matrix size: $(n)x$(n)")
    println("="^50)

    # Generate random matrices
    A = rand(Float64, n, n)
    B = rand(Float64, n, n)

    # Calculate FLOPS for this size (2n³ for matrix multiplication)
    flops = 2.0 * n^3

    # Benchmark classic algorithm
    try
      time_classic = benchmark_algorithm(classic_matmul, "Classic", A, B)
      gflops_classic = flops / (time_classic / 1e3) / 1e9  # Convert ms to s for GFLOPS
      push!(results, BenchmarkResult(n, "Classic", time_classic, gflops_classic))
      @printf("    Time: %.2f ms, Performance: %.2f GFLOPS\n", time_classic, gflops_classic)
    catch e
      println("    Error: $e")
    end

    # Benchmark divide-and-conquer (always parallel)
    try
      time_dc = benchmark_algorithm(divide_conquer_matmul, "Divide-Conquer",
        A, B; threshold=64, parallel=true)
      gflops_dc = flops / (time_dc / 1e3) / 1e9  # Convert ms to s for GFLOPS
      push!(results, BenchmarkResult(n, "Divide-Conquer", time_dc, gflops_dc))
      @printf("    Time: %.2f ms, Performance: %.2f GFLOPS\n", time_dc, gflops_dc)
    catch e
      println("    Error: $e")
    end

    # Benchmark Julia's built-in (for comparison)
    try
      time_builtin = benchmark_algorithm((A, B) -> A * B, "Julia Built-in", A, B)
      gflops_builtin = flops / (time_builtin / 1e3) / 1e9  # Convert ms to s for GFLOPS
      push!(results, BenchmarkResult(n, "Julia-Builtin", time_builtin, gflops_builtin))
      @printf("    Time: %.2f ms, Performance: %.2f GFLOPS\n", time_builtin, gflops_builtin)
    catch e
      println("    Error: $e")
    end
  end

  return results
end

"""
	print_results_table(results::Vector{BenchmarkResult})

Print benchmark results in a formatted table.
"""
function print_results_table(results::Vector{BenchmarkResult})
  println("\n" * "="^80)
  println("BENCHMARK RESULTS SUMMARY")
  println("="^80)
  @printf("%-10s %-25s %12s %12s\n", "Size", "Algorithm", "Time (ms)", "GFLOPS")
  println("-"^80)

  for result in results
    @printf("%-10d %-25s %12.2f %12.2f\n",
      result.size, result.algorithm, result.time, result.gflops)
  end

  println("="^80)
end

"""
		save_results_csv(results::Vector{BenchmarkResult}, filename::String)

Save benchmark results to a CSV file.
"""
function save_results_csv(results::Vector{BenchmarkResult}, filename::String)
  open(filename, "w") do io
    println(io, "size,algorithm,time_ms,gflops")
    for result in results
      println(io, "$(result.size),$(result.algorithm),$(result.time),$(result.gflops)")
    end
  end
  println("\nResults saved to: $filename")
end
