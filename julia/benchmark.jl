using BenchmarkTools
using Statistics
using Printf

"""
		benchmark_algorithm(algorithm_fn, name, A, B; kwargs...)

Benchmark a single algorithm with given matrices.
Returns the mean time in seconds.
"""
function benchmark_algorithm(algorithm_fn, name::String, A::Matrix{T}, B::Matrix{T}; kwargs...) where T
  println("  Benchmarking $name...")

  # Actual benchmark
  trial = @benchmark $algorithm_fn($A, $B; $(kwargs)...) samples = 10 evals = 1

  mean_time = mean(trial.times) / 1e6  # Convert to milliseconds
  mean_memory = mean(trial.memory) / 1e6  # Convert to megabytes

  return (mean_time, mean_memory)
end

"""
		BenchmarkResult

Structure to store benchmark results.
"""
struct BenchmarkResult
  size::Int
  algorithm::String
  time::Float64         # in milliseconds
  memory::Float64       # in megabytes
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

    # Benchmark iterative algorithm
    try
      time_iter, mem_iter = benchmark_algorithm(iterative_matmul, "Iterative", A, B)
      push!(results, BenchmarkResult(n, "Iterative", time_iter, mem_iter))
      @printf("    Time: %.2f ms, Memory: %.2f MB\n",
              time_iter, mem_iter)
    catch e
      println("    Error: $e")
    end

    # Benchmark divide-and-conquer (always parallel)
    try
      time_dc, mem_dc = benchmark_algorithm(divide_conquer_matmul, "Divide-Conquer",
        A, B; threshold=64)
      push!(results, BenchmarkResult(n, "Divide-Conquer", time_dc, mem_dc))
      @printf("    Time: %.2f ms, Memory: %.2f MB\n",
              time_dc, mem_dc)
    catch e
      println("    Error: $e")
    end

    # Benchmark Strassen (always parallel)
    try
      time_strassen, mem_strassen = benchmark_algorithm(strassen_matmul, "Strassen",
        A, B; threshold=64)
      push!(results, BenchmarkResult(n, "Strassen", time_strassen, mem_strassen))
      @printf("    Time: %.2f ms, Memory: %.2f MB\n",
              time_strassen, mem_strassen)
    catch e
      println("    Error: $e")
    end

    # Benchmark Julia's built-in (for comparison)
    try
      time_builtin, mem_builtin = benchmark_algorithm((A, B) -> A * B, "Julia Built-in", A, B)
      push!(results, BenchmarkResult(n, "Julia-Builtin", time_builtin, mem_builtin))
      @printf("    Time: %.2f ms, Memory: %.2f MB\n",
              time_builtin, mem_builtin)
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
  @printf("%-10s %-25s %12s %15s\n", "Size", "Algorithm", "Time (ms)", "Memory (MB)")
  println("-"^80)

  for result in results
    @printf("%-10d %-25s %12.2f %15.2f\n",
      result.size, result.algorithm, result.time, result.memory)
  end

  println("="^80)
end

"""
		save_results_csv(results::Vector{BenchmarkResult}, filename::String)

Save benchmark results to a CSV file.
"""
function save_results_csv(results::Vector{BenchmarkResult}, filename::String)
  open(filename, "w") do io
    println(io, "size,algorithm,time_ms,memory_mb")
    for result in results
      println(io, "$(result.size),$(result.algorithm),$(result.time),$(result.memory)")
    end
  end
  println("\nResults saved to: $filename")
end
