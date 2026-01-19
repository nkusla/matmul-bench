#!/usr/bin/env julia

# Load implementations
include("iterative_matmul.jl")
include("divide_conquer_matmul.jl")
include("strassen_matmul.jl")

# Include benchmark module
include("benchmark.jl")

using InteractiveUtils: @code_llvm

"""
Main script to run matrix multiplication benchmarks.
"""
function main()
  println("="^80)
  println("Matrix Multiplication Benchmark - Julia Implementation")
  println("="^80)
  println("Number of threads: $(Threads.nthreads())")
  println()

  # Define matrix sizes to test
  # Start small and scale up
  sizes = [
    64, 128, 256, 512, 1024, 2048
  ]

  println("Testing sizes: $sizes")
  println()

  # Run benchmarks
  start_time = time()
  results = run_benchmarks(sizes)
  elapsed_time = time() - start_time

  # Display results
  print_results_table(results)

  # Save results to CSV
  timestamp = floor(Int, time())
  arch = lowercase(string(Sys.ARCH))
  os = lowercase(string(Sys.KERNEL))
  nthreads = Threads.nthreads()
  filename = "../results/data/julia_benchmark_$(os)_$(arch)_$(nthreads)t_$(timestamp).csv"
  save_results_csv(results, filename)

  println("\nBenchmark completed!")
  @printf("Total elapsed time: %.2f seconds\n", elapsed_time)
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
