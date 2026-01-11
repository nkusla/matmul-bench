#!/usr/bin/env julia

# Load implementations
include("classic_matmul.jl")
include("divide_conquer_matmul.jl")

# Include benchmark module
include("benchmark.jl")

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
    64,    # Small
    128,   # Medium-small
    256,   # Medium
    512,   # Medium-large
    1024,  # Large
    2048,  # Very large
    #4096   # Extra large
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
  arch = Sys.ARCH
  os = Sys.KERNEL
  nthreads = Threads.nthreads()
  filename = "../results/julia_benchmark_$(os)_$(arch)_$(nthreads)t_$(timestamp).csv"
  save_results_csv(results, filename)

  println("\nBenchmark completed!")
  @printf("Total elapsed time: %.2f seconds\n", elapsed_time)
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
