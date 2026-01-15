# Julia Matrix Multiplication Benchmark

This directory contains Julia implementations of matrix multiplication algorithms for performance comparison.

## Algorithms Implemented

1. **Iterative Matrix Multiplication** (`iterative_matmul.jl`)
   - Standard iterative O(n³) algorithm
   - Three nested loops implementation

2. **Divide and Conquer** (`divide_conquer_matmul.jl`)
   - Recursive divide and conquer approach
   - Splits matrices into quadrants
   - Supports both sequential and parallel execution
   - Uses `Threads.@spawn` for parallel task distribution across threads

## Files

- `iterative_matmul.jl` - Iterative implementation
- `divide_conquer_matmul.jl` - Divide and conquer with parallelization
- `benchmark.jl` - Benchmarking framework using BenchmarkTools
- `main.jl` - Main script to execute benchmarks

## Requirements

The project uses a `Project.toml` file to manage dependencies. Install all required packages:
```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Or manually:
```bash
julia -e 'using Pkg; Pkg.add(["BenchmarkTools"])'
```

## Running Benchmarks

Execute the benchmark suite with multiple threads:
```bash
cd julia
julia --project=. -t 8 -O3 main.jl
```

Or use all available cores:
```bash
julia --project=. -t auto -O3 main.jl
```

## Benchmark Output

The benchmark will:
- Test multiple matrix sizes (64x64 up to 1024x1024 by default)
- Measure execution time for each algorithm
- Compare against Julia's built-in optimized multiplication
- Save results to a CSV file with timestamp

## Customization

Edit `main.jl` to:
- Change matrix sizes: modify the `sizes` array
- Adjust number of threads: use `-t N` flag when running (e.g., `julia -t 8 main.jl`)
- Modify threshold for base case: change `threshold` parameter in divide_conquer_matmul

## Notes

- For parallel execution, the divide-and-conquer algorithm spawns tasks using Julia's `Threads.@spawn` macro
- Uses shared-memory threading for low overhead and efficient data access
- The threshold parameter determines when to switch from recursive division to direct multiplication
- Julia's built-in multiplication (using BLAS) is included as a reference baseline
- Results are saved with timestamps to avoid overwriting previous benchmark runs
- For best performance, run with `-t auto` or `-t N` where N is your CPU core count
