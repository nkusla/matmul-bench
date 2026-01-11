# Rust Matrix Multiplication Benchmark

This directory contains Rust implementations of matrix multiplication algorithms for performance comparison.

## Algorithms Implemented

1. **Classic Matrix Multiplication** (`classic_matmul.rs`)
   - Standard iterative O(n³) algorithm
   - Three nested loops implementation

2. **Divide and Conquer** (`divide_conquer_matmul.rs`)
   - Recursive divide and conquer approach
   - Splits matrices into quadrants
   - Supports both sequential and parallel execution
   - Uses Rayon for parallel task distribution across threads

## Files

- `src/classic_matmul.rs` - Classic iterative implementation
- `src/divide_conquer_matmul.rs` - Divide and conquer with parallelization
- `src/benchmark.rs` - Benchmarking framework
- `src/main.rs` - Main entry point to execute benchmarks
- `src/test.rs` - Correctness tests
- `Cargo.toml` - Project configuration and dependencies

## Requirements

Make sure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

Dependencies are managed through Cargo and will be automatically downloaded:
- `rayon` - Data parallelism library
- `rand` - Random number generation

## Building

Build the project in release mode for optimal performance:
```bash
cd rust
cargo build --release
```

## Running Benchmarks

Execute the benchmark suite:
```bash
cargo run --release
```

By default, Rayon will use all available CPU cores. To control the number of threads:
```bash
RAYON_NUM_THREADS=4 cargo run --release
```

Or use all available cores explicitly:
```bash
RAYON_NUM_THREADS=$(nproc) cargo run --release
```

## Running Tests

Run the correctness tests:
```bash
cargo run --release --bin test
```

## Benchmark Output

The benchmark will:
- Test multiple matrix sizes (64x64 up to 4096x4096 by default)
- Measure execution time and GFLOPS for each algorithm
- Save results to a CSV file with timestamp in the `../results/` directory

## Performance Metrics

- **Time**: Median execution time in milliseconds
- **GFLOPS**: Giga Floating Point Operations Per Second (higher is better)
- Calculated as: 2n³ operations / time / 10⁹

## Customization

Edit `src/main.rs` to:
- Change matrix sizes: modify the `sizes` vector
- Adjust number of threads: use `RAYON_NUM_THREADS` environment variable

Edit `src/divide_conquer_matmul.rs` to:
- Modify threshold for base case: change the `threshold` parameter (default: 64)

## Notes

- For parallel execution, the divide-and-conquer algorithm uses Rayon's parallel iterators
- Uses work-stealing for efficient thread utilization
- The threshold parameter determines when to switch from recursive division to direct multiplication
- Results are saved with timestamps to avoid overwriting previous benchmark runs
- For best performance, always compile and run in release mode (`--release` flag)
- Rayon automatically detects and uses the optimal number of threads for your system

## Comparison with Julia

This Rust implementation mirrors the Julia version to enable fair performance comparisons:
- Same algorithms (Classic and Divide-Conquer)
- Same benchmarking methodology
- Same output format (CSV with size, algorithm, time, GFLOPS)
- Both use parallel execution for divide-and-conquer
