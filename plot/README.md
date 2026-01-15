# Benchmark Plotting

Generate comparison plots for Rust and Julia matrix multiplication benchmarks.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate Plots

Run the plotting script:

```bash
python compare_benchmarks.py
```

This will generate three plots in the `../results` directory:
- `classic_comparison.png` - Classic algorithm comparison
- `divide_conquer_comparison.png` - Divide-and-conquer algorithm comparison (with thread count)
- `all_algorithms_comparison.png` - Combined comparison including Julia's BLAS implementation

## Deactivate

When done:

```bash
deactivate
```
