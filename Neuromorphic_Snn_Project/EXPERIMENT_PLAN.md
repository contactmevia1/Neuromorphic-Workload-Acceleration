# Standard Experiment Plan
# For: Neuromorphic SNN Parallel Simulation
# Run on: Windows Laptop (development) + HPC Cluster (production)

## OVERVIEW

This document gives you EXACT commands to run 
We run 3 categories of experiments:

| Category | What It Shows | Where to Run |
|----------|--------------|--------------|
| A. Correctness & Small Scale | Code works, parameters are reasonable | Laptop |
| B. Standard Benchmarks | Speedup, scaling, SOPS, GPU comparison | Laptop + HPC |
| C. Large-Scale Production | Strong/weak scaling, roofline, publication data | HPC only |

## DATASET: What We Simulate

We use **synthetic Poisson spike trains** — NOT MNIST.

Input generation: Each timestep, the first 10% of neurons receive Poisson spikes
with probability = (input_rate × dt) per timestep. Default rate = 20 Hz.

## NETWORK SIZES

| Name | Neurons | Timesteps | Density | Synapses (approx) | Purpose |
|------|---------|-----------|---------|-------------------|---------|
| Tiny | 1,000 | 100 | 0.1 | ~100,000 | Quick test, debug |
| Small | 5,000 | 500 | 0.05 | ~1,250,000 | Laptop benchmark |
| Medium | 10,000 | 1,000 | 0.05 | ~5,000,000 | Standard benchmark |
| Large | 20,000 | 1,000 | 0.05 | ~20,000,000 | HPC strong scaling |
| XLarge | 50,000 | 2,000 | 0.02 | ~50,000,000 | HPC production |

## EXPERIMENT A: Correctness & Quick Tests (Laptop)

Purpose: Verify all three versions work and produce similar results.

```powershell
# A1. Sequential (baseline)
.\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

# A2. OpenMP (4 threads)
$env:OMP_NUM_THREADS = 4
.\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

# A3. OpenCL (GPU)
.\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42
```

Expected: All three should produce similar total spike counts (may differ slightly
in final potentials due to floating-point ordering in atomics).

## EXPERIMENT B: Standard Benchmarks (Laptop + HPC)

### B1. Execution Time Breakdown (All Versions)

Purpose: Show which phase dominates (should be "Propagate").

```powershell
# Sequential
.\snn_sequential.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42

# OpenMP (use all cores)
$env:OMP_NUM_THREADS = $env:NUMBER_OF_PROCESSORS
.\snn_openmp.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42

# OpenCL
.\snn_opencl.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42
```

Record: Leak time, Propagate time, Spike time, Total time, SOPS for each.

### B2. Strong Scaling (OpenMP)

Purpose: Same problem, more threads. Measure speedup.

```powershell
# Fix problem size
$NEURONS = 10000
$TIMESTEPS = 1000
$DENSITY = 0.05
$SEED = 42

foreach ($t in @(1, 2, 4, 8)) {
    $env:OMP_NUM_THREADS = $t
    Write-Host "=== Threads: $t ==="
    .\snn_openmp.exe --neurons $NEURONS --timesteps $TIMESTEPS --density $DENSITY --seed $SEED
}
```

On HPC (add more threads):
```bash
for t in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$t
    echo "=== Threads: $t ==="
    ./snn_openmp --neurons 20000 --timesteps 1000 --density 0.05 --seed 42
done
```

### B3. Weak Scaling (OpenMP)

Purpose: Fixed work per thread. Time should stay constant.

```powershell
# 1000 neurons per thread
foreach ($t in @(1, 2, 4, 8)) {
    $env:OMP_NUM_THREADS = $t
    $neurons = 1000 * $t
    Write-Host "=== Threads: $t, Neurons: $neurons ==="
    .\snn_openmp.exe --neurons $neurons --timesteps 1000 --density 0.05 --seed 42
}
```

On HPC:
```bash
for t in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$t
    neurons=$((1000 * t))
    echo "=== Threads: $t, Neurons: $neurons ==="
    ./snn_openmp --neurons $neurons --timesteps 1000 --density 0.05 --seed 42
done
```

### B4. GPU vs CPU Comparison

Purpose: Direct comparison of all three implementations.

```powershell
# Same parameters, all three versions
$PARAMS = "--neurons 10000 --timesteps 1000 --density 0.05 --seed 42"

.\snn_sequential.exe $PARAMS
$env:OMP_NUM_THREADS = 8
.\snn_openmp.exe $PARAMS
.\snn_opencl.exe $PARAMS
```

### B5. Work-Group Size Sweep (OpenCL)

Purpose: Find optimal GPU work-group size.

```powershell
foreach ($wg in @(32, 64, 128, 256)) {
    Write-Host "=== Work-group: $wg ==="
    .\snn_opencl.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42 --wg-size $wg
}
```

## EXPERIMENT C: Large-Scale Production (HPC Only)

### C1. Strong Scaling (up to 32-64 threads)

```bash
# Create a job script: large_strong.sh
sbatch large_strong.sh
```

Content of large_strong.sh:
```bash
#!/bin/bash
#SBATCH --job-name=snn_large_strong
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=results/large_strong_%j.out

NEURONS=50000
TIMESTEPS=2000
DENSITY=0.02
SEED=42

make all
mkdir -p results

for t in 1 2 4 8 16 32 64; do
    export OMP_NUM_THREADS=$t
    echo "=== Threads: $t ==="
    ./snn_openmp --neurons $NEURONS --timesteps $TIMESTEPS --density $DENSITY --seed $SEED
done
```

### C2. Weak Scaling (up to 64 threads)

```bash
#SBATCH --cpus-per-task=64

NEURONS_PER_THREAD=2000
TIMESTEPS=2000
DENSITY=0.02

for t in 1 2 4 8 16 32 64; do
    export OMP_NUM_THREADS=$t
    neurons=$((NEURONS_PER_THREAD * t))
    echo "=== Threads: $t, Neurons: $neurons ==="
    ./snn_openmp --neurons $neurons --timesteps $TIMESTEPS --density $DENSITY --seed $SEED
done
```

### C3. OpenCL at Scale

```bash
# Run OpenCL with large network
./snn_opencl --neurons 50000 --timesteps 2000 --density 0.02 --seed 42 --wg-size 128
```

### C4. Roofline Model Data Collection

```bash
# Run all three versions with varying sizes to get memory-bound vs compute-bound data
for n in 5000 10000 20000 50000; do
    ./snn_sequential --neurons $n --timesteps 1000 --density 0.05 --seed 42
done
```

## AUTOMATED RUNNING

Instead of typing commands manually, use the provided scripts:

### On Windows (PowerShell):
```powershell
# Run all standard benchmarks
python benchmarks/run_all_experiments.py --platform laptop

# Generate all plots
python benchmarks/plot_all.py

# Generate roofline data
python benchmarks/roofline.py
```

### On HPC (Linux):
```bash
# Run all benchmarks
python3 benchmarks/run_all_experiments.py --platform hpc

# Generate plots
python3 benchmarks/plot_all.py

# Generate roofline
python3 benchmarks/roofline.py
```

## WHAT TO RECORD FOR YOUR PAPER

For each run, copy these values into a spreadsheet:

| Run | Version | Threads | Neurons | Timesteps | Density | Leak(ms) | Propagate(ms) | Spike(ms) | Total(ms) | Spikes | SOPS |

You need at minimum:
- 3 versions × 1 size = 3 rows (for comparison table)
- 5-6 thread counts × 1 size = 5-6 rows (for strong scaling)
- 5-6 thread counts × proportional sizes = 5-6 rows (for weak scaling)
- 4-5 work-group sizes = 4-5 rows (for OpenCL tuning)

Total: ~20-25 data rows for a complete paper.

## HARDWARE SPECS TO RECORD

For your paper's Experimental Setup section, record:

**Laptop:**
- CPU model: (run `wmic cpu get name` in CMD)
- CPU cores: (run `wmic cpu get NumberOfCores`)
- RAM: (run `wmic computersystem get TotalPhysicalMemory`)
- GPU model: (check Device Manager → Display Adapters)
- OS: Windows 10/11
- Compiler: g++ (MinGW) version (run `g++ --version`)

**HPC:**
- CPU model: (run `lscpu | grep "Model name"`)
- CPU cores: (run `nproc`)
- RAM: (run `free -h`)
- GPU model: (run `clinfo | grep "Device Name"` or `nvidia-smi`)
- OS: (run `cat /etc/os-release`)
- Compiler: (run `g++ --version`)
- OpenCL version: (run `clinfo | grep "Version"`)

## PAPER FIGURES YOU NEED

1. **Speedup Curve** (Strong Scaling): X=threads, Y=speedup
2. **Efficiency Curve** (Strong Scaling): X=threads, Y=efficiency%
3. **Weak Scaling**: X=threads, Y=time (should be flat-ish)
4. **Execution Breakdown**: Bar chart of Leak/Propagate/Spike for each version
5. **SOPS Comparison**: Bar chart of SOPS for Sequential/OpenMP/OpenCL
6. **Roofline Model**: X=operational intensity, Y=performance, with your points
7. **Work-Group Size Analysis**: X=wg_size, Y=time (OpenCL only)

All of these are generated by `benchmarks/plot_all.py`.

## TIMELINE

| Day | Task | Where |
|-----|------|-------|
| 1 | Run Experiments A & B on laptop | Laptop |
| 2 | Upload to HPC, run Experiment C | HPC |
| 3 | Download results, run plot_all.py | Laptop |
| 4 | Write paper (use generated plots) | Laptop |
| 5 | Create presentation | Laptop |

## NEXT STEP

Run the automated benchmark suite:
```powershell
python benchmarks/run_all_experiments.py --platform laptop
```

This will compile everything, run all benchmarks, and save results to `results/`.
Then run:
```powershell
python benchmarks/plot_all.py
```

This generates all figures 
