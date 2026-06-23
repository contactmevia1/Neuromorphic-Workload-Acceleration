# Neuromorphic SNN Parallel Simulation Project

## Complete pipeline for: Spiking Neural Network (SNN) Acceleration using OpenMP & OpenCL

**Author:** Abiye Girma 
**Course:** Parallel Computing / Neuromorphic Architecture  


---

## What This Project Does

We are simulating a **brain-inspired neural network** on a traditional computer to prove that traditional computers struggle with brain-like workloads.

### The Big Picture
- Your brain has ~86 billion neurons connected by trillions of synapses
- These neurons don't compute continuously; they send **spikes** (electrical impulses) only when excited
- This is called **event-driven** computation — most neurons are idle most of the time
- Traditional computers (CPUs/GPUs) are terrible at this because they expect regular, predictable work
- **Neuromorphic chips** (like Intel Loihi, IBM TrueNorth) are built specifically for this
- In this project, we simulate an SNN on CPU and GPU to measure exactly *why* traditional hardware fails

---

## Project Structure

```
neuromorphic-snn-project/
├── README.md                    # This file
├── Makefile                     # Build everything (Linux/HPC)
├── CMakeLists.txt               # Alternative build (cross-platform)
├── docs/
│   ├── EXPLANATIONS.md          # Detailed concept explanations
│   ├── PIPELINE.md              # Step-by-step execution roadmap
│   └── BENCHMARK_GUIDE.md     # How to run each benchmark
├── src/
│   ├── common/                  # Shared code (headers)
│   ├── sequential/              # Baseline single-threaded CPU
│   ├── openmp/                  # Multi-threaded CPU
│   └── opencl/                  # GPU accelerated
├── benchmarks/                  # Scripts for strong/weak scaling
├── data/                        # Dataset generation
├── paper/                       # IEEE paper template
└── presentation/                # Presentation outline
```

---

## Quick Start (What to Run)

### On Windows (VS Code + Terminal)
```bash
# 1. Build sequential baseline
g++ -O3 -std=c++17 src/sequential/snn_sequential.cpp -o snn_sequential.exe

# 2. Run it
snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1

# 3. Build OpenMP version
g++ -O3 -fopenmp -std=c++17 src/openmp/snn_openmp.cpp -o snn_openmp.exe

# 4. Run OpenMP version (4 threads)
set OMP_NUM_THREADS=4
snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1

# 5. Build OpenCL version (Windows)
g++ -O3 -std=c++17 src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL

# 6. Run OpenCL version
snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1
```

### On HPC (Linux/SSH)
```bash
# Connect to HPC
ssh your_username@hpc.university.edu

# Clone/navigate to project
cd neuromorphic-snn-project

# Build everything
make all

# Run sequential
./snn_sequential --neurons 10000 --timesteps 1000 --density 0.05

# Run OpenMP
export OMP_NUM_THREADS=16
./snn_openmp --neurons 10000 --timesteps 1000 --density 0.05

# Run OpenCL
./snn_opencl --neurons 10000 --timesteps 1000 --density 0.05

# Run full benchmark suite
python3 benchmarks/run_benchmarks.py
```

---

## The Four Phases (Mapped to Code)

| Phase | Folder | What It Is | Key Learning |
|-------|--------|-----------|--------------|
| **1. Baseline** | `src/sequential/` | Single-threaded C++ | Understand LIF model, sparse matrices, event-driven simulation |
| **2. CPU Parallel** | `src/openmp/` | Multi-threaded with OpenMP | Load balancing, atomic operations, false sharing |
| **3. GPU Acceleration** | `src/opencl/` | OpenCL kernels | Memory coalescing, work-groups, atomics on GPU |
| **4. Benchmarking** | `benchmarks/` | Python scripts | Strong/weak scaling, speedup, efficiency, SOPS |

---

## What Each Benchmark Measures

1. **Execution Time**: Wall-clock time for the simulation
2. **Speedup**: How many times faster is parallel vs sequential? (Speedup = T_seq / T_par)
3. **Strong Scaling**: Same problem, more threads. Ideal: linear speedup. Reality: diminishing returns.
4. **Weak Scaling**: Problem grows with threads. Ideal: constant time. Reality: communication overhead.
5. **Parallel Efficiency**: Speedup ÷ Number of threads. 100% = perfect.
6. **SOPS**: Synaptic Operations Per Second. The "FLOPS" of neuromorphic computing.

---

## Prerequisites

### Windows Laptop
- [x] VS Code installed
- [x] g++ installed (MinGW-w64 or MSYS2)
- [x] OpenMP support (comes with g++)
- [x] OpenCL installed (GPU vendor SDK: Intel/AMD/NVIDIA)

### HPC Cluster
- SSH access configured
- SLURM job scheduler (most common)
- `module load gcc` or similar
- OpenCL libraries available (check with `locate libOpenCL.so`)

---

## How to Check Your OpenCL Installation

```bash
# Linux/HPC
clinfo | head -n 30

# If not installed, ask your HPC admin:
# "Please install ocl-icd-opencl-dev and clinfo"
```

On Windows, your GPU driver usually includes OpenCL. Check Device Manager → Display adapters to see your GPU brand, then install the corresponding SDK if needed.

---

## Next Steps

1. Read `docs/EXPLANATIONS.md` to understand every concept
2. Read `docs/PIPELINE.md` to see the step-by-step roadmap
3. Read `docs/BENCHMARK_GUIDE.md` to understand each metric
4. Build and run the sequential version first
5. Then move to OpenMP, then OpenCL
6. Run benchmarks and generate plots
7. Write the paper using the template in `paper/`

---

## Help & Troubleshooting

| Problem | Solution |
|---------|----------|
| `g++ not recognized` | Add MinGW/bin to your PATH environment variable |
| `OpenCL library not found` | Install GPU vendor SDK or `ocl-icd-opencl-dev` on Linux |
| `undefined reference to omp_get_thread_num` | Add `-fopenmp` to g++ command |
| HPC job fails | Check `sbatch` script syntax with your HPC documentation |
| Segfault on large networks | Reduce `--neurons` or increase stack size: `ulimit -s unlimited` |

---




