# Paper Data Collection Template
# Copy values from your benchmark outputs into this table

## Table 1: Execution Time Breakdown (for Methodology/Results section)

| Version | Threads | Neurons | Timesteps | Density | Leak (ms) | Propagate (ms) | Spike (ms) | Total (ms) | Spikes | SOPS |
|---------|---------|---------|-----------|---------|-----------|----------------|------------|------------|--------|------|
| Sequential | 1 | 10000 | 1000 | 0.05 | | | | | | |
| OpenMP | 4 | 10000 | 1000 | 0.05 | | | | | | |
| OpenMP | 8 | 10000 | 1000 | 0.05 | | | | | | |
| OpenCL | GPU | 10000 | 1000 | 0.05 | | | | | | |

## Table 2: Strong Scaling (for Results section)

| Threads | Time (ms) | Speedup | Efficiency (%) | Leak (ms) | Propagate (ms) | Spike (ms) |
|---------|-----------|---------|----------------|-----------|----------------|------------|
| 1 | | 1.00 | 100.0 | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |
| 32 | | | | | | |

## Table 3: Weak Scaling (for Results section)

| Threads | Neurons | Time (ms) | Weak Efficiency (%) |
|---------|---------|-----------|---------------------|
| 1 | 1000 | | 100.0 |
| 2 | 2000 | | |
| 4 | 4000 | | |
| 8 | 8000 | | |
| 16 | 16000 | | |
| 32 | 32000 | | |

## Table 4: Work-Group Size (for Results section, OpenCL only)

| Work-Group Size | Time (ms) | Leak (ms) | Propagate (ms) | Spike (ms) |
|-----------------|-----------|-----------|----------------|------------|
| 32 | | | | |
| 64 | | | | |
| 128 | | | | |
| 256 | | | | |

## Hardware Specs (for Experimental Setup section)

### Laptop
- CPU Model: ________________ (run: wmic cpu get name)
- CPU Cores: ________________ (run: wmic cpu get NumberOfCores)
- CPU Threads: ______________ (run: wmic cpu get NumberOfLogicalProcessors)
- RAM: ______________________ (run: wmic computersystem get TotalPhysicalMemory)
- GPU Model: ________________ (Device Manager → Display Adapters)
- OS: _______________________
- Compiler: _________________ (run: g++ --version)
- OpenCL Version: ___________ (run: clinfo | findstr Version)

### HPC Cluster
- CPU Model: ________________ (run: lscpu | grep "Model name")
- CPU Cores per node: _______ (run: nproc)
- Total RAM: _______________ (run: free -h)
- GPU Model: ________________ (run: clinfo | grep "Device Name" or nvidia-smi)
- OS: ______________________ (run: cat /etc/os-release)
- Compiler: _________________ (run: g++ --version)
- OpenCL Version: __________ (run: clinfo | grep "Version")
- Memory Bandwidth: ________ (from spec sheet or stream benchmark)

## Key Metrics to Report

1. **Maximum Speedup**: _____ x (from strong scaling)
2. **Peak Efficiency**: _____ % (at what thread count?)
3. **Best SOPS (Sequential)**: _____ 
4. **Best SOPS (OpenMP)**: _____
5. **Best SOPS (OpenCL)**: _____
6. **OpenCL Speedup vs Sequential**: _____ x
7. **OpenCL Speedup vs Best OpenMP**: _____ x
8. **Dominant Phase**: _____ (should be "Propagate")
9. **Propagate % of Total (Sequential)**: _____ %
10. **Propagate % of Total (OpenMP)**: _____ %
11. **Propagate % of Total (OpenCL)**: _____ %
