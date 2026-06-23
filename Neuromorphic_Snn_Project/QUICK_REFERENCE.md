# QUICK REFERENCE CARD
# Keep this open while running experiments

## COMPILE (One-time)
```powershell
# Windows
.\setup.ps1
# Or manually:
g++ -O3 -std=c++17 -Isrc/common src/sequential/snn_sequential.cpp -o snn_sequential.exe
g++ -O3 -fopenmp -std=c++17 -Isrc/common src/openmp/snn_openmp.cpp -o snn_openmp.exe
g++ -O3 -std=c++17 -Isrc/common src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL
```

## RUN EVERYTHING AUTOMATICALLY
```powershell
# 1. Run all benchmarks (takes 5-30 minutes depending on platform)
python benchmarks/run_all_experiments.py --platform laptop

# 2. Generate all plots
python benchmarks/plot_all.py

# 3. Generate roofline model
python benchmarks/roofline.py
```

## MANUAL TESTS (Quick verification)
```powershell
# Sequential
.\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

# OpenMP (4 threads)
$env:OMP_NUM_THREADS = 4
.\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

# OpenCL
.\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42
```

## STANDARD BENCHMARKS (For paper data)
```powershell
# B1. Execution breakdown (all versions)
.\snn_sequential.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42
$env:OMP_NUM_THREADS = 8
.\snn_openmp.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42
.\snn_opencl.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42

# B2. Strong scaling
foreach ($t in @(1,2,4,8)) {
    $env:OMP_NUM_THREADS = $t
    Write-Host "Threads: $t"
    .\snn_openmp.exe --neurons 10000 --timesteps 1000 --density 0.05 --seed 42
}

# B3. Weak scaling
foreach ($t in @(1,2,4,8)) {
    $env:OMP_NUM_THREADS = $t
    $n = 1000 * $t
    Write-Host "Threads: $t, Neurons: $n"
    .\snn_openmp.exe --neurons $n --timesteps 1000 --density 0.05 --seed 42
}

# B4. Work-group sweep (OpenCL)
foreach ($wg in @(32,64,128,256)) {
    Write-Host "WG: $wg"
    .\snn_opencl.exe --neurons 5000 --timesteps 500 --density 0.05 --seed 42 --wg-size $wg
}
```

## HPC COMMANDS (SSH into HPC)
```bash
# Upload project
scp -r neuromorphic-snn-project username@hpc.university.edu:~/projects/

# SSH in
ssh username@hpc.university.edu
cd ~/projects/neuromorphic-snn-project

# Compile
make all

# Run benchmarks
python3 benchmarks/run_all_experiments.py --platform hpc

# Or submit as batch job
cat > run_benchmarks.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=snn_benchmarks
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=results/benchmark_%j.out

cd $SLURM_SUBMIT_DIR
make all
python3 benchmarks/run_all_experiments.py --platform hpc --skip-compile
python3 benchmarks/plot_all.py
python3 benchmarks/roofline.py
EOF

sbatch run_benchmarks.sh

# Download results back to laptop
scp -r username@hpc.university.edu:~/projects/neuromorphic-snn-project/results ./hpc_results/
```

## PLOT FILES GENERATED
After running plot_all.py and roofline.py, you get:
- results/fig1_strong_scaling_speedup.png
- results/fig2_strong_scaling_efficiency.png
- results/fig3_weak_scaling.png
- results/fig4_execution_breakdown.png
- results/fig5_sops_comparison.png
- results/fig6_speedup_comparison.png
- results/fig7_wg_size_sweep.png
- results/fig8_roofline_model.png

## CSV DATA FILES
- results/correctness.csv
- results/strong_scaling.csv
- results/weak_scaling.csv
- results/comparison.csv
- results/wg_sweep.csv
- results/varying_size.csv
