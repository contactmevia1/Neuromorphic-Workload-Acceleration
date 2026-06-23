@echo off
REM Quick test script for Windows

echo === Testing Sequential ===
.\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

echo.
echo === Testing OpenMP (4 threads) ===
set OMP_NUM_THREADS=4
.\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

echo.
echo === Testing OpenCL ===
.\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42
