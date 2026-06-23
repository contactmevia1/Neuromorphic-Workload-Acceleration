@echo off
REM Build script for Windows (PowerShell/CMD)
REM Usage: build.bat

echo ==========================================
echo Building Neuromorphic SNN Project
echo ==========================================

REM Sequential
echo [1/3] Building Sequential...
g++ -O3 -std=c++17 -Isrc/common src/sequential/snn_sequential.cpp -o snn_sequential.exe
if %errorlevel% neq 0 (
    echo FAILED: Sequential build failed
    exit /b 1
)
echo Sequential: OK

REM OpenMP
echo [2/3] Building OpenMP...
g++ -O3 -fopenmp -std=c++17 -Isrc/common src/openmp/snn_openmp.cpp -o snn_openmp.exe
if %errorlevel% neq 0 (
    echo FAILED: OpenMP build failed
    exit /b 1
)
echo OpenMP: OK

REM OpenCL
echo [3/3] Building OpenCL...
g++ -O3 -std=c++17 -Isrc/common src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL
if %errorlevel% neq 0 (
    echo.
    echo WARNING: OpenCL build failed. This is usually because:
    echo   1. CL/cl.h is not in your include path
    echo   2. OpenCL library is not found
    echo.
    echo To fix, try one of these:
    echo   A) Download Khronos OpenCL headers from:
    echo      https://github.com/KhronosGroup/OpenCL-Headers
    echo      Extract to a folder and compile with:
    echo      g++ -O3 -std=c++17 -Isrc/common -I/path/to/OpenCL-Headers src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL
    echo.
    echo   B) If you have Intel GPU, install Intel OpenCL SDK
    echo   C) If you have NVIDIA GPU, install CUDA Toolkit (includes OpenCL)
    echo   D) If you have AMD GPU, install AMD APP SDK or ROCm
    echo.
    echo Skipping OpenCL for now. Sequential and OpenMP are ready.
    exit /b 0
)
echo OpenCL: OK

echo.
echo ==========================================
echo All builds successful!
echo ==========================================
echo.
echo Quick test commands:
echo   .\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1
echo   set OMP_NUM_THREADS=4
echo   .\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1
echo   .\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1
