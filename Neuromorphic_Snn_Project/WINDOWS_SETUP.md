# Windows Setup Guide

## Prerequisites

1. **MinGW-w64** installed with g++ and OpenMP support
2. **OpenCL** drivers (usually comes with your GPU drivers)
3. **OpenCL headers** (may need to download separately)

## Quick Start

### Option 1: Use build.bat (Recommended)
```cmd
cd neuromorphic-snn-project
build.bat
```

This will try to build all three versions. If OpenCL fails, it will give you instructions to fix it.

### Option 2: Manual Build

```cmd
REM Sequential
g++ -O3 -std=c++17 -Isrc/common src/sequential/snn_sequential.cpp -o snn_sequential.exe

REM OpenMP
g++ -O3 -fopenmp -std=c++17 -Isrc/common src/openmp/snn_openmp.cpp -o snn_openmp.exe

REM OpenCL (if headers are available)
g++ -O3 -std=c++17 -Isrc/common src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL
```

## Fixing OpenCL Build Error: "CL/cl.h: No such file or directory"

### Solution A: Download Khronos Headers (Easiest)

1. Go to https://github.com/KhronosGroup/OpenCL-Headers
2. Click "Code" -> "Download ZIP"
3. Extract the ZIP to a folder, e.g., `C:\OpenCL-Headers`
4. Compile with:
```cmd
g++ -O3 -std=c++17 -Isrc/common -IC:\OpenCL-Headers src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL
```

### Solution B: Use Your GPU Vendor's SDK

- **Intel GPU**: Download Intel OpenCL SDK from https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html
- **NVIDIA GPU**: Install CUDA Toolkit (includes OpenCL headers)
- **AMD GPU**: Install AMD APP SDK or ROCm

After installation, find the include path and add it with `-I` flag.

### Solution C: Copy Headers to Project (Quick & Dirty)

1. Create folder `src/common/CL/`
2. Download these files from Khronos and place them in `src/common/CL/`:
   - `cl.h`
   - `cl_platform.h`
   - `cl_ext.h`
3. Change the include in `snn_opencl.cpp` from `#include <CL/cl.h>` to `#include "CL/cl.h"`
4. Compile with: `g++ -O3 -std=c++17 -Isrc/common src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL`

## Running the Programs

**IMPORTANT**: In PowerShell, you must use `.\` before executables in the current directory:

```powershell
# WRONG (PowerShell security restriction)
snn_sequential.exe --neurons 1000 --timesteps 100

# CORRECT
.\snn_sequential.exe --neurons 1000 --timesteps 100
```

In CMD (Command Prompt), you can run without `.\`:
```cmd
snn_sequential.exe --neurons 1000 --timesteps 100
```

## Setting OpenMP Threads

In PowerShell:
```powershell
$env:OMP_NUM_THREADS = 4
.\snn_openmp.exe --neurons 1000 --timesteps 100
```

In CMD:
```cmd
set OMP_NUM_THREADS=4
snn_openmp.exe --neurons 1000 --timesteps 100
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `g++ not recognized` | Add MinGW `bin` folder to your PATH environment variable |
| `undefined reference to omp_get_thread_num` | Add `-fopenmp` to g++ command |
| `OpenCL.dll not found` | Install GPU drivers. The DLL should be in `C:\Windows\System32\OpenCL.dll` |
| `CL/cl.h not found` | See "Fixing OpenCL Build Error" section above |
| Results differ between versions | This is expected due to floating-point ordering in atomics. Total spike count should be identical. |

## Next Steps

1. Verify sequential works: `.\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1`
2. Verify OpenMP works: `.\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1`
3. Fix OpenCL compilation (if needed)
4. Run larger benchmarks on your HPC cluster
5. Generate plots with Python: `python benchmarks/plot_results.py`
