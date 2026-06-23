# setup.ps1
# One-click setup for Windows PowerShell
# Downloads ALL OpenCL headers and builds all three versions

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Neuromorphic SNN - Windows Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Download ALL OpenCL headers if missing
$clDir = "src/common/CL"
$baseUrl = "https://raw.githubusercontent.com/KhronosGroup/OpenCL-Headers/main/CL"

$headers = @(
    "cl_platform.h",
    "cl_version.h",
    "cl.h",
    "cl_ext.h",
    "cl_gl.h",
    "cl_gl_ext.h"
)

$missing = $false
foreach ($h in $headers) {
    if (-not (Test-Path "$clDir/$h")) {
        $missing = $true
        break
    }
}

if ($missing) {
    Write-Host "[1/4] Downloading OpenCL headers from Khronos..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $clDir | Out-Null

    $failed = $false
    foreach ($h in $headers) {
        $outFile = "$clDir/$h"
        $url = "$baseUrl/$h"
        try {
            Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing -ErrorAction Stop
            Write-Host "      Downloaded: $h" -ForegroundColor Green
        } catch {
            Write-Host "      Failed: $h" -ForegroundColor Red
            $failed = $true
        }
    }

    if ($failed) {
        Write-Host "      Some headers failed to download." -ForegroundColor Red
        Write-Host "      Please manually download from: https://github.com/KhronosGroup/OpenCL-Headers" -ForegroundColor Red
    } else {
        Write-Host "      All headers downloaded successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "[1/4] OpenCL headers already present." -ForegroundColor Green
}

# Step 2: Build Sequential
Write-Host "[2/4] Building Sequential..." -ForegroundColor Cyan
$proc = Start-Process g++ -ArgumentList "-O3 -std=c++17 -Isrc/common src/sequential/snn_sequential.cpp -o snn_sequential.exe" -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -ne 0) {
    Write-Host "      FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "      OK" -ForegroundColor Green

# Step 3: Build OpenMP
Write-Host "[3/4] Building OpenMP..." -ForegroundColor Cyan
$proc = Start-Process g++ -ArgumentList "-O3 -fopenmp -std=c++17 -Isrc/common src/openmp/snn_openmp.cpp -o snn_openmp.exe" -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -ne 0) {
    Write-Host "      FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "      OK" -ForegroundColor Green

# Step 4: Build OpenCL
Write-Host "[4/4] Building OpenCL..." -ForegroundColor Cyan
$proc = Start-Process g++ -ArgumentList "-O3 -std=c++17 -Isrc/common src/opencl/snn_opencl.cpp -o snn_opencl.exe -lOpenCL" -Wait -PassThru -NoNewWindow
if ($proc.ExitCode -ne 0) {
    Write-Host "      FAILED" -ForegroundColor Red
    Write-Host ""
    Write-Host "OpenCL build failed. Common causes:" -ForegroundColor Yellow
    Write-Host "  - OpenCL.dll not found (install GPU drivers)" -ForegroundColor Yellow
    Write-Host "  - g++ cannot find -lOpenCL (add MinGW/bin to PATH)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Sequential and OpenMP are ready. You can still run benchmarks without OpenCL." -ForegroundColor Cyan
} else {
    Write-Host "      OK" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test commands (remember to use .\ prefix in PowerShell):" -ForegroundColor White
Write-Host "  .\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1" -ForegroundColor Gray
Write-Host "  $env:OMP_NUM_THREADS = 4" -ForegroundColor Gray
Write-Host "  .\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1" -ForegroundColor Gray
Write-Host "  .\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1" -ForegroundColor Gray
Write-Host ""
