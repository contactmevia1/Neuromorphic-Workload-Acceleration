# test.ps1
# Quick test of all three versions

Write-Host "=== Testing Sequential ===" -ForegroundColor Cyan
.\snn_sequential.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

Write-Host ""
Write-Host "=== Testing OpenMP (4 threads) ===" -ForegroundColor Cyan
$env:OMP_NUM_THREADS = 4
.\snn_openmp.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42

Write-Host ""
Write-Host "=== Testing OpenCL ===" -ForegroundColor Cyan
if (Test-Path .\snn_opencl.exe) {
    .\snn_opencl.exe --neurons 1000 --timesteps 100 --density 0.1 --seed 42
} else {
    Write-Host "OpenCL executable not found. Run setup.ps1 first." -ForegroundColor Yellow
}
