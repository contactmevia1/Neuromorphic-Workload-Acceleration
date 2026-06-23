#!/usr/bin/env python3
"""
verify_openmp.py

Quick diagnostic to check if OpenMP is working correctly.
Run this before the full benchmark suite.
"""

import subprocess
import os
import time

def run(cmd, env=None):
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout, result.returncode

print("=" * 60)
print("OpenMP Diagnostic")
print("=" * 60)
print()

# 1. Check if executable exists
if not os.path.exists("snn_openmp.exe") and not os.path.exists("snn_openmp"):
    print("ERROR: snn_openmp executable not found. Please compile first.")
    print("  g++ -O3 -fopenmp -std=c++17 -Isrc/common src/openmp/snn_openmp.cpp -o snn_openmp.exe")
    exit(1)

exe = "snn_openmp.exe" if os.path.exists("snn_openmp.exe") else "snn_openmp"

# 2. Check if OpenMP reports correct thread count
print("Test 1: Thread count verification")
print("-" * 40)
for t in [1, 2, 4]:
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(t)
    stdout, rc = run([exe, "--neurons", "1000", "--timesteps", "100", 
                      "--density", "0.1", "--seed", "42"], env=env)

    # Extract thread count from output
    for line in stdout.split('\n'):
        if 'Using' in line and 'threads' in line:
            print(f"  OMP_NUM_THREADS={t} -> {line.strip()}")
            break
    else:
        print(f"  OMP_NUM_THREADS={t} -> Could not verify (check output below)")
        print(f"    Output: {stdout[:200]}")

print()

# 3. Performance sanity check
print("Test 2: Performance sanity check")
print("-" * 40)
print("Running sequential baseline (1000 neurons, 100 timesteps)...")
stdout, rc = run(["snn_sequential.exe" if os.path.exists("snn_sequential.exe") else "snn_sequential",
                  "--neurons", "1000", "--timesteps", "100", "--density", "0.1", "--seed", "42"])
seq_time = None
for line in stdout.split('\n'):
    if 'Total time:' in line:
        seq_time = float(line.split(':')[1].strip().split()[0])
        print(f"  Sequential: {seq_time} ms")
        break

print("Running OpenMP with 1 thread (same problem)...")
env = os.environ.copy()
env['OMP_NUM_THREADS'] = '1'
stdout, rc = run([exe, "--neurons", "1000", "--timesteps", "100", 
                  "--density", "0.1", "--seed", "42"], env=env)
omp1_time = None
for line in stdout.split('\n'):
    if 'Total time:' in line:
        omp1_time = float(line.split(':')[1].strip().split()[0])
        print(f"  OpenMP (1t): {omp1_time} ms")
        break

if seq_time and omp1_time:
    ratio = omp1_time / seq_time
    print(f"  Ratio (OpenMP 1t / Sequential): {ratio:.2f}x")
    if ratio > 2.0:
        print("  WARNING: OpenMP with 1 thread is much slower than sequential!")
        print("  This indicates a compilation or code issue.")
        print("  Try: delete snn_openmp.exe and recompile with -fopenmp")
    elif ratio > 1.3:
        print("  NOTE: OpenMP has moderate overhead ({:.0f}%).".format((ratio-1)*100))
        print("  This is normal for small problems.")
    else:
        print("  PASS: OpenMP overhead is acceptable.")

print()

# 4. Multi-thread speedup check
print("Test 3: Multi-thread speedup check")
print("-" * 40)
if seq_time:
    for t in [2, 4]:
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(t)
        stdout, rc = run([exe, "--neurons", "1000", "--timesteps", "100", 
                          "--density", "0.1", "--seed", "42"], env=env)
        omp_time = None
        for line in stdout.split('\n'):
            if 'Total time:' in line:
                omp_time = float(line.split(':')[1].strip().split()[0])
                speedup = seq_time / omp_time
                print(f"  OpenMP ({t}t): {omp_time} ms, speedup={speedup:.2f}x")
                if speedup < 0.5:
                    print(f"    WARNING: {t} threads should be faster than 1 thread!")
                break

print()
print("=" * 60)
print("Diagnostic complete.")
print("=" * 60)
