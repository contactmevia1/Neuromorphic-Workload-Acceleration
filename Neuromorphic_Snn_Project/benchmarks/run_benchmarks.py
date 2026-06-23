#!/usr/bin/env python3
"""Automated Benchmark Suite for SNN Project"""

import subprocess
import csv
import os
import sys

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NEURONS = 10000
TIMESTEPS = 1000
DENSITY = 0.05
SEED = 42

def run_executable(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception: {e}")
        return None

def parse_output(output):
    metrics = {}
    if not output:
        return metrics
    for line in output.split('\n'):
        if 'Total time:' in line:
            metrics['total_time_ms'] = float(line.split(':')[1].strip().split()[0])
        elif 'SOPS:' in line:
            metrics['sops'] = line.split(':')[1].strip()
        elif 'Total spikes:' in line:
            metrics['total_spikes'] = int(line.split(':')[1].strip())
        elif 'Leak time:' in line:
            metrics['leak_ms'] = float(line.split(':')[1].strip().split()[0])
        elif 'Propagate time:' in line:
            metrics['propagate_ms'] = float(line.split(':')[1].strip().split()[0])
        elif 'Spike time:' in line:
            metrics['spike_ms'] = float(line.split(':')[1].strip().split()[0])
    return metrics

def benchmark_strong_scaling():
    print("\n" + "="*60)
    print("STRONG SCALING BENCHMARK")
    print("="*60)

    results = []
    threads_list = [1, 2, 4, 8, 16]

    print("\n--- Sequential Baseline ---")
    output = run_executable(["./snn_sequential", "--neurons", str(NEURONS),
                             "--timesteps", str(TIMESTEPS), "--density", str(DENSITY),
                             "--seed", str(SEED)])
    seq_metrics = parse_output(output)
    seq_time = seq_metrics.get('total_time_ms', 1.0)
    results.append({'threads': 1, 'version': 'sequential', 'time_ms': seq_time,
                    'speedup': 1.0, 'efficiency': 100.0, **seq_metrics})

    for threads in threads_list:
        print(f"\n--- OpenMP Threads: {threads} ---")
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        old_env = os.environ
        os.environ = env

        output = run_executable(["./snn_openmp", "--neurons", str(NEURONS),
                                 "--timesteps", str(TIMESTEPS), "--density", str(DENSITY),
                                 "--seed", str(SEED)])
        os.environ = old_env

        metrics = parse_output(output)
        time_ms = metrics.get('total_time_ms', 0)
        speedup = seq_time / time_ms if time_ms > 0 else 0
        efficiency = (speedup / threads) * 100.0 if threads > 0 else 0

        results.append({'threads': threads, 'version': 'openmp', 'time_ms': time_ms,
                        'speedup': speedup, 'efficiency': efficiency, **metrics})

    with open(f"{RESULTS_DIR}/strong_scaling.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['threads', 'version', 'time_ms', 'speedup',
                                                'efficiency', 'total_spikes', 'sops'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {RESULTS_DIR}/strong_scaling.csv")

def benchmark_gpu_comparison():
    print("\n" + "="*60)
    print("GPU COMPARISON BENCHMARK")
    print("="*60)

    results = []

    for version, cmd in [('sequential', './snn_sequential'),
                         ('openmp', './snn_openmp'),
                         ('opencl', './snn_opencl')]:
        print(f"\n--- {version.upper()} ---")
        if version == 'openmp':
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '16'
            old_env = os.environ
            os.environ = env

        output = run_executable([cmd, "--neurons", str(NEURONS),
                                 "--timesteps", str(TIMESTEPS), "--density", str(DENSITY),
                                 "--seed", str(SEED)])

        if version == 'openmp':
            os.environ = old_env

        metrics = parse_output(output)
        metrics['version'] = version
        results.append(metrics)

    with open(f"{RESULTS_DIR}/gpu_comparison.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['version', 'total_time_ms', 'leak_ms',
                                                'propagate_ms', 'spike_ms', 'total_spikes', 'sops'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {RESULTS_DIR}/gpu_comparison.csv")

def main():
    print("SNN Automated Benchmark Suite")
    print("Make sure you have compiled all versions first!")
    print("Run: make all")

    for exe in ['./snn_sequential', './snn_openmp', './snn_opencl']:
        if not os.path.exists(exe):
            print(f"Error: {exe} not found. Please run 'make all' first.")
            sys.exit(1)

    benchmark_strong_scaling()
    benchmark_gpu_comparison()

    print("\n" + "="*60)
    print("ALL BENCHMARKS COMPLETE")
    print(f"Results saved in {RESULTS_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
