#!/usr/bin/env python3
"""Plot generation script for SNN benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = "results"

def plot_strong_scaling():
    data = []
    with open(f"{RESULTS_DIR}/strong_scaling.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    threads = [int(d['threads']) for d in data if d['version'] == 'openmp']
    speedups = [float(d['speedup']) for d in data if d['version'] == 'openmp']
    efficiencies = [float(d['efficiency']) for d in data if d['version'] == 'openmp']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(threads, speedups, 'o-', linewidth=2, markersize=8, label='Measured')
    ax1.plot(threads, threads, '--', color='red', alpha=0.7, label='Ideal Linear')
    ax1.set_xlabel('Number of Threads')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Strong Scaling: Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(threads, efficiencies, 's-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal 100%')
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Parallel Efficiency (%)')
    ax2.set_title('Strong Scaling: Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/speedup_strong.png", dpi=300)
    print("Saved speedup_strong.png")

def plot_execution_comparison():
    data = {}
    with open(f"{RESULTS_DIR}/gpu_comparison.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['version']] = row

    versions = ['sequential', 'openmp', 'opencl']
    labels = ['Sequential', 'OpenMP (16t)', 'OpenCL (GPU)']

    leak = [float(data[v]['leak_ms']) for v in versions]
    prop = [float(data[v]['propagate_ms']) for v in versions]
    spike = [float(data[v]['spike_ms']) for v in versions]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, leak, width, label='Leak', color='skyblue')
    ax.bar(x, prop, width, label='Propagate', color='salmon')
    ax.bar(x + width, spike, width, label='Spike Check', color='lightgreen')

    ax.set_ylabel('Time (ms)')
    ax.set_title('Execution Time Breakdown by Phase')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/execution_breakdown.png", dpi=300)
    print("Saved execution_breakdown.png")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_strong_scaling()
    plot_execution_comparison()
    print("\nAll plots generated!")

if __name__ == "__main__":
    main()
