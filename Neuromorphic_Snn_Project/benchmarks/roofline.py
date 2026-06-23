#!/usr/bin/env python3
"""
roofline.py

Generates a Roofline Model for the SNN workload.

The Roofline Model shows whether your code is:
- MEMORY-BOUND (limited by memory bandwidth)
- COMPUTE-BOUND (limited by compute capacity)

For SNNs we expect MEMORY-BOUND because each synaptic operation reads a
weight and a target potential — very little compute per byte transferred.

Usage:
    # Laptop (uses laptop defaults, override with --bw / --compute):
    python benchmarks/roofline.py --platform laptop

    # HPC (MUST supply your actual hardware specs):
    python benchmarks/roofline.py --platform hpc --bw 51.2 --compute 800

    # Fully custom:
    python benchmarks/roofline.py --bw 40.0 --compute 200

Arguments:
    --platform   laptop | hpc   (sets default BW and compute for that platform)
    --bw         Memory bandwidth in GB/s  (overrides platform default)
    --compute    Peak compute in GFLOPS    (overrides platform default)
    --results    Path to results directory (default: results)

How to get your hardware specs:
    Laptop:
        RAM speed: Task Manager > Performance > Memory (e.g. 3200 MT/s)
        Bandwidth = MT/s * 8 bytes / 1000  (single channel)
                  = 3200 * 8 / 1000 = 25.6 GB/s
        CPU peak GFLOPS: look up CPU model on Intel/AMD ARK spec page.

    HPC:
        Run: lscpu   (to get CPU model)
        Run: clinfo  (for OpenCL/GPU device)
        Memory bandwidth: from CPU spec sheet, or run STREAM benchmark.
        For NVIDIA GPU: nvidia-smi --query-gpu=memory.bandwidth
        Common values:
          Intel Xeon E5-2680 v4 (14c): BW=68 GB/s, Compute=560 GFLOPS
          AMD EPYC 7763 (64c):         BW=204 GB/s, Compute=2700 GFLOPS
          NVIDIA V100:                  BW=900 GB/s, Compute=14000 GFLOPS
          NVIDIA A100:                  BW=2000 GB/s, Compute=77000 GFLOPS

Input:  results/varying_size.csv  (produced by run_all_experiments.py)
Output: results/fig8_roofline_model.png
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import csv
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 300

# ── Platform preset defaults ──────────────────────────────────────────────────
# These are ESTIMATES. Always override with --bw and --compute using your
# actual hardware specs for a publishable roofline model.
PLATFORM_PRESETS = {
    "laptop": {
        "bw":      25.6,   # DDR4-3200 single channel
        "compute": 120.0,  # Typical mid-range laptop CPU (single socket, no SIMD boost)
        "note":    "Laptop defaults: DDR4-3200 single-ch. Verify with Task Manager + CPU spec."
    },
    "hpc": {
        "bw":      51.2,   # DDR4-3200 dual channel (common HPC node baseline)
        "compute": 500.0,  # Conservative multi-core estimate; replace with your node's value
        "note":    "HPC defaults are conservative estimates. You MUST supply --bw and --compute "
                   "from your actual node specs (lscpu / nvidia-smi / CPU spec sheet)."
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Roofline Model for SNN workload")
    parser.add_argument("--platform", choices=["laptop", "hpc"], default="laptop",
                        help="Platform preset (sets default BW and compute)")
    parser.add_argument("--bw",      type=float, default=None,
                        help="Memory bandwidth in GB/s (overrides platform default)")
    parser.add_argument("--compute", type=float, default=None,
                        help="Peak compute in GFLOPS (overrides platform default)")
    parser.add_argument("--results", type=str,   default="results",
                        help="Path to results directory (default: results)")
    return parser.parse_args()

def read_csv(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found.")
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def estimate_synaptic_ops(r):
    """
    Fallback: estimate synaptic_ops from total_spikes and network density
    when the CSV field is zero or missing.

    In the sequential code, total_synaptic_ops counts actual weight-add
    operations during propagation. If the small network (N=1000) fires very
    rarely, this number can be near-zero, collapsing OI to ~0.
    We estimate: ops = total_spikes * avg_out_degree
                      = total_spikes * N * density
    This is the same formula used in snn_opencl.cpp.
    """
    n       = int(r.get('neurons',      1))
    density = float(r.get('density',   0.05))
    spikes  = int(r.get('total_spikes', 0))
    avg_out = n * density
    return int(spikes * avg_out)

def compute_oi_and_perf(r):
    """
    Returns (oi, perf_giga_sops, syn_ops, used_fallback).

    Operational Intensity (OI) = synaptic_ops_per_timestep / bytes_per_timestep

    Bytes per timestep breakdown:
      Read  V[]       : 4 bytes × N   (float, read during leak + propagate)
      Write V[]       : 4 bytes × N   (write during propagate + reset)
      Read  spikes[]  : 1 byte  × N   (uint8, read during propagate)
      Read  weights   : 4 bytes × (syn_ops / timesteps)  (active synapses only)
    Total ≈ 9N + 4 × (syn_ops / timesteps)

    This is a lower-bound estimate (ignores CSR row_ptr / col_idx reads and
    cache-miss penalty, so actual effective bandwidth demand is higher).
    """
    n         = int(r['neurons'])
    time_ms   = float(r.get('total_time_ms', 1))
    timesteps = int(r.get('timesteps', 500))

    syn_ops      = int(r.get('synaptic_ops', 0))
    used_fallback = False
    if syn_ops == 0:
        syn_ops       = estimate_synaptic_ops(r)
        used_fallback = True

    ops_per_ts   = syn_ops / timesteps
    bytes_per_ts = 9 * n + 4 * ops_per_ts

    oi   = ops_per_ts / bytes_per_ts if bytes_per_ts > 0 else 0
    time_s = time_ms / 1000.0
    perf = (syn_ops / time_s) / 1e9 if time_s > 0 else 0

    return oi, perf, syn_ops, used_fallback

def main():
    args = parse_args()

    preset = PLATFORM_PRESETS[args.platform]
    MEM_BW      = args.bw      if args.bw      is not None else preset["bw"]
    PEAK_FLOPS  = args.compute if args.compute is not None else preset["compute"]
    RESULTS_DIR = args.results

    print("=" * 60)
    print("Generating Roofline Model")
    print("=" * 60)
    print(f"Platform : {args.platform}")
    print(f"BW       : {MEM_BW} GB/s")
    print(f"Compute  : {PEAK_FLOPS} GFLOPS")
    print(f"NOTE     : {preset['note']}")
    print()

    csv_path = os.path.join(RESULTS_DIR, "varying_size.csv")
    data = read_csv(csv_path)
    if not data:
        print(f"No data at {csv_path}. Run: python benchmarks/run_all_experiments.py")
        return

    oi_values, perf_values, sizes = [], [], []

    for r in data:
        n = int(r['neurons'])
        oi, perf, syn_ops, fallback = compute_oi_and_perf(r)
        flag = " [estimated from spikes×fan-out]" if fallback else ""
        print(f"  {n:5d} neurons: OI={oi:.4f} ops/byte, "
              f"Perf={perf:.3f} Giga-SOPS, syn_ops={syn_ops}{flag}")
        oi_values.append(oi)
        perf_values.append(perf)
        sizes.append(n)

    if not oi_values:
        print("No valid data points. Exiting.")
        return

    # ── Build roofline plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    oi_min, oi_max = 0.001, 10.0
    oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), 500)

    mem_roof  = MEM_BW    * oi_range
    comp_roof = np.full_like(oi_range, PEAK_FLOPS)
    roofline  = np.minimum(mem_roof, comp_roof)

    ax.loglog(oi_range, mem_roof,  '--', color='#E94F37', linewidth=2,
              label=f'Memory Bandwidth Roof ({MEM_BW} GB/s)')
    ax.axhline(y=PEAK_FLOPS, color='#2E86AB', linewidth=2,
               label=f'Compute Peak ({PEAK_FLOPS} GFLOPS)')
    ax.loglog(oi_range, roofline, '-', color='gray', linewidth=1, alpha=0.4)

    # Data points
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(oi_values)))
    for i, (oi, perf, n) in enumerate(zip(oi_values, perf_values, sizes)):
        ax.plot(oi, perf, 'o', markersize=12, color=colors[i],
                markeredgecolor='black', markeredgewidth=1.5,
                label=f'{n} neurons', zorder=5)

    # Ridge point
    ridge_oi = PEAK_FLOPS / MEM_BW
    ax.axvline(x=ridge_oi, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    # Place ridge label above the data to avoid overlap
    ax.text(ridge_oi * 1.15, PEAK_FLOPS * 1.05,
            f'Ridge\nOI={ridge_oi:.2f}',
            fontsize=8, color='green', fontweight='bold', va='bottom')

    # Memory-bound annotation — placed safely below data points
    min_perf = min(perf_values)
    annot_y  = max(min_perf * 0.3, 0.005)
    annot_oi = min(oi_values) * 0.8
    ax.text(annot_oi, annot_y,
            'Memory-Bound\nRegion\n(SNNs here)',
            fontsize=9, color='#E94F37', fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Gap annotation — explains why points sit below the diagonal
    # (irregular memory access / cache misses reduce effective BW)
    max_perf = max(perf_values)
    max_oi   = max(oi_values)
    expected_perf = MEM_BW * max_oi  # what roofline predicts
    if max_perf < expected_perf * 0.5:
        ax.annotate(
            f'Below roofline: irregular\nmemory access reduces\neffective bandwidth',
            xy=(max_oi, max_perf),
            xytext=(max_oi * 0.3, max_perf * 4),
            fontsize=8, color='#555555',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
        )

    ax.set_xlabel('Operational Intensity (Synaptic Ops / Byte)', fontsize=12)
    ax.set_ylabel('Performance (Giga-SOPS)', fontsize=12)
    ax.set_title(f'Roofline Model: SNN Simulation ({args.platform.upper()})',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(oi_min, oi_max)
    y_top = max(PEAK_FLOPS * 2, max(perf_values) * 5)
    y_bot = min(perf_values) * 0.1
    ax.set_ylim(max(y_bot, 0.001), y_top)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fig8_roofline_model.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()

    # ── Interpretation ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"  Ridge point OI : {ridge_oi:.3f} ops/byte")
    print(f"  Your OI values : {[f'{x:.4f}' for x in oi_values]}")
    all_memory_bound = all(oi < ridge_oi for oi in oi_values)
    if all_memory_bound:
        print("  RESULT: ALL points are MEMORY-BOUND (OI < ridge point).")
        print("  This confirms the 'memory wall' thesis for SNN on von Neumann hardware.")
        print()
        # Check how far below the roofline the points are
        for oi, perf, n in zip(oi_values, perf_values, sizes):
            ceiling = MEM_BW * oi
            util = (perf / ceiling * 100) if ceiling > 0 else 0
            print(f"  {n:5d} neurons: achieving {util:.1f}% of memory bandwidth ceiling")
            if util < 20:
                print(f"           → Large gap below roofline indicates irregular")
                print(f"             memory access (random V[col_idx] writes in propagate).")
    else:
        print("  WARNING: Some points appear COMPUTE-BOUND. Check hardware specs.")

if __name__ == "__main__":
    main()
