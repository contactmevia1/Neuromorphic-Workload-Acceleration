#!/usr/bin/env python3
"""

FIXES:
1. compile_linux(): compiles sequential+openmp via make, then compiles opencl
   separately with automatic libOpenCL.so path detection. A failing opencl
   link no longer blocks the other two binaries.

2. HPC neurons capped at [10000, 20000, 30000] — 50000 removed because
   50000^2 = 2.5B > INT_MAX, causing vector::reserve to throw length_error
   in the CSR sparse matrix generator.

3. Weak scaling: N = N_base * sqrt(threads) keeps per-thread synaptic work
   constant (synapses = N^2 * density, so linear N scaling doubles per-thread
   work every step — that is NOT weak scaling).

4. Weak scaling baseline explicitly sets OMP_NUM_THREADS=1.
"""

import subprocess, csv, os, sys, time, math
from datetime import datetime

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = {
    "laptop": {
        "neurons": [1000, 3000, 5000],
        "timesteps": 500,
        "density": 0.05,
        "seed": 42,
        "threads": [1, 2, 4, 8],
        "wg_sizes": [32, 64, 128, 256],
        "weak_neurons_base": 500,
        "description": "Laptop benchmarks (small networks, fast)"
    },
    "hpc": {
        "neurons": [10000, 20000, 30000],  # 50000 removed: int overflow in CSR
        "timesteps": 2000,
        "density": 0.02,
        "seed": 42,
        "threads": [1, 2, 4, 8, 16, 32, 64],
        "wg_sizes": [64, 128, 256, 512],
        "weak_neurons_base": 2000,
        "description": "HPC benchmarks (large networks)"
    }
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_executable(cmd, env=None, timeout=600):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=timeout)
        if result.returncode != 0:
            log(f"  ERROR (exit {result.returncode}): {result.stderr[:300]}")
            return None, False
        return result.stdout, True
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s"); return None, False
    except Exception as e:
        log(f"  EXCEPTION: {e}"); return None, False

def parse_output(output):
    metrics = {}
    if not output: return metrics
    for line in output.split('\n'):
        if 'Total time:' in line:
            try: metrics['total_time_ms'] = float(line.split(':')[1].strip().split()[0])
            except: pass
        elif 'SOPS:' in line:
            metrics['sops'] = line.split(':')[1].strip()
        elif 'Total spikes:' in line:
            try: metrics['total_spikes'] = int(line.split(':')[1].strip())
            except: pass
        elif 'Synaptic ops:' in line:
            try: metrics['synaptic_ops'] = int(line.split(':')[1].strip().replace(',',''))
            except: pass
        elif 'Leak time:' in line:
            try: metrics['leak_ms'] = float(line.split(':')[1].strip().split()[0])
            except: pass
        elif 'Propagate time:' in line:
            try: metrics['propagate_ms'] = float(line.split(':')[1].strip().split()[0])
            except: pass
        elif 'Spike time:' in line:
            try: metrics['spike_ms'] = float(line.split(':')[1].strip().split()[0])
            except: pass
    return metrics

def save_csv(filename, rows, fieldnames):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    log(f"Saved: {filepath}")

def exe(name):
    return f"./{name}.exe" if os.name == 'nt' else f"./{name}"

def find_opencl_lib():
    """Search common HPC/CUDA locations for libOpenCL.so."""
    candidates = [
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.3/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.0/targets/x86_64-linux/lib",
        "/usr/local/cuda-11.8/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "libOpenCL.so")) or \
           os.path.exists(os.path.join(path, "libOpenCL.so.1")):
            return path
    try:
        result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'libOpenCL' in line and '=>' in line:
                return os.path.dirname(line.split('=>')[1].strip())
    except Exception:
        pass
    return None

def compile_windows():
    log("Compiling Sequential...")
    run_executable(["g++", "-O3", "-std=c++17", "-Isrc/common",
                    "src/sequential/snn_sequential.cpp", "-o", "snn_sequential.exe"])
    log("Compiling OpenMP...")
    run_executable(["g++", "-O3", "-fopenmp", "-std=c++17", "-Isrc/common",
                    "src/openmp/snn_openmp.cpp", "-o", "snn_openmp.exe"])
    log("Compiling OpenCL...")
    for method in [
        ["g++","-O3","-std=c++17","-Isrc/common","src/opencl/snn_opencl.cpp",
         "-o","snn_opencl.exe","C:\\Windows\\System32\\OpenCL.dll"],
        ["g++","-O3","-std=c++17","-Isrc/common","src/opencl/snn_opencl.cpp",
         "-o","snn_opencl.exe","-lOpenCL"],
    ]:
        _, ok = run_executable(method)
        if ok and os.path.exists("snn_opencl.exe"):
            log("  OpenCL compiled!"); return
    log("  WARNING: OpenCL compilation failed.")

def compile_linux():
    """
    Compile sequential+openmp via make, then opencl separately with
    auto-detected OpenCL library path. This way opencl link failure does
    not prevent the other two binaries from being built.
    """
    log("Compiling Sequential and OpenMP...")
    _, ok = run_executable(["make", "sequential", "openmp"])
    if not ok:
        log("  make failed, trying direct g++...")
        run_executable(["g++", "-O3", "-std=c++17", "-Isrc/common",
                        "src/sequential/snn_sequential.cpp", "-o", "snn_sequential"])
        run_executable(["g++", "-O3", "-fopenmp", "-std=c++17", "-Isrc/common",
                        "src/openmp/snn_openmp.cpp", "-o", "snn_openmp"])

    log("Compiling OpenCL...")
    ocl_path = find_opencl_lib()
    if ocl_path:
        log(f"  Found OpenCL at: {ocl_path}")
        _, ok = run_executable([
            "g++", "-O3", "-std=c++17", "-Isrc/common",
            f"-L{ocl_path}", "src/opencl/snn_opencl.cpp", "-o", "snn_opencl", "-lOpenCL"
        ])
        if ok:
            log("  OpenCL compiled!")
        else:
            log("  WARNING: OpenCL compilation failed — GPU experiments will be skipped.")
    else:
        log("  WARNING: libOpenCL.so not found — GPU experiments will be skipped.")

# ============================================================
# EXPERIMENT A: Correctness
# ============================================================
def benchmark_correctness(config):
    log("=" * 60); log("EXPERIMENT A: Correctness Verification"); log("=" * 60)
    n, t, d, seed = 1000, 100, 0.1, 42
    results = []

    log("Sequential...")
    output, ok = run_executable([exe("snn_sequential"),
        "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)])
    if ok:
        m = parse_output(output)
        m.update({'version':'sequential','threads':1,'neurons':n,'timesteps':t,'density':d})
        results.append(m)
        log(f"  Seq: {m.get('total_time_ms','ERR')}ms, {m.get('total_spikes','ERR')} spikes")

    log("OpenMP (4 threads)...")
    env = os.environ.copy(); env['OMP_NUM_THREADS'] = '4'
    output, ok = run_executable([exe("snn_openmp"),
        "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)], env=env)
    if ok:
        m = parse_output(output)
        m.update({'version':'openmp','threads':4,'neurons':n,'timesteps':t,'density':d})
        results.append(m)
        log(f"  OMP: {m.get('total_time_ms','ERR')}ms, {m.get('total_spikes','ERR')} spikes")

    ocl = exe("snn_opencl")
    if os.path.exists(ocl):
        log("OpenCL...")
        output, ok = run_executable([ocl,
            "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)])
        if ok:
            m = parse_output(output)
            m.update({'version':'opencl','threads':'gpu','neurons':n,'timesteps':t,'density':d})
            results.append(m)
            log(f"  OCL: {m.get('total_time_ms','ERR')}ms, {m.get('total_spikes','ERR')} spikes")

    if results:
        save_csv("correctness.csv", results,
            ['version','threads','neurons','timesteps','density','total_time_ms',
             'leak_ms','propagate_ms','spike_ms','total_spikes','synaptic_ops','sops'])
        spikes = [r.get('total_spikes',-1) for r in results if 'total_spikes' in r]
        log(f"Spike counts: {spikes}")
        if len(spikes) > 1 and max(spikes)-min(spikes) <= max(spikes)*0.01:
            log("PASS: All versions agree within 1%.")
        elif len(set(spikes)) == 1:
            log(f"PASS: All versions produced {spikes[0]} spikes.")
        else:
            log("WARNING: Spike counts differ — check correctness.")

# ============================================================
# EXPERIMENT B2: Strong Scaling
# ============================================================
def benchmark_strong_scaling(config):
    log("=" * 60); log("EXPERIMENT B2: Strong Scaling"); log("=" * 60)
    n = 5000 if config.get("density", 0.05) > 0.03 else 10000
    t, d, seed, threads = config["timesteps"], config["density"], config["seed"], config["threads"]
    log(f"Problem: {n} neurons, {t} timesteps, density={d}")

    log("Sequential baseline...")
    output, ok = run_executable([exe("snn_sequential"),
        "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)])
    if not ok: log("  Sequential FAILED - aborting strong scaling"); return
    seq_time = parse_output(output).get('total_time_ms', 1.0)
    log(f"  Sequential: {seq_time} ms")

    results = []
    for tc in threads:
        log(f"OpenMP {tc} threads...")
        env = os.environ.copy(); env['OMP_NUM_THREADS'] = str(tc)
        output, ok = run_executable([exe("snn_openmp"),
            "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)], env=env)
        if not ok: log(f"  FAILED - skipping"); continue
        m = parse_output(output)
        time_ms = m.get('total_time_ms', 0)
        speedup = seq_time / time_ms if time_ms > 0 else 0
        eff = (speedup / tc) * 100.0
        results.append({'threads':tc,'neurons':n,'timesteps':t,'density':d,
            'time_ms':time_ms,'speedup':round(speedup,3),'efficiency':round(eff,2),
            'leak_ms':m.get('leak_ms',0),'propagate_ms':m.get('propagate_ms',0),
            'spike_ms':m.get('spike_ms',0),'total_spikes':m.get('total_spikes',0),
            'synaptic_ops':m.get('synaptic_ops',0),'sops':m.get('sops','')})
        log(f"  {tc}t: {time_ms:.1f}ms, speedup={speedup:.2f}x, eff={eff:.1f}%")

    if results:
        save_csv("strong_scaling.csv", results,
            ['threads','neurons','timesteps','density','time_ms','speedup','efficiency',
             'leak_ms','propagate_ms','spike_ms','total_spikes','synaptic_ops','sops'])

# ============================================================
# EXPERIMENT B3: Weak Scaling
# ============================================================
def benchmark_weak_scaling(config):
    log("=" * 60); log("EXPERIMENT B3: Weak Scaling"); log("=" * 60)
    t, d, seed = config["timesteps"], config["density"], config["seed"]
    threads, neurons_base = config["threads"], config["weak_neurons_base"]

    log(f"Weak scaling: N_base={neurons_base}, formula N=N_base*sqrt(threads)")
    log(f"  (keeps synaptic ops per thread constant; synapses scale as N^2)")

    log("Baseline (1 thread, N=N_base)...")
    env_base = os.environ.copy(); env_base['OMP_NUM_THREADS'] = '1'
    output, ok = run_executable([exe("snn_openmp"),
        "--neurons", str(neurons_base), "--timesteps", str(t),
        "--density", str(d), "--seed", str(seed)], env=env_base)
    if not ok: log("  Baseline FAILED - aborting weak scaling"); return
    base_time = parse_output(output).get('total_time_ms', 1.0)
    log(f"  Baseline: {base_time:.2f} ms ({neurons_base} neurons, 1 thread)")

    results = []
    for tc in threads:
        neurons = int(neurons_base * math.sqrt(tc))
        log(f"{tc} threads, {neurons} neurons (sqrt scaling)...")
        env = os.environ.copy(); env['OMP_NUM_THREADS'] = str(tc)
        output, ok = run_executable([exe("snn_openmp"),
            "--neurons", str(neurons), "--timesteps", str(t),
            "--density", str(d), "--seed", str(seed)], env=env)
        if not ok: log(f"  FAILED - skipping"); continue
        m = parse_output(output)
        time_ms = m.get('total_time_ms', 0)
        weak_eff = (base_time / time_ms) * 100.0 if time_ms > 0 else 0
        results.append({'threads':tc,'neurons':neurons,'timesteps':t,'density':d,
            'time_ms':time_ms,'weak_efficiency':round(weak_eff,2),
            'leak_ms':m.get('leak_ms',0),'propagate_ms':m.get('propagate_ms',0),
            'spike_ms':m.get('spike_ms',0),'total_spikes':m.get('total_spikes',0),
            'sops':m.get('sops','')})
        log(f"  {tc}t: {time_ms:.1f}ms, weak_eff={weak_eff:.1f}%")

    if results:
        save_csv("weak_scaling.csv", results,
            ['threads','neurons','timesteps','density','time_ms','weak_efficiency',
             'leak_ms','propagate_ms','spike_ms','total_spikes','sops'])

# ============================================================
# EXPERIMENT B4: Version Comparison
# ============================================================
def benchmark_comparison(config):
    log("=" * 60); log("EXPERIMENT B4: Version Comparison"); log("=" * 60)
    n = 5000 if config.get("density", 0.05) > 0.03 else 10000
    t, d, seed = config["timesteps"], config["density"], config["seed"]
    log(f"Problem: {n} neurons, {t} timesteps, density={d}")
    results = []

    log("Sequential...")
    output, ok = run_executable([exe("snn_sequential"),
        "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)])
    if not ok: log("  Sequential FAILED"); return
    m = parse_output(output); seq_time = m.get('total_time_ms', 1.0)
    m.update({'version':'sequential','threads':1,'neurons':n,'speedup_vs_seq':1.0})
    results.append(m); log(f"  Seq: {seq_time}ms")

    max_threads = 8 if 8 in config["threads"] else config["threads"][-1]
    log(f"OpenMP ({max_threads} threads)...")
    env = os.environ.copy(); env['OMP_NUM_THREADS'] = str(max_threads)
    output, ok = run_executable([exe("snn_openmp"),
        "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)], env=env)
    if ok:
        m = parse_output(output)
        m.update({'version':'openmp','threads':max_threads,'neurons':n,
                  'speedup_vs_seq':round(seq_time/m.get('total_time_ms',1),3)})
        results.append(m); log(f"  OMP: {m.get('total_time_ms','ERR')}ms")

    ocl = exe("snn_opencl")
    if os.path.exists(ocl):
        log("OpenCL...")
        output, ok = run_executable([ocl,
            "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)])
        if ok:
            m = parse_output(output)
            m.update({'version':'opencl','threads':'gpu','neurons':n,
                      'speedup_vs_seq':round(seq_time/m.get('total_time_ms',1),3)})
            results.append(m); log(f"  OCL: {m.get('total_time_ms','ERR')}ms")
    else:
        log("OpenCL not found, skipping.")

    if results:
        save_csv("comparison.csv", results,
            ['version','threads','neurons','timesteps','density','total_time_ms',
             'leak_ms','propagate_ms','spike_ms','total_spikes','synaptic_ops',
             'sops','speedup_vs_seq'])

# ============================================================
# EXPERIMENT B5: Work-Group Size Sweep (OpenCL)
# ============================================================
def benchmark_wg_sweep(config):
    log("=" * 60); log("EXPERIMENT B5: Work-Group Size Sweep (OpenCL)"); log("=" * 60)
    ocl = exe("snn_opencl")
    if not os.path.exists(ocl): log("OpenCL not available, skipping."); return
    n, t, d, seed = 3000, 300, config["density"], config["seed"]
    log(f"Problem: {n} neurons, {t} timesteps")
    results = []
    for wg in config["wg_sizes"]:
        log(f"Work-group {wg}...")
        output, ok = run_executable([ocl,
            "--neurons", str(n), "--timesteps", str(t), "--density", str(d),
            "--seed", str(seed), "--wg-size", str(wg)])
        if not ok: log(f"  FAILED - skipping"); continue
        m = parse_output(output); m.update({'wg_size':wg,'neurons':n,'timesteps':t})
        results.append(m); log(f"  wg={wg}: {m.get('total_time_ms','ERR')}ms")
    if results:
        save_csv("wg_sweep.csv", results,
            ['wg_size','neurons','timesteps','density','total_time_ms',
             'leak_ms','propagate_ms','spike_ms','total_spikes','synaptic_ops','sops'])

# ============================================================
# EXPERIMENT: Varying Sizes (Roofline)
# ============================================================
def benchmark_varying_size(config):
    log("=" * 60); log("EXPERIMENT: Varying Sizes (for Roofline)"); log("=" * 60)
    t, d, seed, sizes = config["timesteps"], config["density"], config["seed"], config["neurons"]
    results = []
    for n in sizes:
        log(f"Size: {n} neurons...")
        output, ok = run_executable([exe("snn_sequential"),
            "--neurons", str(n), "--timesteps", str(t), "--density", str(d), "--seed", str(seed)],
            timeout=300)
        if not ok: log(f"  FAILED - skipping {n} neurons"); continue
        m = parse_output(output); m.update({'version':'sequential','neurons':n,'timesteps':t})
        results.append(m); log(f"  {n} neurons: {m.get('total_time_ms','ERR')}ms")
    if results:
        save_csv("varying_size.csv", results,
            ['version','neurons','timesteps','density','total_time_ms',
             'leak_ms','propagate_ms','spike_ms','total_spikes','synaptic_ops','sops'])

# ============================================================
# MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all SNN benchmarks")
    parser.add_argument("--platform", choices=["laptop","hpc"], default="laptop")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    config = CONFIGS[args.platform]
    log(f"Platform: {args.platform}")
    log(f"Description: {config['description']}")
    log(f"Results will be saved to: {RESULTS_DIR}/")
    log("")

    if not args.skip_compile:
        if os.name == 'nt': compile_windows()
        else: compile_linux()

    start_time = time.time()
    benchmark_correctness(config)
    benchmark_strong_scaling(config)
    benchmark_weak_scaling(config)
    benchmark_comparison(config)
    benchmark_wg_sweep(config)
    benchmark_varying_size(config)

    elapsed = time.time() - start_time
    log("")
    log("=" * 60)
    log(f"ALL BENCHMARKS COMPLETE in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log(f"Results saved in: {RESULTS_DIR}/")
    log("=" * 60)
    log("")
    log("Next steps:")
    log("  1. python benchmarks/plot_all.py")
    log("  2. python benchmarks/roofline.py --platform hpc --bw 76.8 --compute 1050")

if __name__ == "__main__":
    main()
