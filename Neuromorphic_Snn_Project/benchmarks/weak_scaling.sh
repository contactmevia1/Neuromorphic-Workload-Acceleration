#!/bin/bash
#SBATCH --job-name=snn_weak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=results/weak_scaling_%j.out

TIMESTEPS=1000
DENSITY=0.05
SEED=42
NEURONS_PER_THREAD=1000

echo "=== Weak Scaling Benchmark ==="
echo "Work per thread: $NEURONS_PER_THREAD neurons"
echo ""

cd ..
make openmp
cd benchmarks
mkdir -p results

for threads in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$threads
    NEURONS=$((NEURONS_PER_THREAD * threads))
    echo "--- Threads: $threads, Neurons: $NEURONS ---"
    ../snn_openmp --neurons $NEURONS --timesteps $TIMESTEPS --density $DENSITY --seed $SEED
    echo ""
done

echo "=== Weak Scaling Complete ==="
