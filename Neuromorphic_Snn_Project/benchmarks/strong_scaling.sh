#!/bin/bash
#SBATCH --job-name=snn_strong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=results/strong_scaling_%j.out

NEURONS=20000
TIMESTEPS=1000
DENSITY=0.05
SEED=42

echo "=== Strong Scaling Benchmark ==="
echo "Problem size: $NEURONS neurons, $TIMESTEPS timesteps, density=$DENSITY"
echo ""

cd ..
make openmp
cd benchmarks
mkdir -p results

for threads in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$threads
    echo "--- Threads: $threads ---"
    ../snn_openmp --neurons $NEURONS --timesteps $TIMESTEPS --density $DENSITY --seed $SEED
    echo ""
done

echo "=== Strong Scaling Complete ==="
