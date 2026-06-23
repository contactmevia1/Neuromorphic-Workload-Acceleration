/**
 * kernels.cl
 * 
 * OpenCL device kernels for SNN simulation.
 * These functions run on the GPU (or CPU via OpenCL).
 * 
 * KERNELS:
 * 1. leak_kernel: Update membrane potentials (decay toward rest)
 * 2. propagate_kernel: Propagate spikes through synapses (with atomics)
 * 3. spike_kernel: Check thresholds and emit spikes
 */

// ============================================================
// ATOMIC FLOAT ADD (Portable Implementation)
// ============================================================
// OpenCL 1.x does not have native atomic_add for float.
// We implement it using atomic_cmpxchg (compare-and-swap) on int bits.
// 
// How it works:
// 1. Read the current float value as int bits
// 2. Compute new value = old + delta
// 3. Try to swap old with new using atomic_cmpxchg
// 4. If another thread changed the value in between, retry
// 
// This is SLOW because threads serialize when targeting the same neuron.
// Exactly the bottleneck we want to measure!

inline void atomic_add_float(volatile __global float* addr, float val) {
    // Read current value as int
    int old = as_int(*addr);
    int cmp;
    int new_val;

    do {
        cmp = old;
        // Compute new float value, then convert back to int bits
        new_val = as_int(as_float(cmp) + val);
        // Try to atomically swap: if *addr == cmp, write new_val
        // Returns the value that was in *addr before the operation
        old = atomic_cmpxchg((__global int*)addr, cmp, new_val);
        // If old == cmp, the swap succeeded. If not, retry with new old.
    } while (old != cmp);
}

// ============================================================
// KERNEL 1: LEAK
// ============================================================
// One work-item per neuron.
// Updates membrane potential: V = V * decay + V_rest * (1 - decay)
// 
// This kernel is perfectly parallel with no dependencies.
// Memory access is coalesced (work-item i reads V[i]).

__kernel void leak_kernel(
    __global float* V,           // Membrane potentials (read/write)
    int num_neurons,             // Total neurons
    float decay,                 // Decay factor
    float v_rest                 // Resting potential
) {
    // Each work-item handles one neuron
    int i = get_global_id(0);

    // Bounds check (important if global size > num_neurons)
    if (i < num_neurons) {
        V[i] = V[i] * decay + v_rest * (1.0f - decay);
    }
}

// ============================================================
// KERNEL 2: PROPAGATE (Scatter Approach with Atomics)
// ============================================================
// One work-item per SYNAPSE.
// If the presynaptic neuron spiked, add weight to postsynaptic neuron.
// 
// WHY ONE WORK-ITEM PER SYNAPSE?
// This maximizes parallelism. With 1 million synapses, we launch 
// 1 million work-items. GPUs have thousands of cores, so this keeps
// them busy.
// 
// THE PROBLEM:
// Multiple synapses may target the same neuron. This requires
// atomic_add_float, which serializes threads. This is the bottleneck.
// 
// MEMORY ACCESS PATTERN:
// - Read spikes[src]: coalesced if adjacent synapses have similar src
// - Read weights[syn]: coalesced (adjacent synapses, adjacent weights)
// - Write V[dst]: UNCOALESCED (random targets) and ATOMIC (slow)

__kernel void propagate_kernel(
    __global const int* spikes,      // Spike array (1=spiked, 0=not)
    __global const int* src_idx,       // Source neuron per synapse
    __global const int* col_idx,       // Target neuron per synapse (CSR)
    __global const float* weights,     // Synaptic weight per synapse
    __global float* V,                 // Membrane potentials (atomic updates)
    int total_synapses                 // Total number of synapses
) {
    int syn = get_global_id(0);

    if (syn < total_synapses) {
        // Check if presynaptic neuron spiked
        int src = src_idx[syn];
        if (spikes[src] != 0) {
            int dst = col_idx[syn];
            float w = weights[syn];

            // ATOMIC UPDATE: Major bottleneck!
            // When many synapses target the same neuron, threads serialize.
            atomic_add_float(&V[dst], w);
        }
    }
}

// ============================================================
// KERNEL 2b: PROPAGATE (Gather Approach - No Atomics)
// ============================================================
// Alternative kernel: one work-item per NEURON.
// Each work-item reads all its INCOMING connections and sums them.
// This uses CSC (transposed) format.
// 
// ADVANTAGE: No atomics needed! Perfectly parallel.
// DISADVANTAGE: Must check all incoming synapses even if presynaptic
//   neuron didn't spike. Less efficient when activity is sparse.
// 
// Use this if the scatter kernel has too much atomic contention.

__kernel void propagate_kernel_gather(
    __global const int* spikes,      // Spike array
    __global const int* col_ptr,       // CSC column pointers (start per target)
    __global const int* row_idx,       // CSC row indices (source per synapse)
    __global const float* weights,     // Weights
    __global float* V,                 // Potentials
    int num_neurons                    // Total neurons
) {
    int j = get_global_id(0);  // Target neuron (postsynaptic)

    if (j < num_neurons) {
        float sum = 0.0f;

        // Iterate over all incoming synapses to neuron j
        for (int idx = col_ptr[j]; idx < col_ptr[j + 1]; idx++) {
            int src = row_idx[idx];  // Presynaptic neuron
            if (spikes[src] != 0) {
                sum += weights[idx];
            }
        }

        V[j] += sum;
    }
}

// ============================================================
// KERNEL 3: SPIKE CHECK
// ============================================================
// One work-item per neuron.
// Checks if V > threshold. If so, emits spike and resets V.
// 
// Perfectly parallel. Memory access is coalesced.

__kernel void spike_kernel(
    __global float* V,           // Membrane potentials (read/write)
    __global int* spikes,        // Spike output array (write)
    int num_neurons,             // Total neurons
    float threshold,             // Firing threshold
    float v_reset                // Reset potential
) {
    int i = get_global_id(0);

    if (i < num_neurons) {
        if (V[i] > threshold) {
            spikes[i] = 1;       // Emit spike
            V[i] = v_reset;      // Reset potential
        } else {
            spikes[i] = 0;       // No spike
        }
    }
}

// ============================================================
// KERNEL 4: COUNT SPIKES (Reduction)
// ============================================================
// Count total spikes using parallel reduction.
// Useful for verification and metrics.

__kernel void count_spikes(
    __global const int* spikes,
    __global int* partial_sums,
    int num_neurons
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int wg_size = get_local_size(0);

    // Local memory for reduction within work-group
    __local int local_sum[256];  // Assumes wg_size <= 256

    // Load into local memory
    local_sum[lid] = (gid < num_neurons) ? spikes[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int stride = wg_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    if (lid == 0) {
        partial_sums[get_group_id(0)] = local_sum[0];
    }
}
