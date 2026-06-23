/**
 * types.hpp
 * 
 * Basic type definitions and constants for the SNN simulation.
 * This file is included by ALL implementations (sequential, OpenMP, OpenCL).
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstddef>
#include <cstdint>

// ============================================================
// SIMULATION PARAMETERS (LIF Neuron Model)
// ============================================================
// These are the biological parameters for our Leaky Integrate-and-Fire model.
// We use float (32-bit) for performance. double is more accurate but slower.

namespace SNN {

    // Time step for simulation (1 millisecond)
    constexpr float DT = 1.0f;

    // Membrane time constant (20 milliseconds)
    // This controls how fast the membrane potential leaks back to rest.
    // Larger tau_m = slower leak = neuron "remembers" inputs longer.
    constexpr float TAU_M = 20.0f;

    // Decay factor per timestep: exp(-DT / TAU_M)
    // Calculated at compile time for efficiency.
    // V_new = V_old * DECAY + V_REST * (1 - DECAY)
    constexpr float DECAY = 0.951229424500714f;  // exp(-0.05)

    // Resting potential (0 millivolts)
    // The "baseline" voltage when no input is received.
    constexpr float V_REST = 0.0f;

    // Firing threshold (1 millivolt)
    // When V crosses this, the neuron emits a spike and resets.
    constexpr float V_THRESHOLD = 1.0f;

    // Reset potential (0 millivolts)
    // After firing, V is set to this value.
    constexpr float V_RESET = 0.0f;

    // Default Poisson input rate (20 Hz)
    // Each input neuron fires with probability (rate * DT) per timestep.
    // 20 Hz * 1 ms = 0.02 probability per timestep.
    constexpr float DEFAULT_INPUT_RATE = 20.0f;

    // Default random seed for reproducibility
    constexpr uint32_t DEFAULT_SEED = 42;
}

#endif // TYPES_HPP
