/**
 * utils.hpp
 * 
 * Utility functions shared across all implementations:
 * - High-resolution timer
 * - Command-line argument parsing
 * - Random number generation helpers
 * - Output formatting
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

// ============================================================
// HIGH-RESOLUTION TIMER
// ============================================================
// We use std::chrono::high_resolution_clock for precise timing.
// This measures wall-clock time (not CPU time), which is what we want
// for benchmarking parallel code.

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running = false;

public:
    // Start the timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }

    // Stop the timer and return elapsed time in milliseconds
    double stop() {
        end_time = std::chrono::high_resolution_clock::now();
        running = false;
        return elapsed_ms();
    }

    // Get elapsed time in milliseconds without stopping
    double elapsed_ms() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }

    // Get elapsed time in seconds
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }
};

// ============================================================
// COMMAND-LINE ARGUMENTS
// ============================================================
// Simple struct to hold all simulation parameters.
// We parse these from command-line arguments.

struct SimulationParams {
    int num_neurons = 1000;        // Number of neurons in network
    int num_timesteps = 100;       // Number of simulation timesteps
    float density = 0.1f;          // Connection probability (0.0-1.0)
    uint32_t seed = 42;            // Random seed for reproducibility
    float input_rate = 20.0f;      // Poisson input rate (Hz)
    int wg_size = 128;             // OpenCL work-group size (OpenCL only)
    bool verbose = false;          // Print per-timestep details
    bool check = false;            // Verify against reference
    int num_threads = 0;           // OpenMP threads (0 = use all available)

    // Parse command-line arguments
    // Format: --neurons 1000 --timesteps 100 --density 0.1 --seed 42
    static SimulationParams parse(int argc, char** argv) {
        SimulationParams params;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--neurons" && i + 1 < argc) {
                params.num_neurons = std::stoi(argv[++i]);
            } else if (arg == "--timesteps" && i + 1 < argc) {
                params.num_timesteps = std::stoi(argv[++i]);
            } else if (arg == "--density" && i + 1 < argc) {
                params.density = std::stof(argv[++i]);
            } else if (arg == "--seed" && i + 1 < argc) {
                params.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--input-rate" && i + 1 < argc) {
                params.input_rate = std::stof(argv[++i]);
            } else if (arg == "--wg-size" && i + 1 < argc) {
                params.wg_size = std::stoi(argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                params.num_threads = std::stoi(argv[++i]);
            } else if (arg == "--verbose") {
                params.verbose = true;
            } else if (arg == "--check") {
                params.check = true;
            } else if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(0);
            }
        }
        return params;
    }

    static void print_help() {
        std::cout << "Spiking Neural Network Simulator\n";
        std::cout << "Usage: ./snn_[version] [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --neurons N      Number of neurons (default: 1000)\n";
        std::cout << "  --timesteps T    Number of timesteps (default: 100)\n";
        std::cout << "  --density d      Connection density 0-1 (default: 0.1)\n";
        std::cout << "  --seed s         Random seed (default: 42)\n";
        std::cout << "  --input-rate r   Poisson input rate Hz (default: 20)\n";
        std::cout << "  --wg-size w      OpenCL work-group size (default: 128)\n";
        std::cout << "  --threads t      OpenMP thread count (default: all cores)\n";
        std::cout << "  --verbose        Print per-timestep details\n";
        std::cout << "  --check          Verify output correctness\n";
        std::cout << "  --help           Show this help\n";
    }

    // Print parameters for logging
    void print() const {
        std::cout << "=== Simulation Parameters ===\n";
        std::cout << "Neurons:     " << num_neurons << "\n";
        std::cout << "Timesteps:   " << num_timesteps << "\n";
        std::cout << "Density:     " << density << "\n";
        std::cout << "Seed:        " << seed << "\n";
        std::cout << "Input rate:  " << input_rate << " Hz\n";
        if (wg_size != 128) {
            std::cout << "Work-group:  " << wg_size << "\n";
        }
        if (num_threads > 0) {
            std::cout << "Threads:     " << num_threads << "\n";
        }
        std::cout << "\n";
    }
};

// ============================================================
// OUTPUT FORMATTING
// ============================================================

inline void print_results(const std::string& version,
                          double t_leak, double t_propagate, double t_spike, double t_total,
                          long long total_synaptic_ops, long long total_spikes) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "=== " << version << " Results ===\n";
    std::cout << "Leak time:       " << t_leak << " ms\n";
    std::cout << "Propagate time:  " << t_propagate << " ms\n";
    std::cout << "Spike time:      " << t_spike << " ms\n";
    std::cout << "Total time:      " << t_total << " ms\n";
    std::cout << "Total spikes:    " << total_spikes << "\n";
    std::cout << "Synaptic ops:    " << total_synaptic_ops << "\n";

    double sops = static_cast<double>(total_synaptic_ops) / (t_total / 1000.0);
    std::cout << "SOPS:            " << std::scientific << std::setprecision(3) << sops << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n";
}

inline void print_speedup(double t_seq, double t_par, int num_threads) {
    double speedup = t_seq / t_par;
    double efficiency = (speedup / num_threads) * 100.0;
    std::cout << "=== Speedup Analysis ===\n";
    std::cout << "Sequential time: " << t_seq << " ms\n";
    std::cout << "Parallel time:   " << t_par << " ms\n";
    std::cout << "Speedup:         " << speedup << "x\n";
    std::cout << "Efficiency:      " << efficiency << "%\n";
    std::cout << "\n";
}

#endif // UTILS_HPP
