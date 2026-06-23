
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "../common/types.hpp"
#include "../common/sparse_matrix.hpp"
#include "../common/utils.hpp"

using namespace SNN;

class OpenMPSNN {
public:
    int num_neurons;
    int num_timesteps;
    float input_rate;
    int num_threads;

    std::vector<float> V;
    std::vector<uint8_t> spikes;
    std::vector<uint8_t> new_spikes;

    CSRMatrix<float> weights;
    std::mt19937 rng;

    long long total_synaptic_ops = 0;
    long long total_spikes = 0;

    double t_leak = 0.0;
    double t_propagate = 0.0;
    double t_spike = 0.0;
    double t_total = 0.0;

    OpenMPSNN(int neurons, int timesteps, float density,
              uint32_t seed, float rate, int threads)
        : num_neurons(neurons), num_timesteps(timesteps),
          input_rate(rate), num_threads(threads)
    {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        } else {
            num_threads = omp_get_max_threads();
        }

        rng.seed(seed);
        V.resize(num_neurons, V_REST);
        spikes.resize(num_neurons, 0);
        new_spikes.resize(num_neurons, 0);

        std::cout << "Generating sparse connectivity (density=" << density << ")...\n";
        weights = CSRMatrix<float>(num_neurons, num_neurons, density, rng, 0.01f, 0.15f);
        std::cout << "Generated " << weights.num_nonzeros << " synapses.\n";
        std::cout << "OpenMP threads: " << num_threads << "\n\n";
    }

    void simulate() {
        Timer timer_total;
        timer_total.start();

        float spike_prob = input_rate * (DT / 1000.0f);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        const int PAD = 64 / sizeof(float);  // 16 floats = one cache line
        int max_threads = omp_get_max_threads();
        std::vector<float> local_V(max_threads * (num_neurons + PAD), 0.0f);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int actual_threads = omp_get_num_threads();
            float* my_V = &local_V[tid * (num_neurons + PAD)];
            long long local_syn_ops = 0;
            long long local_spikes = 0;
            // FIX: initialize to 0.0 to suppress -Wmaybe-uninitialized warning.
            // local_t0 is always set inside the omp single nowait block before
            // being read in the next single nowait block, but the compiler
            // cannot prove this across the intervening omp for barrier.
            double local_t0 = 0.0;

            for (int step = 0; step < num_timesteps; step++) {

                // STEP 1: Input spikes — one thread, implicit barrier after
                #pragma omp single
                {
                    int num_inputs = num_neurons / 10;
                    for (int i = 0; i < num_inputs; i++) {
                        spikes[i] = (dist(rng) < spike_prob) ? 1 : 0;
                    }
                    local_t0 = omp_get_wtime();
                }
                // [BARRIER 1] implicit from omp single

                // STEP 2: LEAK
                #pragma omp for schedule(static)
                for (int i = 0; i < num_neurons; i++) {
                    V[i] = V[i] * DECAY + V_REST * (1.0f - DECAY);
                }
                // [BARRIER 2] implicit from omp for

                #pragma omp single nowait
                {
                    double t1 = omp_get_wtime();
                    t_leak += (t1 - local_t0) * 1000.0;
                    local_t0 = t1;
                }

                // STEP 3: PROPAGATE — local buffers, no atomics
                memset(my_V, 0, num_neurons * sizeof(float));

                #pragma omp for schedule(guided)
                for (int i = 0; i < num_neurons; i++) {
                    if (spikes[i]) {
                        for (int idx = weights.row_ptr[i];
                             idx < weights.row_ptr[i + 1]; idx++) {
                            my_V[weights.col_idx[idx]] += weights.values[idx];
                            local_syn_ops++;
                        }
                    }
                }
                // [BARRIER 3] implicit from omp for

                #pragma omp single nowait
                {
                    double t1 = omp_get_wtime();
                    t_propagate += (t1 - local_t0) * 1000.0;
                    local_t0 = t1;
                }

                // MERGE: each thread handles a disjoint chunk of neuron array
                #pragma omp for schedule(static)
                for (int i = 0; i < num_neurons; i++) {
                    for (int t = 0; t < actual_threads; t++) {
                        V[i] += local_V[t * (num_neurons + PAD) + i];
                    }
                }
                // [BARRIER 4] implicit from omp for

                // STEP 4: SPIKE CHECK
                #pragma omp for schedule(static)
                for (int i = 0; i < num_neurons; i++) {
                    if (V[i] > V_THRESHOLD) {
                        new_spikes[i] = 1;
                        V[i] = V_RESET;
                        local_spikes++;
                    } else {
                        new_spikes[i] = 0;
                    }
                }
                // [BARRIER 5] implicit from omp for

                #pragma omp single nowait
                {
                    double t1 = omp_get_wtime();
                    t_spike += (t1 - local_t0) * 1000.0;
                }

                // Swap — one thread, implicit barrier guarantees all threads
                // see the new spikes[] before next timestep begins
                #pragma omp single
                {
                    std::swap(spikes, new_spikes);
                }
                // [BARRIER 6] implicit from omp single
            }

            #pragma omp atomic
            total_synaptic_ops += local_syn_ops;
            #pragma omp atomic
            total_spikes += local_spikes;
        }

        t_total = timer_total.stop();
    }
};

int main(int argc, char** argv) {
    SimulationParams params = SimulationParams::parse(argc, argv);
    params.print();

    OpenMPSNN snn(params.num_neurons, params.num_timesteps,
                  params.density, params.seed,
                  params.input_rate, params.num_threads);

    std::cout << "Running OpenMP simulation...\n";
    snn.simulate();

    print_results("OpenMP",
                  snn.t_leak, snn.t_propagate, snn.t_spike, snn.t_total,
                  snn.total_synaptic_ops, snn.total_spikes);

    return 0;
}
