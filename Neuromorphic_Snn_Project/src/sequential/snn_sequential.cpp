
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

#include "../common/types.hpp"
#include "../common/sparse_matrix.hpp"
#include "../common/utils.hpp"

using namespace SNN;

class SequentialSNN {
public:
    int num_neurons;
    int num_timesteps;
    float input_rate;

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

    SequentialSNN(int neurons, int timesteps, float density,
                  uint32_t seed, float rate)
        : num_neurons(neurons), num_timesteps(timesteps), input_rate(rate)
    {
        rng.seed(seed);
        V.resize(num_neurons, V_REST);
        spikes.resize(num_neurons, 0);
        new_spikes.resize(num_neurons, 0);

        std::cout << "Generating sparse connectivity (density=" << density << ")...\n";
        weights = CSRMatrix<float>(num_neurons, num_neurons, density, rng, 0.01f, 0.15f);
        std::cout << "Generated " << weights.num_nonzeros << " synapses.\n\n";
    }

    void simulate() {
        Timer timer_total;
        timer_total.start();

        float spike_prob = input_rate * (DT / 1000.0f);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int t = 0; t < num_timesteps; t++) {
            if (t % 100 == 0 && t > 0) {
                std::cout << "Timestep " << t << "/" << num_timesteps << "\n";
            }

            // STEP 1: Generate Poisson input spikes (first 10% of neurons)
       
            int num_inputs = num_neurons / 10;
            for (int i = 0; i < num_inputs; i++) {
                spikes[i] = (dist(rng) < spike_prob) ? 1 : 0;
            }

            // STEP 2: LEAK — decay all membrane potentials
            Timer timer_leak;
            timer_leak.start();
            for (int i = 0; i < num_neurons; i++) {
                V[i] = V[i] * DECAY + V_REST * (1.0f - DECAY);
            }
            t_leak += timer_leak.stop();

            // STEP 3: PROPAGATE — route spikes through synapses
            Timer timer_propagate;
            timer_propagate.start();
            for (int i = 0; i < num_neurons; i++) {
                if (spikes[i]) {
                    for (int idx = weights.row_ptr[i];
                         idx < weights.row_ptr[i + 1]; idx++) {
                        V[weights.col_idx[idx]] += weights.values[idx];
                        total_synaptic_ops++;
                    }
                }
            }
            t_propagate += timer_propagate.stop();

            // STEP 4: SPIKE CHECK — threshold and reset
            Timer timer_spike;
            timer_spike.start();
            for (int i = 0; i < num_neurons; i++) {
                if (V[i] > V_THRESHOLD) {
                    new_spikes[i] = 1;
                    V[i] = V_RESET;
                    total_spikes++;
                } else {
                    new_spikes[i] = 0;
                }
            }
            t_spike += timer_spike.stop();

            std::swap(spikes, new_spikes);
        }

        t_total = timer_total.stop();
    }
};

int main(int argc, char** argv) {
    SimulationParams params = SimulationParams::parse(argc, argv);
    params.print();

    SequentialSNN snn(params.num_neurons, params.num_timesteps,
                      params.density, params.seed, params.input_rate);

    std::cout << "Running sequential simulation...\n";
    snn.simulate();

    print_results("Sequential",
                  snn.t_leak, snn.t_propagate, snn.t_spike, snn.t_total,
                  snn.total_synaptic_ops, snn.total_spikes);

    if (params.check) {
        std::cout << "Final potential [0]: " << snn.V[0] << "\n";
        std::cout << "Total spikes: " << snn.total_spikes << "\n";
    }

    return 0;
}
