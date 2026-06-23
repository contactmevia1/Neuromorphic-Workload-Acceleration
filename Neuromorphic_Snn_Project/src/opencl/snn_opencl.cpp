
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl.h"

#include "../common/types.hpp"
#include "../common/sparse_matrix.hpp"
#include "../common/utils.hpp"

using namespace SNN;

#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error " << err << " at " << msg << std::endl; \
        std::exit(1); \
    }

std::string read_kernel_source(const std::string& filename) {
    std::vector<std::string> paths = {
        filename, "src/opencl/kernels.cl", "../src/opencl/kernels.cl",
        "../../src/opencl/kernels.cl", "kernels.cl"
    };
    for (const auto& path : paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream buf;
            buf << file.rdbuf();
            std::cout << "Loaded kernels from: " << path << std::endl;
            return buf.str();
        }
    }
    std::cerr << "Cannot find kernel file: " << filename << std::endl;
    std::exit(1);
}

class OpenCLSNN {
public:
    int num_neurons;
    int num_timesteps;
    float input_rate;
    size_t wg_size;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel leak_kernel = nullptr;
    cl_kernel propagate_kernel = nullptr;
    cl_kernel spike_kernel = nullptr;

    cl_mem d_V = nullptr;
    cl_mem d_spikes = nullptr;
    cl_mem d_row_ptr = nullptr;
    cl_mem d_col_idx = nullptr;
    cl_mem d_src_idx = nullptr;
    cl_mem d_weights = nullptr;

    std::vector<float> h_V;
    std::vector<int> h_spikes;      // Current timestep spikes (from GPU)
    std::vector<int> h_spikes_prev; // Previous timestep spikes (for propagate)
    CSRMatrix<float> weights;
    std::mt19937 rng;

    long long total_synaptic_ops = 0;
    long long total_spikes = 0;
    double t_leak = 0.0;
    double t_propagate = 0.0;
    double t_spike = 0.0;
    double t_total = 0.0;

    OpenCLSNN(int neurons, int timesteps, float density,
              uint32_t seed, float rate, int workgroup_size)
        : num_neurons(neurons), num_timesteps(timesteps),
          input_rate(rate), wg_size(static_cast<size_t>(workgroup_size))
    {
        rng.seed(seed);
        h_V.resize(num_neurons, V_REST);
        h_spikes.resize(num_neurons, 0);
        h_spikes_prev.resize(num_neurons, 0);

        std::cout << "Generating sparse connectivity (density=" << density << ")..." << std::endl;
        weights = CSRMatrix<float>(num_neurons, num_neurons, density, rng, 0.01f, 0.15f);
        std::cout << "Generated " << weights.num_nonzeros << " synapses." << std::endl;

        setup_opencl();
        create_buffers();
        build_kernels();
    }

    void setup_opencl() {
        cl_int err;
        cl_uint num_platforms;
        err = clGetPlatformIDs(1, &platform, &num_platforms);
        CHECK_CL(err, "clGetPlatformIDs");

        if (num_platforms == 0) {
            std::cerr << "No OpenCL platforms found!" << std::endl;
            std::exit(1);
        }

        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            std::cout << "GPU not found, trying CPU..." << std::endl;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
            if (err != CL_SUCCESS || num_devices == 0) {
                std::cerr << "No OpenCL devices found!" << std::endl;
                std::exit(1);
            }
        }

        char device_name[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        std::cout << "OpenCL Device: " << device_name << std::endl;

        size_t max_wg_size;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, NULL);
        std::cout << "Max work-group size: " << max_wg_size << std::endl;
        if (wg_size > max_wg_size) {
            std::cout << "Warning: Adjusting wg_size to " << max_wg_size << std::endl;
            wg_size = max_wg_size;
        }

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CHECK_CL(err, "clCreateContext");

        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_CL(err, "clCreateCommandQueue");
    }

    void create_buffers() {
        cl_int err;
        d_V = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             num_neurons * sizeof(float), h_V.data(), &err);
        CHECK_CL(err, "clCreateBuffer V");

        d_spikes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  num_neurons * sizeof(int), h_spikes.data(), &err);
        CHECK_CL(err, "clCreateBuffer spikes");

        d_row_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   (num_neurons + 1) * sizeof(int), weights.row_ptr.data(), &err);
        CHECK_CL(err, "clCreateBuffer row_ptr");

        d_col_idx = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   weights.num_nonzeros * sizeof(int), weights.col_idx.data(), &err);
        CHECK_CL(err, "clCreateBuffer col_idx");

        d_src_idx = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   weights.num_nonzeros * sizeof(int), weights.src_idx.data(), &err);
        CHECK_CL(err, "clCreateBuffer src_idx");

        d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   weights.num_nonzeros * sizeof(float), weights.values.data(), &err);
        CHECK_CL(err, "clCreateBuffer weights");
    }

    void build_kernels() {
        cl_int err;
        std::string source = read_kernel_source("src/opencl/kernels.cl");
        const char* source_cstr = source.c_str();
        size_t source_len = source.length();

        program = clCreateProgramWithSource(context, 1, &source_cstr, &source_len, &err);
        CHECK_CL(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
            std::cerr << "Build log:\n" << log.data() << std::endl;
            CHECK_CL(err, "clBuildProgram");
        }

        leak_kernel = clCreateKernel(program, "leak_kernel", &err);
        CHECK_CL(err, "clCreateKernel leak_kernel");

        propagate_kernel = clCreateKernel(program, "propagate_kernel", &err);
        CHECK_CL(err, "clCreateKernel propagate_kernel");

        spike_kernel = clCreateKernel(program, "spike_kernel", &err);
        CHECK_CL(err, "clCreateKernel spike_kernel");
    }

    void simulate() {
        auto start_total = std::chrono::high_resolution_clock::now();

        cl_int err;
        float spike_prob = input_rate * (DT / 1000.0f);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        total_synaptic_ops = 0;

        size_t global_leak = ((num_neurons + wg_size - 1) / wg_size) * wg_size;
        size_t global_prop = ((weights.num_nonzeros + wg_size - 1) / wg_size) * wg_size;
        size_t global_spike = global_leak;

        // Initialize h_spikes_prev to zero for first timestep
        std::fill(h_spikes_prev.begin(), h_spikes_prev.end(), 0);

        for (int t = 0; t < num_timesteps; t++) {
            if (t % 100 == 0 && t > 0) {
                std::cout << "Timestep " << t << "/" << num_timesteps << std::endl;
            }

            // --------------------------------------------------------
            // FIX: Preserve spikes from previous timestep, only overwrite inputs
            // h_spikes_prev already contains spikes from previous timestep
            // (set in previous iteration from h_spikes)
            // Now overwrite only the input portion with new Poisson spikes
            // --------------------------------------------------------
            int num_inputs = num_neurons / 10;
            for (int i = 0; i < num_inputs; i++) {
                h_spikes_prev[i] = (dist(rng) < spike_prob) ? 1 : 0;
            }
            // Copy merged spikes to device
            err = clEnqueueWriteBuffer(queue, d_spikes, CL_TRUE, 0,
                                       num_neurons * sizeof(int), h_spikes_prev.data(), 0, NULL, NULL);
            CHECK_CL(err, "clEnqueueWriteBuffer spikes");

            // LEAK KERNEL
            cl_event event_leak;
            clSetKernelArg(leak_kernel, 0, sizeof(cl_mem), &d_V);
            clSetKernelArg(leak_kernel, 1, sizeof(int), &num_neurons);
            float decay = DECAY;
            float v_rest = V_REST;
            clSetKernelArg(leak_kernel, 2, sizeof(float), &decay);
            clSetKernelArg(leak_kernel, 3, sizeof(float), &v_rest);

            err = clEnqueueNDRangeKernel(queue, leak_kernel, 1, NULL,
                                         &global_leak, &wg_size, 0, NULL, &event_leak);
            CHECK_CL(err, "leak_kernel");

            clWaitForEvents(1, &event_leak);
            cl_ulong start, end;
            clGetEventProfilingInfo(event_leak, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event_leak, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            t_leak += (end - start) * 1e-6;
            clReleaseEvent(event_leak);

            // PROPAGATE KERNEL
            cl_event event_prop;
            clSetKernelArg(propagate_kernel, 0, sizeof(cl_mem), &d_spikes);
            clSetKernelArg(propagate_kernel, 1, sizeof(cl_mem), &d_src_idx);
            clSetKernelArg(propagate_kernel, 2, sizeof(cl_mem), &d_col_idx);
            clSetKernelArg(propagate_kernel, 3, sizeof(cl_mem), &d_weights);
            clSetKernelArg(propagate_kernel, 4, sizeof(cl_mem), &d_V);
            int total_syn = weights.num_nonzeros;
            clSetKernelArg(propagate_kernel, 5, sizeof(int), &total_syn);

            err = clEnqueueNDRangeKernel(queue, propagate_kernel, 1, NULL,
                                         &global_prop, &wg_size, 0, NULL, &event_prop);
            CHECK_CL(err, "propagate_kernel");

            clWaitForEvents(1, &event_prop);
            clGetEventProfilingInfo(event_prop, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event_prop, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            t_propagate += (end - start) * 1e-6;
            clReleaseEvent(event_prop);

            // SPIKE KERNEL
            cl_event event_spike;
            clSetKernelArg(spike_kernel, 0, sizeof(cl_mem), &d_V);
            clSetKernelArg(spike_kernel, 1, sizeof(cl_mem), &d_spikes);
            clSetKernelArg(spike_kernel, 2, sizeof(int), &num_neurons);
            float threshold = V_THRESHOLD;
            float v_reset = V_RESET;
            clSetKernelArg(spike_kernel, 3, sizeof(float), &threshold);
            clSetKernelArg(spike_kernel, 4, sizeof(float), &v_reset);

            err = clEnqueueNDRangeKernel(queue, spike_kernel, 1, NULL,
                                         &global_spike, &wg_size, 0, NULL, &event_spike);
            CHECK_CL(err, "spike_kernel");

            clWaitForEvents(1, &event_spike);
            clGetEventProfilingInfo(event_spike, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event_spike, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            t_spike += (end - start) * 1e-6;
            clReleaseEvent(event_spike);

            // Read spikes back from device
            // These spikes will be used in the NEXT timestep's propagate kernel
            err = clEnqueueReadBuffer(queue, d_spikes, CL_TRUE, 0,
                                      num_neurons * sizeof(int), h_spikes.data(), 0, NULL, NULL);
            CHECK_CL(err, "clEnqueueReadBuffer spikes");

            // Copy to h_spikes_prev for next timestep
            std::copy(h_spikes.begin(), h_spikes.end(), h_spikes_prev.begin());

            // Count spikes for metrics
            for (int i = 0; i < num_neurons; i++) {
                if (h_spikes[i]) total_spikes++;
            }
        }

        float avg_out_degree = (float)weights.num_nonzeros / num_neurons;
        total_synaptic_ops = (long long)(total_spikes * avg_out_degree);

        auto end_total = std::chrono::high_resolution_clock::now();
        t_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    }

    ~OpenCLSNN() {
        if (d_V) clReleaseMemObject(d_V);
        if (d_spikes) clReleaseMemObject(d_spikes);
        if (d_row_ptr) clReleaseMemObject(d_row_ptr);
        if (d_col_idx) clReleaseMemObject(d_col_idx);
        if (d_src_idx) clReleaseMemObject(d_src_idx);
        if (d_weights) clReleaseMemObject(d_weights);
        if (leak_kernel) clReleaseKernel(leak_kernel);
        if (propagate_kernel) clReleaseKernel(propagate_kernel);
        if (spike_kernel) clReleaseKernel(spike_kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

int main(int argc, char** argv) {
    SimulationParams params = SimulationParams::parse(argc, argv);
    params.print();

    try {
        OpenCLSNN snn(params.num_neurons,
                      params.num_timesteps,
                      params.density,
                      params.seed,
                      params.input_rate,
                      params.wg_size);

        std::cout << "Running OpenCL simulation..." << std::endl;
        snn.simulate();

        print_results("OpenCL",
                      snn.t_leak, snn.t_propagate, snn.t_spike, snn.t_total,
                      snn.total_synaptic_ops, snn.total_spikes);
    } catch (...) {
        std::cerr << "OpenCL initialization failed." << std::endl;
        return 1;
    }

    return 0;
}
