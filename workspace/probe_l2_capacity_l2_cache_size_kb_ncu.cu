#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <numeric>

#define WARMUP_ITERATIONS 5
#define MEASUREMENT_ITERATIONS 10
#define CHASE_STEPS 5000

__global__ void initializeNodes(int* nodes, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        nodes[idx] = (idx + 1) % num_nodes;
    }
}

__global__ void shuffleNodes(int* nodes, int num_nodes, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        // Simple hash-based pseudo-randomization
        unsigned long long hash = seed ^ idx;
        hash ^= hash << 13;
        hash ^= hash >> 17;
        hash ^= hash << 5;
        
        int target_idx = hash % num_nodes;
        if (target_idx != idx) {
            int temp = nodes[idx];
            nodes[idx] = nodes[target_idx];
            nodes[target_idx] = temp;
        }
    }
}

__global__ void pointerChase(volatile int* nodes, int num_nodes, int start_idx, unsigned long long* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        unsigned long long start_cycle = clock64();
        
        for (int i = 0; i < CHASE_STEPS; i++) {
            idx = nodes[idx];
        }
        
        unsigned long long end_cycle = clock64();
        *result = (end_cycle - start_cycle) / CHASE_STEPS;
    }
}

int main() {
    // Define the sizes to test in KB
    std::vector<int> sizes_kb = {128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 
                                4608, 5120, 5632, 6144, 7168, 8192, 10240, 12288, 16384, 20480, 32768};
    
    std::vector<double> latencies;
    
    for (int size_kb : sizes_kb) {
        int size_bytes = size_kb * 1024;
        int node_size_bytes = 128; // One cache line
        int num_nodes = size_bytes / node_size_bytes;
        
        if (num_nodes == 0) continue;
        
        int *d_nodes;
        unsigned long long *d_result;
        unsigned long long h_result;
        
        cudaMalloc(&d_nodes, num_nodes * sizeof(int));
        cudaMalloc(&d_result, sizeof(unsigned long long));
        
        dim3 init_block(256);
        dim3 init_grid((num_nodes + init_block.x - 1) / init_block.x);
        
        // Initialize nodes in sequence
        initializeNodes<<<init_grid, init_block>>>(d_nodes, num_nodes);
        cudaDeviceSynchronize();
        
        // Shuffle nodes to create random access pattern
        shuffleNodes<<<init_grid, init_block>>>(d_nodes, num_nodes, 12345ULL);
        cudaDeviceSynchronize();
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            pointerChase<<<1, 1>>>(d_nodes, num_nodes, 0, d_result);
            cudaDeviceSynchronize();
        }
        
        // Measurement loop
        std::vector<unsigned long long> measurements;
        for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
            pointerChase<<<1, 1>>>(d_nodes, num_nodes, iter % num_nodes, d_result);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            measurements.push_back(h_result);
        }
        
        // Calculate median latency
        std::sort(measurements.begin(), measurements.end());
        double median_latency = measurements[measurements.size() / 2];
        latencies.push_back(median_latency);
        
        cudaFree(d_nodes);
        cudaFree(d_result);
    }
    
    // Find the cliff in the latency vs size curve
    int l2_cache_size_kb = sizes_kb.back(); // Default to largest size
    
    for (size_t i = 1; i < latencies.size(); i++) {
        double prev_latency = latencies[i-1];
        double curr_latency = latencies[i];
        
        // Check if there's a significant jump (>50% increase)
        if (curr_latency > prev_latency * 1.5) {
            l2_cache_size_kb = sizes_kb[i-1]; // The size before the cliff
            break;
        }
    }
    
    printf("%d\n", l2_cache_size_kb);
    
    return 0;
}