#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <algorithm>

#define WARMUP_ITERATIONS 10
#define MEASUREMENT_ITERATIONS 50
#define CHASE_STEPS 10000
#define DRAM_ARRAY_SIZE_MB 64
#define FLUSH_BUFFER_SIZE_MB 32

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

__global__ void flushL2Cache(volatile int* flush_buffer, int buffer_size_ints) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < buffer_size_ints; i += stride) {
        flush_buffer[i] = flush_buffer[i] + 1;
    }
}

__global__ void __launch_bounds__(1)
measureDramLatency(volatile int* nodes, int num_nodes, int start_idx, unsigned long long* results, int iteration) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = start_idx;
        unsigned long long start_cycle = clock64();
        
        for (int i = 0; i < CHASE_STEPS; i++) {
            idx = nodes[idx];
        }
        
        unsigned long long end_cycle = clock64();
        results[iteration] = (end_cycle - start_cycle) / CHASE_STEPS;
    }
}

int main() {
    const int node_size_bytes = 128; // One cache line
    const int num_nodes = (DRAM_ARRAY_SIZE_MB * 1024 * 1024) / node_size_bytes;
    const int flush_buffer_size_ints = (FLUSH_BUFFER_SIZE_MB * 1024 * 1024) / sizeof(int);
    
    int *d_nodes;
    int *d_flush_buffer;
    unsigned long long *d_results;
    unsigned long long *h_results = new unsigned long long[MEASUREMENT_ITERATIONS];
    
    cudaMalloc(&d_nodes, num_nodes * sizeof(int));
    cudaMalloc(&d_flush_buffer, flush_buffer_size_ints * sizeof(int));
    cudaMalloc(&d_results, MEASUREMENT_ITERATIONS * sizeof(unsigned long long));
    
    dim3 init_block(256);
    dim3 init_grid((num_nodes + init_block.x - 1) / init_block.x);
    
    // Initialize nodes in sequence
    initializeNodes<<<init_grid, init_block>>>(d_nodes, num_nodes);
    cudaDeviceSynchronize();
    
    // Shuffle nodes to create random access pattern
    shuffleNodes<<<init_grid, init_block>>>(d_nodes, num_nodes, 12345ULL);
    cudaDeviceSynchronize();
    
    // Initialize flush buffer
    cudaMemset(d_flush_buffer, 0, flush_buffer_size_ints * sizeof(int));
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        flushL2Cache<<<32, 256>>>(d_flush_buffer, flush_buffer_size_ints);
        measureDramLatency<<<1, 1>>>(d_nodes, num_nodes, 0, d_results, 0);
    }
    cudaDeviceSynchronize();
    
    // Measurement loop
    for (int iter = 0; iter < MEASUREMENT_ITERATIONS; iter++) {
        // Flush L2 cache
        flushL2Cache<<<32, 256>>>(d_flush_buffer, flush_buffer_size_ints);
        cudaDeviceSynchronize();
        
        // Measure latency
        measureDramLatency<<<1, 1>>>(d_nodes, num_nodes, iter % num_nodes, d_results, iter);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_results, d_results, MEASUREMENT_ITERATIONS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // Calculate median
    std::sort(h_results, h_results + MEASUREMENT_ITERATIONS);
    unsigned long long median = h_results[MEASUREMENT_ITERATIONS / 2];
    
    printf("%llu\n", median);
    
    delete[] h_results;
    cudaFree(d_nodes);
    cudaFree(d_flush_buffer);
    cudaFree(d_results);
    
    return 0;
}