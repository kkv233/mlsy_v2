#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define SHARED_MEM_SIZE 4096  // 4KB per block
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 64
#define ITERATIONS 100

__global__ void shared_memory_bandwidth_kernel(unsigned long long* result) {
    __shared__ float smem[SHARED_MEM_SIZE / sizeof(float)];
    
    int tid = threadIdx.x;
    unsigned long long start_cycle, end_cycle;
    
    if (tid == 0) {
        start_cycle = clock64();
    }
    
    __syncthreads();
    
    // Perform multiple iterations of read/write operations
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Write to shared memory
        smem[tid] = tid + iter;
        
        // Read from shared memory
        float val = smem[(tid + 1) % THREADS_PER_BLOCK];
        
        // Additional accesses to increase utilization
        smem[(tid + 32) % (SHARED_MEM_SIZE / sizeof(float))] = val + iter;
        float val2 = smem[(tid + 64) % (SHARED_MEM_SIZE / sizeof(float))];
        smem[(tid + 96) % (SHARED_MEM_SIZE / sizeof(float))] = val2 * 2.0f;
    }
    
    __syncthreads();
    
    if (tid == 0) {
        end_cycle = clock64();
        *result = end_cycle - start_cycle;
    }
}

int main() {
    // Warm-up kernel
    unsigned long long* d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));
    
    // Warm-up
    shared_memory_bandwidth_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result);
    cudaDeviceSynchronize();
    
    // Measure clock frequency
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Run the actual measurement kernel
    cudaEventRecord(start);
    shared_memory_bandwidth_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_result);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    unsigned long long cycles;
    cudaMemcpy(&cycles, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // Calculate total bytes transferred
    // Each thread does multiple reads and writes per iteration
    // 4 operations per iteration: 2 writes + 2 reads
    size_t bytes_per_thread_per_iter = 4 * sizeof(float);  // 4 operations: 2 writes + 2 reads
    size_t total_bytes = NUM_BLOCKS * 
                         THREADS_PER_BLOCK * 
                         ITERATIONS * 
                         bytes_per_thread_per_iter;
    
    // Convert cycles to seconds using assumed clock rate (we'll use elapsed time instead)
    double elapsed_seconds = elapsed_ms / 1000.0;
    
    // Calculate bandwidth using elapsed time (more accurate than cycle counting)
    double bandwidth_gbps = (double)total_bytes / (elapsed_seconds * 1e9);
    
    std::cout << bandwidth_gbps << std::endl;
    
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}