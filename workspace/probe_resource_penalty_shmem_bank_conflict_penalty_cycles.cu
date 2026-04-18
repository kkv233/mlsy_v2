#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

#define WARP_SIZE 32
#define NUM_THREADS_PER_BLOCK 256
#define NUM_WARPS_PER_BLOCK (NUM_THREADS_PER_BLOCK / WARP_SIZE)
#define SHARED_MEM_SIZE 1024
#define NUM_ACCESS_ITERATIONS 100

__global__ void conflict_free_kernel(unsigned long long* result) {
    __shared__ float smem[SHARED_MEM_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned long long start_cycle, end_cycle;
    
    // Initialize shared memory
    if (tid < SHARED_MEM_SIZE) {
        smem[tid] = tid + 1.0f;
    }
    __syncthreads();
    
    start_cycle = clock64();
    
    float temp = 0.0f;
    for (int iter = 0; iter < NUM_ACCESS_ITERATIONS; iter++) {
        // Conflict-free access: each thread accesses its own bank
        temp += smem[tid];
        smem[tid] = temp;
        __syncthreads(); // Ensure all writes complete before next iteration
    }
    
    end_cycle = clock64();
    
    if (tid == 0) {
        *result = end_cycle - start_cycle;
    }
}

__global__ void conflict_kernel(unsigned long long* result) {
    __shared__ float smem[SHARED_MEM_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned long long start_cycle, end_cycle;
    
    // Initialize shared memory
    if (tid < SHARED_MEM_SIZE) {
        smem[tid] = tid + 1.0f;
    }
    __syncthreads();
    
    start_cycle = clock64();
    
    float temp = 0.0f;
    for (int iter = 0; iter < NUM_ACCESS_ITERATIONS; iter++) {
        // Create bank conflict: all threads in warp access same bank
        // Bank index = (address % 32) -> all threads access address that maps to same bank
        int bank_conflict_addr = (tid * 32) % SHARED_MEM_SIZE;
        temp += smem[bank_conflict_addr];
        smem[bank_conflict_addr] = temp;
        __syncthreads(); // Ensure all writes complete before next iteration
    }
    
    end_cycle = clock64();
    
    if (tid == 0) {
        *result = end_cycle - start_cycle;
    }
}

float calculate_median(float arr[], int n) {
    std::sort(arr, arr + n);
    if (n % 2 == 0) {
        return (arr[n/2 - 1] + arr[n/2]) / 2.0f;
    } else {
        return arr[n/2];
    }
}

int main() {
    const int num_iterations = 10;
    float conflict_times[num_iterations];
    float free_times[num_iterations];
    
    unsigned long long *d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));
    
    dim3 grid(1);
    dim3 block(NUM_THREADS_PER_BLOCK);
    
    for (int i = 0; i < num_iterations; i++) {
        // Run conflict-free kernel
        conflict_free_kernel<<<grid, block>>>(d_result);
        cudaDeviceSynchronize();
        unsigned long long free_cycles;
        cudaMemcpy(&free_cycles, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        free_times[i] = static_cast<float>(free_cycles);
        
        // Run conflict kernel
        conflict_kernel<<<grid, block>>>(d_result);
        cudaDeviceSynchronize();
        unsigned long long conflict_cycles;
        cudaMemcpy(&conflict_cycles, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        conflict_times[i] = static_cast<float>(conflict_cycles);
    }
    
    cudaFree(d_result);
    
    // Calculate median times
    float median_free_time = calculate_median(free_times, num_iterations);
    float median_conflict_time = calculate_median(conflict_times, num_iterations);
    
    // Calculate penalty per access
    // Total number of accesses per warp = NUM_ACCESS_ITERATIONS * WARP_SIZE
    float total_accesses = NUM_ACCESS_ITERATIONS * WARP_SIZE;
    float penalty_per_access = (median_conflict_time - median_free_time) / total_accesses;
    
    printf("%.2f\n", penalty_per_access);
    
    return 0;
}