#include <cuda_runtime.h>
#include <iostream>

#define DATA_SIZE_MB 512
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 256
#define ITERATIONS 10

__global__ void bandwidth_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

int main() {
    const int n = (DATA_SIZE_MB * 1024 * 1024) / sizeof(float);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    // Warm-up
    for (int i = 0; i < 3; i++) {
        bandwidth_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        bandwidth_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    float elapsed_s = elapsed_ms / 1000.0f;
    size_t bytes_transferred = 2ULL * n * sizeof(float) * ITERATIONS;  // Read + Write * Iterations
    float bandwidth_gbps = (float)bytes_transferred / (elapsed_s * 1e9);
    
    printf("%.6f\n", bandwidth_gbps);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}