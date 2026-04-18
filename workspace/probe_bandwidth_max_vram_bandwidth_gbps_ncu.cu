#include <cuda_runtime.h>
#include <iostream>

__global__ void bandwidth_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

int main() {
    const size_t array_size = 134217728; // 256MB / sizeof(float) = 67M floats
    const int total_elements = array_size;
    const size_t bytes_per_array = total_elements * sizeof(float);
    
    float* d_input;
    float* d_output;
    
    cudaMalloc(&d_input, bytes_per_array);
    cudaMalloc(&d_output, bytes_per_array);
    
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);
    if (grid.x < 256) grid.x = 256; // Ensure minimum 256 blocks
    
    // Warmup - run kernel 3 times
    for (int i = 0; i < 3; i++) {
        bandwidth_kernel<<<grid, block>>>(d_input, d_output, total_elements);
    }
    cudaDeviceSynchronize();
    
    // Timing events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    const int iterations = 10;
    
    cudaEventRecord(start_event);
    for (int i = 0; i < iterations; i++) {
        bandwidth_kernel<<<grid, block>>>(d_input, d_output, total_elements);
    }
    cudaEventRecord(stop_event);
    
    cudaEventSynchronize(stop_event);
    
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    
    double elapsed_seconds = elapsed_ms / 1000.0;
    double total_bytes_transferred = 2.0 * bytes_per_array * iterations; // Read + Write
    double bandwidth_gbps = total_bytes_transferred / (elapsed_seconds * 1e9);
    
    printf("%.2f\n", bandwidth_gbps);
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}