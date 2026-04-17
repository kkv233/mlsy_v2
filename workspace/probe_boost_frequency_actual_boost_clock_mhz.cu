#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

// Warm-up kernel - compute intensive
__global__ void warmup_kernel(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = data[idx];
    for (int i = 0; i < iterations; i++) {
        val = val * 1.00001f + 0.00001f;
    }
    data[idx] = val;
}

// Measurement kernel - compute intensive with clock64
__global__ void measure_kernel(float *data, int n, int iterations, unsigned long long *clock_deltas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned long long start_clock = 0, end_clock = 0;
    
    if (threadIdx.x == 0) {
        start_clock = clock64();
    }
    
    // Compute-intensive work
    float val = 1.0f;
    if (idx < n) {
        val = data[idx];
    }
    for (int i = 0; i < iterations; i++) {
        val = val * 1.00001f + 0.00001f;
    }
    if (idx < n) {
        data[idx] = val;
    }
    
    if (threadIdx.x == 0) {
        end_clock = clock64();
        clock_deltas[blockIdx.x] = end_clock - start_clock;
    }
}

int main() {
    // Configuration
    const int num_threads = 256;
    const int num_blocks = 128;  // Enough to fill all SMs
    const int n = num_threads * num_blocks;
    const int measure_iterations = 100000;  // FMA iterations per thread for measurement
    const int warmup_iterations = 100000;   // FMA iterations per thread for warmup
    const int num_measurements = 11;        // Odd number for median
    
    // Allocate memory
    float *d_data;
    unsigned long long *d_clock_deltas;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_clock_deltas, num_blocks * sizeof(unsigned long long));
    
    // Initialize data
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);
    
    // Warm-up phase - run until at least 100ms of compute
    cudaEvent_t warmup_start, warmup_end;
    cudaEventCreate(&warmup_start);
    cudaEventCreate(&warmup_end);
    
    float total_warmup_ms = 0.0f;
    while (total_warmup_ms < 200.0f) {  // 200ms warm-up to be safe
        cudaEventRecord(warmup_start);
        warmup_kernel<<<num_blocks, num_threads>>>(d_data, n, warmup_iterations);
        cudaEventRecord(warmup_end);
        cudaEventSynchronize(warmup_end);
        float elapsed;
        cudaEventElapsedTime(&elapsed, warmup_start, warmup_end);
        total_warmup_ms += elapsed;
    }
    
    cudaEventDestroy(warmup_start);
    cudaEventDestroy(warmup_end);
    
    // Measurement phase
    double frequencies[num_measurements];
    unsigned long long *h_clock_deltas = (unsigned long long*)malloc(num_blocks * sizeof(unsigned long long));
    
    for (int run = 0; run < num_measurements; run++) {
        cudaEvent_t evt_start, evt_end;
        cudaEventCreate(&evt_start);
        cudaEventCreate(&evt_end);
        
        cudaEventRecord(evt_start);
        measure_kernel<<<num_blocks, num_threads>>>(d_data, n, measure_iterations, d_clock_deltas);
        cudaEventRecord(evt_end);
        cudaEventSynchronize(evt_end);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, evt_start, evt_end);
        
        cudaMemcpy(h_clock_deltas, d_clock_deltas, num_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        // Average clock deltas across blocks
        double avg_delta = 0.0;
        for (int b = 0; b < num_blocks; b++) {
            avg_delta += (double)h_clock_deltas[b];
        }
        avg_delta /= num_blocks;
        
        // frequency_MHz = avg_delta / (elapsed_ms * 1000)
        double freq_mhz = avg_delta / (elapsed_ms * 1000.0);
        frequencies[run] = freq_mhz;
        
        cudaEventDestroy(evt_start);
        cudaEventDestroy(evt_end);
    }
    
    // Sort and find median
    std::sort(frequencies, frequencies + num_measurements);
    double median_freq = frequencies[num_measurements / 2];
    
    // Output
    printf("%.0f\n", median_freq);
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_clock_deltas);
    free(h_clock_deltas);
    
    return 0;
}