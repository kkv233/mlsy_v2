#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

__global__ void warmup_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        // Perform many FMA operations to create sustained load
        for (int i = 0; i < 10000; i++) {
            val = fmaf(val, val, val);
        }
        data[idx] = val;
    }
}

__global__ void measure_kernel(float* data, int size, unsigned long long* start_clock, unsigned long long* end_clock) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        *start_clock = clock64();
    }
    
    __syncthreads();
    
    if (idx < size) {
        float val = data[idx];
        // Perform many FMA operations to maintain high utilization
        for (int i = 0; i < 10000; i++) {
            val = fmaf(val, val, val);
        }
        data[idx] = val;
    }
    
    __syncthreads();
    
    if (idx == 0) {
        *end_clock = clock64();
    }
}

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    
    const int block_size = 256;
    const int num_blocks = std::max(256, num_sms * 8);  // Ensure we have enough blocks to saturate SMs
    const int total_threads = num_blocks * block_size;
    
    float* d_data;
    cudaMalloc(&d_data, total_threads * sizeof(float));
    cudaMemset(d_data, 1, total_threads * sizeof(float));
    
    // Warm-up phase - run for at least 500ms of sustained compute
    for (int i = 0; i < 10; i++) {  // Run warmup multiple times
        warmup_kernel<<<num_blocks, block_size>>>(d_data, total_threads);
        cudaDeviceSynchronize();
    }
    
    // Measurement phase
    std::vector<float> frequencies;
    unsigned long long *d_start_clock, *d_end_clock;
    cudaMalloc(&d_start_clock, sizeof(unsigned long long));
    cudaMalloc(&d_end_clock, sizeof(unsigned long long));
    
    for (int run = 0; run < 5; run++) {
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        
        cudaEventRecord(start_event);
        
        measure_kernel<<<num_blocks, block_size>>>(d_data, total_threads, d_start_clock, d_end_clock);
        
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
        
        unsigned long long start_clk, end_clk;
        cudaMemcpy(&start_clk, d_start_clock, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&end_clk, d_end_clock, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        unsigned long long clk_diff = end_clk - start_clk;
        float freq_mhz = static_cast<float>(clk_diff) / (elapsed_ms * 1000.0f);
        
        frequencies.push_back(freq_mhz);
        
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    // Calculate median
    std::sort(frequencies.begin(), frequencies.end());
    float median_freq = frequencies[frequencies.size() / 2];
    
    std::cout << median_freq << std::endl;
    
    cudaFree(d_data);
    cudaFree(d_start_clock);
    cudaFree(d_end_clock);
    
    return 0;
}