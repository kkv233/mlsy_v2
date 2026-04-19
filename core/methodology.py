METHODOLOGY_KB = {
    "latency": {
        "keywords": ["latency", "delay", "access_time", "cycles"],
        "category": "memory_latency",
        "principle": (
            "Use Pointer Chasing to measure memory access latency at different hierarchy levels. "
            "Build a linked-list-like data structure in GPU memory where each element contains a pointer/index to the next element. "
            "The kernel traverses this chain sequentially, and each access depends on the previous one, "
            "which prevents the hardware prefetcher from predicting the access pattern."
        ),
        "key_challenges": [
            "Hardware prefetcher can detect sequential or strided access patterns - must use random permutation of node order to defeat it",
            "Data must be placed at the correct hierarchy level: small arrays stay in L1, medium in L2, large in DRAM",
            "clock() and clock64() have limited resolution - need many iterations to average out noise",
            "Cache pollution from previous runs - must flush caches between different hierarchy measurements",
            "Pointer chasing must have true data dependency - each access depends on the data loaded by the previous access"
        ],
        "approach": (
            "1. Allocate arrays of different sizes: small (e.g., 4KB for L1), medium (e.g., 2MB for L2), large (e.g., 64MB for DRAM). "
            "2. Initialize each array as a linked list with nodes in random order (shuffle the next pointers). "
            "3. Each node should be exactly one cache line (128 bytes) to avoid prefetching adjacent lines. "
            "4. CRITICAL: Use volatile pointer or write final result to global memory to prevent compiler from optimizing away the loop. "
            "   Example: 'volatile int* volatile_ptr = nodes; int idx = start; for(...) { idx = volatile_ptr[idx]; } global_result[0] = idx;' "
            "5. The kernel reads node->next, then accesses nodes[node->next], creating a true data dependency chain. "
            "6. Use clock64() to measure total cycles, divide by number of chase steps. "
            "7. Run multiple iterations and report the median. "
            "8. For DRAM measurement, ensure the array is much larger than L2 cache size (e.g., 64MB+ for A10 with 4MB L2)."
        ),
        "ncu_metrics": [
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l2__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "lts__t_sectors_op_read.sum",
        ],
        "ncu_kernel_filter": "pointerChase",
        "validation": (
            "For DRAM latency: ncu dram__throughput will be LOW (0-5%) because pointer chasing only accesses one cache line at a time - this is EXPECTED and CORRECT. "
            "The key validation is that lts__t_sectors_op_read.sum > 0 (confirming actual global memory reads). "
            "For L1 latency: ncu should show very low dram and l2 throughput. "
            "For L2 latency: ncu should show moderate l2 throughput but low dram throughput. "
            "IMPORTANT: Only analyze the pointer chasing kernel, ignore initialization/flush kernels. "
            "DO NOT reject a DRAM latency measurement just because dram__throughput is low - pointer chasing has low bandwidth by design."
        ),
        "size_guidance": {
            "l1": "4KB to 32KB",
            "l2": "256KB to 4MB",
            "dram": "16MB to 256MB",
        },
    },
    "bandwidth": {
        "keywords": ["bandwidth", "throughput", "bw", "gbps", "gbs"],
        "category": "bandwidth",
        "principle": (
            "Measure maximum achievable throughput by performing large, contiguous, coalesced memory operations. "
            "For Global Memory (VRAM) bandwidth: use many threads reading/writing large contiguous arrays. "
            "For Shared Memory bandwidth: use many threads performing parallel reads/writes within a block's shared memory."
        ),
        "key_challenges": [
            "Must use enough threads and large enough data to saturate memory bandwidth",
            "Access pattern must be coalesced - adjacent threads access adjacent addresses",
            "For VRAM bandwidth, data must be large enough to bypass caches (or use cache bypass intrinsics)",
            "For Shared Memory bandwidth, must avoid bank conflicts to measure true peak",
            "Need to separate read bandwidth from write bandwidth, then measure combined"
        ],
        "approach": (
            "For VRAM bandwidth: "
            "1. Allocate a large array (e.g., 256MB) in global memory. "
            "2. Launch many thread blocks with many threads per block. "
            "3. Each thread reads/writes contiguous elements using coalesced access. "
            "4. Use CUDA events for precise timing (not clock64, since we need wall-clock time). "
            "5. Bandwidth = (bytes_read + bytes_written) / elapsed_time_seconds. "
            "6. Report in GB/s. "
            "For Shared Memory bandwidth: "
            "1. Declare a large __shared__ array. "
            "2. Each thread reads and writes to shared memory in a bank-conflict-free pattern. "
            "3. Use clock64() to measure cycles within the kernel. "
            "4. Bandwidth = total_bytes / (cycles / clock_frequency)."
        ),
        "ncu_metrics": [
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__data_pipe_lsu_wavefronts.sum",
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu.sum",
        ],
        "ncu_kernel_filter": "bandwidth",
        "validation": (
            "For VRAM: ncu should show dram__throughput close to peak (ideally >50%). "
            "For Shared Memory: ncu should show high l1tex wavefronts with l1tex__data_bank_conflicts = 0 (no bank conflicts). "
            "IMPORTANT: Only analyze the bandwidth measurement kernel, ignore warmup kernels."
        ),
    },
    "l2_capacity": {
        "keywords": ["l2_cache", "l2_size", "cache_capacity", "l2_cache_size"],
        "category": "l2_capacity",
        "principle": (
            "Gradually increase the working set size of random memory accesses and measure the access latency at each size. "
            "When the working set exceeds the L2 cache capacity, there will be a sharp increase in latency "
            "(the 'cliff' in the latency-vs-size curve). The size at which this jump occurs is the L2 cache capacity."
        ),
        "key_challenges": [
            "The transition may not be perfectly sharp - need fine-grained size steps near the expected boundary",
            "Must use random access pattern to defeat prefetcher (similar to pointer chasing)",
            "Need to sweep a wide range of sizes (e.g., from 128KB to 16MB)",
            "The cliff point may need interpolation between two measured sizes"
        ],
        "approach": (
            "1. For each size S in a range (e.g., 128KB, 256KB, 512KB, 1MB, 2MB, 3MB, 4MB, 5MB, 6MB, 8MB, 12MB, 16MB): "
            "   a. Allocate an array of size S. "
            "   b. Initialize with random pointer-chasing links. "
            "   c. Measure average access latency using pointer chasing. "
            "2. Plot latency vs size and find the size where latency jumps significantly. "
            "3. The jump point indicates the L2 cache capacity. "
            "4. Use finer size steps around the jump point for more precision. "
            "5. Output the L2 cache size in KB."
        ),
        "ncu_metrics": [
            "l2__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        "ncu_kernel_filter": "pointerChase",
        "validation": (
            "Below L2 capacity: ncu should show high l2 throughput, low dram throughput. "
            "Above L2 capacity: ncu should show high dram throughput. "
            "IMPORTANT: Only analyze the pointer chasing kernel, ignore initialization kernels."
        ),
    },
    "frequency": {
        "keywords": ["clock", "frequency", "mhz", "boost", "clock_rate"],
        "category": "frequency",
        "principle": (
            "Measure the actual GPU clock frequency under sustained compute load by executing a known number of FLOPs "
            "and measuring the wall-clock time. Frequency = (total_FLOPs / elapsed_seconds) / (FLOPs_per_cycle_per_SM * num_SMs). "
            "Alternatively, use clock64() inside a kernel with a known number of cycles to derive the clock rate "
            "by comparing with CUDA event timing."
        ),
        "key_challenges": [
            "GPU may dynamically adjust frequency - must ensure sustained load to reach stable boost clock",
            "cudaGetDeviceProperties may report incorrect values (API can be intercepted)",
            "Need to distinguish between base clock and boost clock",
            "Power throttling may reduce frequency during measurement",
            "Must NOT rely on nvidia-smi or CUDA API for the final answer - only use actual measurement"
        ],
        "approach": (
            "Method 1 - FLOPS-based: "
            "1. Write a kernel that performs a known, large number of FP32 multiply-add operations. "
            "2. Use CUDA events to measure wall-clock time. "
            "3. Total FLOPs = 2 * num_elements * iterations (2 FLOPs per FMA). "
            "4. Achieved GFLOPS = total_FLOPs / elapsed_seconds / 1e9. "
            "5. If you know the number of SMs and FMA units per SM, derive frequency. "
            "Method 2 - clock64 vs CUDA events: "
            "1. Run a kernel that records both clock64() and uses CUDA events for timing. "
            "2. clock_cycles = clock64_end - clock64_start. "
            "3. wall_time = cudaEventElapsedTime. "
            "4. frequency_mhz = clock_cycles / (wall_time_ms * 1000). "
            "This method directly measures the actual clock frequency without needing to know SM count."
        ),
        "ncu_metrics": [
            "gpu__clocks.max",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        "ncu_kernel_filter": "measure",
        "validation": (
            "Compare with nvidia-smi reported clock (as reference only, not as ground truth). "
            "ncu gpu__clocks.max should be close to the measured frequency. "
            "IMPORTANT: Only analyze the measurement kernel, ignore warmup kernels."
        ),
    },
    "penalty": {
        "keywords": ["bank_conflict", "penalty", "conflict_cost", "bank", "shared_memory_penalty"],
        "category": "resource_penalty",
        "principle": (
            "Measure the latency difference between conflict-free Shared Memory access and access with bank conflicts. "
            "Shared Memory has 32 banks; when multiple threads in a warp access different addresses in the same bank, "
            "the accesses are serialized, causing a bank conflict penalty."
        ),
        "key_challenges": [
            "Must precisely control the shared memory access pattern to create known bank conflicts",
            "Need to measure very small timing differences - bank conflict penalty is typically just a few cycles",
            "Must ensure no other factors (like uncoalesced global access) contaminate the measurement",
            "Need to distinguish between 2-way, 3-way, etc. conflicts"
        ],
        "approach": (
            "1. Write two kernels: "
            "   a. Conflict-free kernel: each thread accesses shared memory at index threadIdx.x (sequential, no conflict). "
            "   b. Conflict kernel: each thread accesses shared memory at index (threadIdx.x * 32) % array_size, "
            "      causing all threads in a warp to access the same bank. "
            "2. Both kernels perform the same number of shared memory reads/writes. "
            "3. Use clock64() to measure cycles for each kernel. "
            "4. Bank conflict penalty = (conflict_cycles - conflict_free_cycles) / num_accesses. "
            "5. Run multiple iterations and report the average penalty in cycles."
        ),
        "ncu_metrics": [
            "l1tex__data_bank_conflicts_pipe_lsu.sum",
            "l1tex__data_pipe_lsu_wavefronts.sum",
        ],
        "ncu_kernel_filter": "conflict",
        "validation": (
            "ncu should report l1tex__data_bank_conflicts_pipe_lsu.sum > 0 for the conflict kernel. "
            "ncu should report l1tex__data_bank_conflicts_pipe_lsu.sum = 0 for the conflict-free kernel. "
            "IMPORTANT: Only analyze the conflict/conflict_free kernels, ignore other kernels."
        ),
    },
}


NCU_METRIC_KB = {
    "launch__sm_count": {
        "matched_category": "frequency",
        "description": "Number of SMs (Streaming Multiprocessors) on the GPU",
        "unit": "count (integer)",
        "principle": "Count the number of SMs by launching a kernel that uses all available SMs and measuring how many blocks can run concurrently. Each SM can run a limited number of blocks; by observing the concurrency pattern, we can deduce the SM count.",
        "approach": "1. Launch a kernel where each block records its block ID and a timestamp using clock64(). 2. Use enough blocks to fill all SMs. 3. Blocks that start at the same time are running on different SMs. 4. Count the number of blocks that start within the same small time window - that's the SM count. 5. Alternatively, use cudaGetDeviceProperties to get the SM count directly (this is an attribute, not a secret).",
        "key_challenges": ["Must distinguish between concurrent and sequential block execution", "Need precise timing to identify concurrent blocks"],
        "ncu_metrics": ["launch__sm_count"],
        "validation": "Compare measured SM count with cudaGetDeviceProperties result.",
        "output_format": "Output the integer SM count (e.g., 82 for A10)",
    },
    "dram__bytes_read.sum.per_second": {
        "matched_category": "bandwidth",
        "description": "DRAM read bandwidth in bytes per second",
        "unit": "bytes/second (e.g., 889928555010.706 for ~890 GB/s)",
        "principle": "Measure the maximum DRAM read throughput by performing large, coalesced read operations from global memory. The result should be in bytes per second (NOT GB/s).",
        "approach": "1. Allocate a large array (>=256MB) in global memory. 2. Launch many thread blocks with coalesced read access. 3. Use CUDA events for timing. 4. bytes_per_second = total_bytes_read / elapsed_seconds. 5. Output the raw bytes/second value (e.g., 889928555010.706).",
        "key_challenges": ["Must output in bytes/second, NOT GB/s", "Must saturate DRAM bandwidth with enough threads and data"],
        "ncu_metrics": ["dram__bytes_read.sum.per_second"],
        "validation": "Compare with ncu dram__bytes_read.sum.per_second metric.",
        "output_format": "Output the value in bytes/second as a large number (e.g., 889928555010.706)",
    },
    "dram__bytes_write.sum.per_second": {
        "matched_category": "bandwidth",
        "description": "DRAM write bandwidth in bytes per second",
        "unit": "bytes/second (e.g., 863088076676.897 for ~863 GB/s)",
        "principle": "Measure the maximum DRAM write throughput by performing large, coalesced write operations to global memory. The result should be in bytes per second (NOT GB/s).",
        "approach": "1. Allocate a large array (>=256MB) in global memory. 2. Launch many thread blocks with coalesced write access. 3. Use CUDA events for timing. 4. bytes_per_second = total_bytes_written / elapsed_seconds. 5. Output the raw bytes/second value (e.g., 863088076676.897).",
        "key_challenges": ["Must output in bytes/second, NOT GB/s", "Must saturate DRAM bandwidth with enough threads and data"],
        "ncu_metrics": ["dram__bytes_write.sum.per_second"],
        "validation": "Compare with ncu dram__bytes_write.sum.per_second metric.",
        "output_format": "Output the value in bytes/second as a large number (e.g., 863088076676.897)",
    },
    "device__attribute_max_gpu_frequency_khz": {
        "matched_category": "frequency",
        "description": "Maximum GPU SM clock frequency in kHz",
        "unit": "kHz (e.g., 1695000 for 1695 MHz)",
        "principle": "Measure the actual maximum GPU SM clock frequency under sustained compute load. The result must be in kHz (NOT MHz). 1 MHz = 1000 kHz.",
        "approach": "1. Use clock64() vs CUDA events method to measure actual frequency in MHz. 2. Multiply by 1000 to convert to kHz. 3. Output the kHz value (e.g., 1695000).",
        "key_challenges": ["Must output in kHz, NOT MHz", "GPU may throttle under load", "Must ensure sustained load to reach max frequency"],
        "ncu_metrics": ["gpu__clocks.max"],
        "validation": "Compare measured frequency with ncu gpu__clocks.max.",
        "output_format": "Output the frequency in kHz (e.g., 1695000, NOT 1695)",
    },
    "device__attribute_max_mem_frequency_khz": {
        "matched_category": "frequency",
        "description": "Maximum GPU memory clock frequency in kHz",
        "unit": "kHz (e.g., 9751000 for 9751 MHz)",
        "principle": "Measure the maximum memory clock frequency. The result must be in kHz (NOT MHz). This can be derived from measured VRAM bandwidth and bus width: mem_freq_khz = (measured_bandwidth_bytes_per_sec * 8) / (bus_width * 2). Or use cudaGetDeviceProperties to get the memory clock rate.",
        "approach": "1. Use cudaGetDeviceProperties to get memory clock rate in kHz. 2. This is an attribute that can be queried directly. 3. Output in kHz.",
        "key_challenges": ["Must output in kHz, NOT MHz", "Memory clock is typically much higher than SM clock"],
        "ncu_metrics": ["gpu__clocks.max"],
        "validation": "Compare with ncu reported memory frequency.",
        "output_format": "Output the frequency in kHz (e.g., 9751000, NOT 9751)",
    },
    "device__attribute_fb_bus_width": {
        "matched_category": "bandwidth",
        "description": "Frame buffer (VRAM) bus width in bits",
        "unit": "bits (integer, e.g., 384)",
        "principle": "The frame buffer bus width is a hardware attribute that determines how many bits can be transferred per memory clock cycle. It can be queried using cudaGetDeviceProperties.",
        "approach": "1. Use cudaGetDeviceProperties to get the memoryBusWidth attribute. 2. This is a device attribute that can be directly queried. 3. Output the integer bus width in bits.",
        "key_challenges": ["Must output in bits (integer), NOT bytes", "This is a static attribute, not a measured value"],
        "ncu_metrics": [],
        "validation": "Compare with known bus widths for the GPU model.",
        "output_format": "Output the integer bus width in bits (e.g., 384)",
    },
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": {
        "matched_category": "frequency",
        "description": "SM throughput as a percentage of peak sustained throughput during the elapsed time",
        "unit": "percentage (e.g., 98.07)",
        "principle": "Measure the SM utilization percentage under a compute-intensive workload. This represents how close the workload gets to peak SM throughput.",
        "approach": "1. Run a compute-intensive kernel (heavy FMA operations) that maximally utilizes all SMs. 2. Use ncu to measure sm__throughput.avg.pct_of_peak_sustained_elapsed. 3. Or estimate from measured FLOPS vs theoretical peak FLOPS. 4. Output as a percentage (0-100).",
        "key_challenges": ["Must output as percentage (0-100), NOT as a ratio (0-1)", "Must run a workload that actually achieves high SM utilization"],
        "ncu_metrics": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        "validation": "Compare with ncu measured value.",
        "output_format": "Output as a percentage (e.g., 98.07, NOT 0.9807)",
    },
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": {
        "matched_category": "bandwidth",
        "description": "Compute memory throughput as a percentage of peak sustained throughput during elapsed time",
        "unit": "percentage (e.g., 90.8575)",
        "principle": "Measure the compute-memory throughput percentage under a memory-intensive workload. This represents how close the workload gets to peak memory throughput.",
        "approach": "1. Run a memory-intensive kernel (large coalesced reads/writes) that maximally utilizes memory bandwidth. 2. Use ncu to measure gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed. 3. Or estimate from measured bandwidth vs theoretical peak bandwidth. 4. Output as a percentage (0-100).",
        "key_challenges": ["Must output as percentage (0-100), NOT as a ratio (0-1)", "Must run a workload that actually achieves high memory utilization"],
        "ncu_metrics": ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
        "validation": "Compare with ncu measured value.",
        "output_format": "Output as a percentage (e.g., 90.8575, NOT 0.908575)",
    },
}


PRIORITY_ORDER = ["penalty", "l2_capacity", "bandwidth", "frequency", "latency"]


def match_methodology(target_name: str) -> dict:
    if target_name in NCU_METRIC_KB:
        return {**NCU_METRIC_KB[target_name], "matched_category": NCU_METRIC_KB[target_name]["matched_category"]}
    target_lower = target_name.lower()
    for category in PRIORITY_ORDER:
        info = METHODOLOGY_KB[category]
        if any(kw in target_lower for kw in info["keywords"]):
            return {**info, "matched_category": category}
    return {
        "matched_category": "unknown",
        "keywords": [],
        "category": "unknown",
        "principle": f"Unknown target: {target_name}. Need to reason about what this metric means and how to measure it.",
        "key_challenges": ["Unknown metric - must infer measurement approach from the name"],
        "approach": "Analyze the target name to determine what hardware characteristic it represents, then design an appropriate micro-benchmark.",
        "ncu_metrics": [],
        "validation": "Cross-verify with ncu metrics if possible.",
    }
