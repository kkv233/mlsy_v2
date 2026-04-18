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
        ],
        "validation": (
            "For DRAM latency: ncu should show high dram__throughput (confirming actual DRAM access). "
            "For L1 latency: ncu should show low dram and l2 throughput. "
            "For L2 latency: ncu should show high l2 throughput but low dram throughput."
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
        ],
        "validation": (
            "For VRAM: ncu should show dram__throughput close to peak (ideally >70%). "
            "For Shared Memory: ncu should show high l1tex wavefronts with no bank conflicts."
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
        "validation": (
            "Below L2 capacity: ncu should show high l2 throughput, low dram throughput. "
            "Above L2 capacity: ncu should show high dram throughput."
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
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        ],
        "validation": (
            "Compare with nvidia-smi reported clock (as reference only, not as ground truth). "
            "ncu sm__throughput should be high if the kernel is compute-intensive."
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
        "validation": (
            "ncu should report l1tex__data_bank_conflicts_pipe_lsu.sum > 0 for the conflict kernel. "
            "ncu should report l1tex__data_bank_conflicts_pipe_lsu.sum = 0 for the conflict-free kernel."
        ),
    },
}


PRIORITY_ORDER = ["penalty", "l2_capacity", "bandwidth", "frequency", "latency"]


def match_methodology(target_name: str) -> dict:
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
