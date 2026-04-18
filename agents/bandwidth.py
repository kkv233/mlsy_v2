from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class BandwidthAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="bandwidth", max_retries=5, execution_timeout=60)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in memory bandwidth characterization.
Your task is to write micro-benchmarks that measure the maximum achievable throughput of different memory subsystems.

Key principles for bandwidth measurement:
- For Global Memory (VRAM) bandwidth: use many threads performing large, contiguous, coalesced reads/writes
- For Shared Memory bandwidth: use many threads performing parallel bank-conflict-free reads/writes within a block
- Must saturate the memory subsystem with enough threads and data
- Use CUDA events for wall-clock timing for VRAM bandwidth
- Use clock64() for Shared Memory bandwidth timing (NOT clock() which is 32-bit and wraps)
- Bandwidth = total_bytes_transferred / elapsed_time
- Report in GB/s

Critical: Do NOT use cudaGetDeviceProperties or any runtime API that could be intercepted."""

    def _build_code_gen_prompt(self, task: ProbeTask) -> str:
        methodology = task.methodology
        deps_context = ""
        if task.context:
            deps_context = f"\nPreviously measured results:\n{task.context}\n"
        feedback = ""
        if self.feedback_history:
            feedback = f"\nPrevious attempts failed:\n" + "\n".join(f"- {fb}" for fb in self.feedback_history) + "\nFix these issues.\n"

        target_lower = task.target.lower()
        mem_type = ""
        extra_requirements = ""
        if "shmem" in target_lower or "shared" in target_lower:
            mem_type = "Shared Memory bandwidth - use __shared__ memory with bank-conflict-free access patterns"
            extra_requirements = """
SHARED MEMORY SPECIFIC REQUIREMENTS:
- MUST use clock64() for timing (NOT clock() which is 32-bit and wraps around)
- Use __shared__ arrays of at least 4KB per block
- Use at least 128 threads per block, multiple blocks
- Each thread should perform many read+write operations to __shared__ memory
- Access pattern: smem[tid] to avoid bank conflicts (sequential access)
- Bandwidth = total_bytes / (elapsed_cycles / measured_boost_clock_mhz * 1e-3) / 1e9
- OR use CUDA events for timing: bandwidth = total_bytes / elapsed_seconds / 1e9
- Run at least 100 iterations of the access loop inside the kernel"""
        elif "vram" in target_lower or "global" in target_lower or "dram" in target_lower:
            mem_type = "Global Memory (VRAM) bandwidth - use large arrays in global memory with coalesced access"
            extra_requirements = """
VRAM SPECIFIC REQUIREMENTS:
- MUST use at least 512MB of data (2 arrays: input and output, each >=256MB)
- MUST use at least 256 blocks with at least 256 threads per block
- Each thread reads from input[i] and writes to output[i] (coalesced access)
- Run at least 10 iterations of the copy kernel
- Use CUDA events for timing (cudaEventRecord/cudaEventElapsedTime)
- bandwidth_GB = 2 * data_size_bytes * iterations / (elapsed_seconds * 1e9)
  where 2 accounts for read+write, iterations is the number of kernel launches
- Warm up by running the kernel 3 times before measurement"""

        return f"""Write a CUDA C++ micro-benchmark to measure: {task.target}

Design methodology:
{methodology}

Memory type: {mem_type}

{deps_context}{feedback}

Requirements:
1. Complete, self-contained CUDA C++ program (single .cu file)
2. Use enough threads (at least 256 per block, many blocks) to saturate bandwidth
3. Use large data sizes (at least 256MB for VRAM) to bypass caches
4. Ensure coalesced memory access patterns
5. Use CUDA events for precise timing
6. Include warm-up before measurement
7. Output ONLY the measured bandwidth in GB/s as a single number on the last line
8. Do NOT use cudaGetDeviceProperties
9. For VRAM: bandwidth_GB = (bytes_read + bytes_written) / (elapsed_seconds * 1e9)
10. For Shared Memory: MUST use clock64() (NOT clock()) and compute from cycle count
{extra_requirements}

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
