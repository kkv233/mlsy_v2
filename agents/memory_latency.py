from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class MemoryLatencyAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="memory_latency", max_retries=2, execution_timeout=30)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in memory hierarchy characterization.
Your task is to write micro-benchmarks that measure memory access latency at different levels of the GPU memory hierarchy (L1, L2, DRAM).

Key principles for latency measurement:
- Pointer Chasing is the gold standard: create a chain of dependent memory accesses where each access depends on the data loaded by the previous one
- This defeats hardware prefetchers because the next address is only known after the current load completes
- Different data sizes target different cache levels: small arrays stay in L1, medium in L2, large must go to DRAM
- Use clock64() for cycle-accurate timing inside kernels
- Always include warm-up iterations before measurement
- Report the median of multiple runs to reduce noise
- The output must be a single number on the last line of stdout

Critical: Do NOT use cudaGetDeviceProperties or any runtime API that could be intercepted.
Rely solely on actual hardware measurement."""

    def _build_code_gen_prompt(self, task: ProbeTask) -> str:
        methodology = task.methodology
        deps_context = ""
        if task.context:
            deps_context = f"\nPreviously measured results:\n{task.context}\n"
        feedback = ""
        if self.feedback_history:
            feedback = f"\nPrevious attempts failed:\n" + "\n".join(f"- {fb}" for fb in self.feedback_history) + "\nFix these issues.\n"

        target_lower = task.target.lower()
        size_hint = ""
        extra_requirements = ""
        if "l1" in target_lower:
            size_hint = "Use a small array (4KB-32KB) to ensure data stays in L1 cache."
        elif "l2" in target_lower:
            size_hint = "Use a medium array (256KB-4MB) that fits in L2 but exceeds L1."
        elif "dram" in target_lower:
            size_hint = "Use a large array (64MB-256MB) that far exceeds L2 cache capacity to force DRAM access."
            extra_requirements = """
DRAM LATENCY SPECIFIC REQUIREMENTS:
- Array size MUST be at least 64MB (524288 nodes of 128 bytes each) to exceed L2 cache
- MUST flush L2 cache before each measurement by accessing a separate large buffer
- Pointer chasing MUST use a single thread (threadIdx.x == 0 only)
- Random permutation MUST be done on the GPU before measurement
- Use at least 10000 chase steps per measurement
- Run at least 50 iterations, report median
- The flush kernel should access a different large array (at least 32MB) with stride-1 pattern
  to evict all data from L2 before the pointer chase"""

        return f"""Write a CUDA C++ micro-benchmark to measure: {task.target}

Design methodology:
{methodology}

Size guidance: {size_hint}

{deps_context}{feedback}

Requirements:
1. Complete, self-contained CUDA C++ program (single .cu file)
2. Use pointer chasing with random permutation to defeat prefetcher
3. Each node should be one cache line (128 bytes) with the next pointer at the start
4. MUST use clock64() for timing (NOT clock() which is 32-bit and wraps)
5. Output ONLY the measured latency in cycles as a single number on the last line
6. Do NOT use cudaGetDeviceProperties
7. For DRAM: ensure array size >> L2 cache (at least 64MB)
8. For L1: ensure array size << L1 cache (at most 16KB)
9. For L2: ensure array size > L1 but < L2 (256KB to 2MB)
{extra_requirements}

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
