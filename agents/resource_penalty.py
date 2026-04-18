from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class ResourcePenaltyAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="resource_penalty", max_retries=8)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in Shared Memory bank conflict characterization.
Your task is to write micro-benchmarks that measure the latency penalty caused by bank conflicts in Shared Memory.

Key principles for bank conflict penalty measurement:
- Shared Memory has 32 banks, each 4 bytes wide
- Successive 4-byte words map to successive banks: word[i] -> bank[i % 32]
- When multiple threads in a warp access different addresses in the same bank, access is serialized
- A 2-way bank conflict means 2 threads hit the same bank, causing 2x serialization
- To measure penalty: compare latency of conflict-free access vs. access with known conflicts
- Penalty = (conflict_latency - conflict_free_latency) / num_accesses

Critical: Do NOT use cudaGetDeviceProperties or any runtime API that could be intercepted."""

    def _build_code_gen_prompt(self, task: ProbeTask) -> str:
        methodology = task.methodology
        deps_context = ""
        if task.context:
            deps_context = f"\nPreviously measured results:\n{task.context}\n"
        feedback = ""
        if self.feedback_history:
            feedback = f"\nPrevious attempts failed:\n" + "\n".join(f"- {fb}" for fb in self.feedback_history) + "\nFix these issues.\n"

        return f"""Write a CUDA C++ micro-benchmark to measure: {task.target}

Design methodology:
{methodology}

{deps_context}{feedback}

Requirements:
1. Complete, self-contained CUDA C++ program (single .cu file)
2. Implement TWO kernels:
   a. conflict_free_kernel: each thread reads smem[tid] (sequential access, no bank conflicts)
   b. conflict_kernel: all threads in a warp read from the SAME bank but DIFFERENT addresses
      Example: thread i accesses smem[i * 32] - all threads access bank 0 but at different addresses
      This creates a 32-way bank conflict (worst case)
3. Both kernels must perform the SAME total number of shared memory read+write operations
4. MUST use clock64() for timing (NOT clock() which is 32-bit)
5. Each kernel: start = clock64(), do N accesses, end = clock64(), compute cycles
6. Penalty = (conflict_cycles - free_cycles) / (WARP_SIZE * num_accesses_per_thread)
7. Run at least 10 iterations, report median
8. Output ONLY the penalty in cycles as a single number on the last line
9. Do NOT use cudaGetDeviceProperties
10. Use __syncthreads() between writes and reads
11. IMPORTANT: The conflict pattern must be smem[i * 32] or smem[i * 32 + offset] where all threads
    map to the same bank. NOT smem[tid * 32 % SIZE] which may map to different addresses/banks.

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
