from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class L2CapacityAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="l2_capacity", max_retries=5, execution_timeout=60)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in cache hierarchy characterization.
Your task is to write micro-benchmarks that determine the L2 cache capacity by finding the latency cliff.

Key principles for L2 capacity measurement:
- Sweep working set sizes from small (below L2) to large (above L2)
- At each size, measure random access latency using pointer chasing
- When working set exceeds L2 capacity, latency will jump sharply (the 'cliff')
- The size at the cliff point is the L2 cache capacity
- Use fine-grained size steps near the expected boundary for precision
- Report the result in KB

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
2. MUST sweep a WIDE range of working set sizes from 128KB to 32MB
   Use these sizes in KB: 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 7168, 8192, 10240, 12288, 16384, 20480, 32768
   This is critical - modern GPUs can have L2 caches from 2MB to 40MB!
3. At each size, use pointer chasing with random permutation to measure average access latency in cycles
4. MUST use clock64() for timing (NOT clock() which is 32-bit)
5. Cliff detection algorithm:
   - Compute latency at each size
   - Find the size where latency increases by more than 50% compared to the previous size
   - That size is the L2 cache capacity (the first size that exceeds L2)
6. Output ONLY the L2 cache size in KB as a single number on the last line
7. Do NOT use cudaGetDeviceProperties
8. Pointer chasing MUST use a single thread (threadIdx.x == 0 only)
9. Each node in the chase should be 128 bytes (one cache line) aligned
10. Use at least 5000 chase steps per size for accuracy
11. Keep total runtime under 30 seconds (reduce iterations if needed for large sizes)

IMPORTANT: The cliff detection MUST be done in the C++ code (not printed as a table for manual analysis).
The program must automatically find the cliff and print ONLY the size in KB.

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
