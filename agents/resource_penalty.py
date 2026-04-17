from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class ResourcePenaltyAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="resource_penalty", max_retries=5)

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
   a. Conflict-free kernel: threads access shared memory sequentially (thread i accesses index i)
   b. Conflict kernel: threads access shared memory with a pattern that causes bank conflicts
      (e.g., thread i accesses index i * stride where stride causes same-bank access)
3. Both kernels must perform the SAME total number of shared memory operations
4. Use clock64() to measure cycles for each kernel
5. Bank conflict penalty = (conflict_cycles - free_cycles) / num_accesses_per_thread
6. Run multiple iterations and report the median
7. Output ONLY the penalty in cycles as a single number on the last line
8. Do NOT use cudaGetDeviceProperties
9. Use __syncthreads() between writes and reads to ensure proper ordering
10. Make sure the conflict pattern actually causes bank conflicts (verify with ncu later)

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
