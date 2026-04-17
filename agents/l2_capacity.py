from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class L2CapacityAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="l2_capacity", max_retries=5)

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
2. Sweep working set sizes: 128KB, 256KB, 512KB, 1MB, 1.5MB, 2MB, 2.5MB, 3MB, 3.5MB, 4MB, 5MB, 6MB, 8MB, 12MB, 16MB
3. At each size, use pointer chasing with random permutation to measure average access latency
4. Find the size where latency jumps significantly (the 'cliff')
5. Use finer steps around the cliff point for more precision
6. Output ONLY the L2 cache size in KB as a single number on the last line
7. Do NOT use cudaGetDeviceProperties
8. Use clock64() for timing
9. The program should automatically detect the cliff and output the result

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
