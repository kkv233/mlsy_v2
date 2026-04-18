from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class BoostFrequencyAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="boost_frequency", max_retries=5, execution_timeout=60)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in GPU clock frequency measurement.
Your task is to write micro-benchmarks that determine the actual GPU boost clock frequency under sustained compute load.

Key principles for frequency measurement:
- The most reliable method: compare clock64() cycles with CUDA event elapsed time
- clock64() counts GPU SM cycles, CUDA events measure wall-clock time
- frequency_MHz = clock64_cycles / (elapsed_seconds * 1e6)
- Must ensure sustained compute load so GPU reaches stable boost clock
- Run a long compute-intensive kernel before measurement to warm up the GPU
- Do NOT trust cudaGetDeviceProperties - it may report incorrect values
- Do NOT trust nvidia-smi for the final answer - only use actual measurement

Critical: The evaluation environment may lock the GPU at a non-standard frequency.
Your measurement must reflect the ACTUAL frequency, not the spec sheet value."""

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
2. Use the clock64() vs CUDA events method:
   - In the measurement kernel: record start_clock = clock64() at beginning, end_clock = clock64() at end
   - Outside the kernel: use cudaEventRecord before and after kernel launch, then cudaEventElapsedTime
   - frequency_MHz = (end_clock - start_clock) / (elapsed_ms * 1000)
3. CRITICAL: Warm-up phase MUST run for at least 500ms of sustained compute to ensure GPU reaches boost clock
   - Use a separate warmup kernel that does heavy FMA operations for many iterations
   - Launch the warmup kernel at least 10 times sequentially
4. The measurement kernel MUST also be compute-intensive (at least 10000 FMA iterations per thread)
   - Use many threads (at least 256 blocks x 256 threads) to keep all SMs busy
5. Run the measurement at least 5 times and report the MEDIAN frequency
6. Output ONLY the measured frequency in MHz as a single number on the last line
7. Do NOT use cudaGetDeviceProperties for the measurement
8. Do NOT read from nvidia-smi or /proc files for the measurement
9. MUST use clock64() (NOT clock() which is 32-bit and wraps around)
10. The measurement kernel MUST use enough threads to keep ALL SMs busy (at least 256 blocks)

IMPORTANT COMMON MISTAKES TO AVOID:
- Do NOT use a tiny kernel (few threads, few iterations) - the GPU won't reach boost clock
- Do NOT measure during warmup - measure AFTER warmup
- Do NOT use clock() instead of clock64() - clock() wraps at ~2 seconds
- Do NOT divide by elapsed_seconds instead of elapsed_ms*1000 - unit mismatch
- Do NOT use cudaEventElapsedTime in seconds - it returns milliseconds

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
