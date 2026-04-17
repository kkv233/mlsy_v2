from core.llm import LLMClient
from core.agent_base import SpecialistAgent, ProbeTask, ProbeResult


class BoostFrequencyAgent(SpecialistAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm, agent_name="boost_frequency", max_retries=5)

    SYSTEM_PROMPT = """You are an expert CUDA programmer specializing in GPU clock frequency measurement.
Your task is to write micro-benchmarks that determine the actual GPU boost clock frequency under sustained compute load.

Key principles for frequency measurement:
- The most reliable method: compare clock64() cycles with CUDA event elapsed time
- clock64() counts GPU cycles, CUDA events measure wall-clock time
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
   - Record clock64() at start and end of a compute-intensive kernel
   - Record CUDA events before and after the same kernel
   - frequency_MHz = (clock64_end - clock64_start) / (cudaEventElapsed_ms * 1000)
3. First run a warm-up kernel (at least 100ms of compute) to ensure GPU reaches boost clock
4. Run the measurement kernel multiple times and report the median
5. The measurement kernel should be compute-intensive (e.g., many FMAs) to keep GPU at boost
6. Output ONLY the measured frequency in MHz as a single number on the last line
7. Do NOT use cudaGetDeviceProperties for the measurement
8. Do NOT read from nvidia-smi or /proc files for the measurement

Output ONLY the CUDA C++ source code."""

    def probe(self, task: ProbeTask) -> ProbeResult:
        return super().probe(task)
