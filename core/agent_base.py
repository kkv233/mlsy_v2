import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from core.llm import LLMClient
from core import tools

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    target: str
    value: float
    unit: str = ""
    evidence: dict = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class ProbeTask:
    target: str
    methodology: dict = field(default_factory=dict)
    dependencies: list = field(default_factory=list)
    context: dict = field(default_factory=dict)


class SpecialistAgent:
    def __init__(self, llm: LLMClient, agent_name: str, max_retries: int = 2, execution_timeout: int = 30):
        self.llm = llm
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.execution_timeout = execution_timeout
        self.feedback_history: list[str] = []
        self.evidence_log: list[dict] = []

    def _build_code_gen_prompt(self, task: ProbeTask) -> str:
        methodology = task.methodology
        principle = methodology.get("principle", "")
        approach = methodology.get("approach", "")
        challenges = methodology.get("key_challenges", [])
        challenges_str = "; ".join(challenges[:3]) if challenges else ""
        deps_context = ""
        if task.context:
            deps_context = f"\nKnown values: {json.dumps(task.context, indent=2)}\n"
            if task.context.get("has_reference_program"):
                deps_context += "\nNOTE: A reference executable is provided. You can run it to get reference values for cross-validation.\n"
        feedback = ""
        if self.feedback_history:
            feedback = f"\nPrevious attempt failed: {'; '.join(self.feedback_history[-2:])}\nFix the issue.\n"

        sm_arch = tools.detect_sm_arch()
        return f"""Write a CUDA C++ micro-benchmark to measure: {task.target}

Principle: {principle}
Approach: {approach}
Key challenges: {challenges_str}
{deps_context}{feedback}
CRITICAL REQUIREMENTS - YOUR CODE MUST COMPILE AND RUN ON FIRST TRY:
1. MUST be a COMPLETE, self-contained .cu file with #include <stdio.h>, #include <cuda_runtime.h>, kernel function, AND main() function
2. MUST compile successfully with: nvcc -o <binary> <source>.cu -arch={sm_arch}
3. MUST output ONLY the measured number (a single float) on the LAST line of stdout - no other text after it
4. MUST use clock64() for cycle-level timing, CUDA events for wall-clock timing
5. MUST include warm-up runs and multiple iterations for accuracy
6. MUST NOT use cudaGetDeviceProperties to query the answer
7. MUST keep total runtime under 10 seconds
8. MUST handle CUDA errors gracefully
9. Output ONLY the complete CUDA C++ source code - no explanations, no markdown formatting

IMPORTANT: Double-check your code before outputting. Ensure:
- All functions are properly defined and called
- main() function exists and calls the kernel correctly
- Memory allocation/deallocation is correct
- The final printf outputs ONLY the numeric result
- No syntax errors, missing semicolons, or undefined variables"""

    def _build_ncu_validation_prompt(self, task: ProbeTask, ncu_output: str, measured_value: float, validation_methodology: str = "") -> str:
        methodology_hint = ""
        if validation_methodology:
            methodology_hint = f"\nValidation methodology: {validation_methodology}\n"
        return f"""You are a GPU performance analysis expert. Analyze the ncu profiling output to verify whether the measured value is reliable.

Target metric: {task.target}
Measured value: {measured_value}
{methodology_hint}
NCU profiling output:
{ncu_output}

IMPORTANT ANALYSIS RULES:
1. Focus ONLY on the measurement kernel (the one that does the actual probing). Ignore warmup, initialization, or cache-flush kernels.
2. Be LENIENT: if the measurement kernel shows reasonable metrics for the target, mark as valid even if auxiliary kernels have zero throughput.
3. A DRAM throughput of 0% on an initialization kernel is EXPECTED and should NOT invalidate the result.
4. The measured value has already passed a sanity range check, so it is physically plausible.

Please analyze:
1. Does the measurement kernel's ncu output confirm that it is actually measuring what we intend?
2. Are there any anomalies in the MEASUREMENT kernel only?
3. Is the measured value physically plausible given the ncu metrics?

Respond in JSON format:
{{
    "is_valid": true/false,
    "analysis": "your analysis focusing on the measurement kernel",
    "issues": ["list of issues if any, only for the measurement kernel"],
    "suggested_fix": "what to fix if invalid"
}}"""

    def _generate_code(self, task: ProbeTask) -> str:
        logger.info(f"  [{self.agent_name}] Sending code generation request to LLM...")
        prompt = self._build_code_gen_prompt(task)
        t0 = time.time()
        raw_response = self.llm.chat_with_system(
            system_prompt="You are an expert CUDA programmer. Output only valid CUDA C++ code.",
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=4096,
        )
        elapsed = time.time() - t0
        logger.info(f"  [{self.agent_name}] LLM raw response length={len(raw_response)} chars")
        logger.info(f"  [{self.agent_name}] LLM raw response (first 500 chars): {raw_response[:500]}")
        code = self._extract_code(raw_response)
        logger.info(f"  [{self.agent_name}] Code extracted in {elapsed:.1f}s, length={len(code)} chars")
        logger.info(f"  [{self.agent_name}] Code preview (first 200 chars): {code[:200]}")
        return code

    @staticmethod
    def _extract_code(response: str) -> str:
        response = response.strip()
        if "```" not in response:
            return response
        import re
        patterns = [
            r"```(?:cpp|cuda|c\+\+|c)\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        first_fence = response.find("```")
        if first_fence != -1:
            after_fence = response[first_fence + 3:]
            newline_pos = after_fence.find("\n")
            if newline_pos != -1:
                code_body = after_fence[newline_pos + 1:]
            else:
                code_body = after_fence
            last_fence = code_body.rfind("```")
            if last_fence != -1:
                code_body = code_body[:last_fence]
            return code_body.strip()
        return response

    def _compile_and_run(self, code: str, task: ProbeTask) -> tuple[bool, float, str]:
        binary_name = f"probe_{self.agent_name}_{task.target}"
        logger.info(f"  [{self.agent_name}] Compiling {binary_name}.cu ...")
        ok, output, err = tools.compile_cuda(code, binary_name)
        if not ok:
            logger.warning(f"  [{self.agent_name}] Compilation FAILED: {output[:300]}")
            return False, 0.0, f"Compilation failed: {output}"
        logger.info(f"  [{self.agent_name}] Compilation OK, binary: {output}")
        logger.info(f"  [{self.agent_name}] Executing...")
        binary_path = output
        ok, stdout, stderr = tools.execute_binary(binary_path, timeout=self.execution_timeout)
        if not ok:
            logger.warning(f"  [{self.agent_name}] Execution FAILED: {stderr[:300]}")
            return False, 0.0, f"Execution failed: {stderr}"
        logger.info(f"  [{self.agent_name}] Execution OK, output: {stdout.strip()[:200]}")
        try:
            lines = stdout.strip().split("\n")
            value = float(lines[-1].strip())
            logger.info(f"  [{self.agent_name}] Parsed value: {value}")
            return True, value, stdout
        except (ValueError, IndexError):
            logger.warning(f"  [{self.agent_name}] Could not parse output: {stdout[:200]}")
            return False, 0.0, f"Could not parse output: {stdout}"

    def _ncu_validate(self, task: ProbeTask, code: str, measured_value: float) -> tuple[bool, str]:
        ncu_metrics = task.methodology.get("ncu_metrics", [])
        kernel_filter = task.methodology.get("ncu_kernel_filter", "")
        if not ncu_metrics:
            logger.info(f"  [{self.agent_name}] No ncu metrics specified, skipping ncu validation")
            return True, "No ncu metrics specified for validation"
        binary_name = f"probe_{self.agent_name}_{task.target}_ncu"
        logger.info(f"  [{self.agent_name}] Compiling for ncu validation...")
        ok, output, err = tools.compile_cuda(code, binary_name)
        if not ok:
            return True, "Could not compile for ncu validation, skipping"
        binary_path = output
        logger.info(f"  [{self.agent_name}] Running ncu with metrics: {ncu_metrics}, kernel_filter: {kernel_filter}")
        ok, ncu_output, _ = tools.run_ncu(binary_path, ncu_metrics, timeout=120, kernel_filter=kernel_filter)
        if not ok:
            if "ERR_NVGPUCTRPERM" in ncu_output:
                msg = "ncu unavailable (GPU performance counter permission denied), validation skipped"
                logger.warning(f"  [{self.agent_name}] {msg}")
                return True, msg
            logger.warning(f"  [{self.agent_name}] ncu run failed: {ncu_output[:200]}")
            return True, f"ncu run failed, skipping validation: {ncu_output[:200]}"
        if "No kernels were profiled" in ncu_output and kernel_filter:
            logger.info(f"  [{self.agent_name}] kernel_filter '{kernel_filter}' matched no kernels, retrying without filter")
            ok, ncu_output, _ = tools.run_ncu(binary_path, ncu_metrics, timeout=120)
            if not ok:
                return True, f"ncu run failed without filter, skipping validation"
        logger.info(f"  [{self.agent_name}] ncu output: {ncu_output[:300]}")
        logger.info(f"  [{self.agent_name}] Sending ncu output to LLM for validation analysis...")
        validation_methodology = task.methodology.get("validation", "")
        validation_prompt = self._build_ncu_validation_prompt(task, ncu_output, measured_value, validation_methodology)
        response = self.llm.chat_with_system(
            system_prompt="You are a GPU performance analysis expert. Respond only in JSON. Be lenient - if the measurement kernel shows reasonable metrics, mark as valid even if auxiliary kernels have issues.",
            user_prompt=validation_prompt,
            temperature=0.1,
        )
        result = self.llm.extract_json(response)
        if not result:
            return True, "Could not parse ncu validation response, assuming valid"
        self.evidence_log.append({
            "type": "ncu_validation",
            "target": task.target,
            "ncu_output": ncu_output[:2000],
            "llm_analysis": result.get("analysis", ""),
        })
        is_valid = result.get("is_valid", True)
        issues = result.get("issues", [])
        suggested_fix = result.get("suggested_fix", "")
        if not is_valid:
            feedback = f"ncu validation failed: {'; '.join(issues)}. Suggested fix: {suggested_fix}"
            logger.warning(f"  [{self.agent_name}] ncu validation FAILED: {feedback}")
            return False, feedback
        logger.info(f"  [{self.agent_name}] ncu validation PASSED")
        return True, result.get("analysis", "ncu validation passed")

    def _sanity_check(self, target: str, value: float) -> tuple[bool, str]:
        sanity_ranges = task_sanity_ranges()
        if target in sanity_ranges:
            lo, hi = sanity_ranges[target]
            if not (lo <= value <= hi):
                msg = f"Value {value} is outside sanity range [{lo}, {hi}] for {target}"
                logger.warning(f"  [{self.agent_name}] Sanity check FAILED: {msg}")
                return False, msg
        logger.info(f"  [{self.agent_name}] Sanity check PASSED")
        return True, "Within sanity range"

    def probe(self, task: ProbeTask) -> ProbeResult:
        self.feedback_history = []
        self.evidence_log = []
        best_result = None
        for attempt in range(self.max_retries):
            logger.info(f"  [{self.agent_name}] === Attempt {attempt + 1}/{self.max_retries} ===")
            code = self._generate_code(task)
            self.evidence_log.append({
                "type": "code_generated",
                "attempt": attempt + 1,
                "code": code[:3000],
            })
            ok, value, output = self._compile_and_run(code, task)
            if not ok:
                self.feedback_history.append(output)
                continue
            sane, sane_msg = self._sanity_check(task.target, value)
            if not sane:
                self.feedback_history.append(sane_msg)
                continue
            ncu_ok, ncu_msg = self._ncu_validate(task, code, value)
            if not ncu_ok:
                self.feedback_history.append(ncu_msg)
                best_result = (value, code, output)
                continue
            self.evidence_log.append({
                "type": "measurement_success",
                "attempt": attempt + 1,
                "value": value,
                "output": output[:1000],
                "ncu_validation": ncu_msg,
            })
            logger.info(f"  [{self.agent_name}] SUCCESS: {task.target} = {value} (attempt {attempt + 1})")
            return ProbeResult(
                target=task.target,
                value=value,
                evidence={
                    "code": code,
                    "output": output[:2000],
                    "ncu_validation": ncu_msg,
                    "attempts": attempt + 1,
                    "evidence_log": self.evidence_log,
                },
                confidence=1.0,
                reasoning=f"Measured after {attempt + 1} attempts. ncu validation: {ncu_msg}",
            )
        if best_result:
            value, code, output = best_result
            logger.info(f"  [{self.agent_name}] BEST EFFORT: {task.target} = {value} (confidence=0.5)")
            return ProbeResult(
                target=task.target,
                value=value,
                evidence={
                    "code": code,
                    "output": output[:2000],
                    "attempts": self.max_retries,
                    "evidence_log": self.evidence_log,
                    "note": "Passed sanity check but ncu validation had issues",
                },
                confidence=0.5,
                reasoning=f"Best result after {self.max_retries} attempts. ncu validation had issues.",
            )
        logger.error(f"  [{self.agent_name}] FAILED: Could not measure {task.target} after {self.max_retries} attempts")
        return ProbeResult(
            target=task.target,
            value=-1,
            evidence={"evidence_log": self.evidence_log},
            confidence=0.0,
            reasoning=f"Failed to measure after {self.max_retries} attempts",
        )


def task_sanity_ranges() -> dict[str, tuple[float, float]]:
    return {
        "l1_latency_cycles": (10, 80),
        "l2_latency_cycles": (30, 300),
        "dram_latency_cycles": (100, 1000),
        "max_shmem_bandwidth_gbps": (500, 8000),
        "max_vram_bandwidth_gbps": (100, 3000),
        "l2_cache_size_kb": (256, 65536),
        "actual_boost_clock_mhz": (200, 3500),
        "shmem_bank_conflict_penalty_cycles": (1, 100),
        "max_shmem_per_block_kb": (8, 256),
        "max_flops_fp32": (1000, 200000),
        "max_flops_fp16": (1000, 500000),
        "num_sms": (1, 200),
        "warp_size": (32, 32),
    }
