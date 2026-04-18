import os
import sys
import json
import time
import logging
from pathlib import Path

from core.llm import LLMClient
from core.agent_base import ProbeResult, ProbeTask
from core import tools
from agents.planner import PlannerAgent
from agents.memory_latency import MemoryLatencyAgent
from agents.bandwidth import BandwidthAgent
from agents.l2_capacity import L2CapacityAgent
from agents.boost_frequency import BoostFrequencyAgent
from agents.resource_penalty import ResourcePenaltyAgent
from agents.analyzer import AnalyzerAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SPECIALIST_MAP = {
    "memory_latency": MemoryLatencyAgent,
    "bandwidth": BandwidthAgent,
    "l2_capacity": L2CapacityAgent,
    "frequency": BoostFrequencyAgent,
    "resource_penalty": ResourcePenaltyAgent,
}

MAX_ANALYSIS_RETRIES = 2


def select_specialist(task: ProbeTask, llm: LLMClient):
    category = task.methodology.get("matched_category", "unknown")
    if category in SPECIALIST_MAP:
        return SPECIALIST_MAP[category](llm)
    target_lower = task.target.lower()
    if any(kw in target_lower for kw in ["latency", "delay", "access_time"]):
        return MemoryLatencyAgent(llm)
    if any(kw in target_lower for kw in ["bandwidth", "throughput", "bw"]):
        return BandwidthAgent(llm)
    if any(kw in target_lower for kw in ["l2_cache", "l2_size", "cache_capacity"]):
        return L2CapacityAgent(llm)
    if any(kw in target_lower for kw in ["clock", "frequency", "mhz", "boost"]):
        return BoostFrequencyAgent(llm)
    if any(kw in target_lower for kw in ["bank_conflict", "penalty", "conflict_cost"]):
        return ResourcePenaltyAgent(llm)
    return MemoryLatencyAgent(llm)


def run(target_spec_path: str = "target_spec.json", output_path: str = "results.json"):
    logger.info("Starting GPU Hardware Probing Agent System")
    logger.info(f"Reading target spec from: {target_spec_path}")

    spec_path = Path(target_spec_path)
    if not spec_path.is_absolute():
        spec_path = Path(__file__).resolve().parent / target_spec_path
    if not spec_path.exists():
        logger.error(f"target_spec.json not found at {spec_path}")
        sys.exit(1)

    with open(spec_path) as f:
        target_spec = json.load(f)

    targets = target_spec.get("targets", [])
    run_executable = target_spec.get("run", None)
    logger.info(f"Targets: {targets}")
    if run_executable:
        logger.info(f"Executable provided: {run_executable}")

    llm = LLMClient()
    logger.info(f"LLM: model={llm.model}, base_url={llm.base_url}")

    env_anomalies = tools.check_environment_tampering()
    if env_anomalies:
        logger.warning(f"Environment anomalies detected: {list(env_anomalies.keys())}")
    else:
        logger.info("No environment anomalies detected")

    gpu_info = tools.get_gpu_info()
    logger.info(f"GPU info: {gpu_info}")

    if run_executable:
        logger.info(f"Phase 0: Running provided executable: {run_executable}")
        ok, stdout, stderr = tools.execute_binary(run_executable, timeout=120)
        if ok:
            logger.info(f"Executable output: {stdout.strip()[:500]}")
            try:
                reference_values = json.loads(stdout.strip())
                logger.info(f"Parsed reference values: {reference_values}")
            except json.JSONDecodeError:
                logger.warning("Executable output is not valid JSON, treating as reference data")
                reference_values = {"raw_output": stdout.strip()}
        else:
            logger.warning(f"Executable failed: {stderr[:300]}")
            reference_values = {"error": stderr[:500]}
    else:
        reference_values = {}

    logger.info("Phase 1: Planning")
    planner = PlannerAgent(llm)
    tasks = planner.plan(target_spec)
    logger.info(f"Planned {len(tasks)} tasks in order: {[t.target for t in tasks]}")

    logger.info("Phase 2: Executing specialist agents")
    results: dict[str, ProbeResult] = {}
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Probing: {task.target}")
        task.context.update({
            k: v for k, v in results.items()
            if k in task.dependencies
        })
        task.context["gpu_info"] = gpu_info
        task.context["env_anomalies"] = env_anomalies
        if reference_values:
            task.context["reference_values"] = reference_values

        specialist = select_specialist(task, llm)
        result = specialist.probe(task)
        results[task.target] = result
        logger.info(f"  Result: {task.target} = {result.value} (confidence={result.confidence})")

    logger.info("Phase 3: Analysis and validation")
    analyzer = AnalyzerAgent(llm)
    for retry_round in range(MAX_ANALYSIS_RETRIES):
        verdict = analyzer.analyze(results, env_anomalies)
        if verdict.all_valid:
            logger.info("All results validated successfully")
            break
        logger.warning(f"Analysis round {retry_round + 1}: {len(verdict.retries)} issues found")
        for retry_info in verdict.retries:
            target = retry_info["target"]
            reason = retry_info["reason"]
            logger.warning(f"  Retrying {target}: {reason}")
            matching_task = next((t for t in tasks if t.target == target), None)
            if matching_task:
                matching_task.context.update({
                    k: v for k, v in results.items()
                    if k in matching_task.dependencies
                })
                specialist = select_specialist(matching_task, llm)
                specialist.feedback_history = [reason]
                new_result = specialist.probe(matching_task)
                if new_result.confidence > results[target].confidence or new_result.value >= 0:
                    results[target] = new_result
                    logger.info(f"  Retry result: {target} = {new_result.value} (confidence={new_result.confidence})")

    evidence_summary = analyzer.generate_evidence_summary(results, env_anomalies)
    logger.info("\n" + evidence_summary)

    output = {}
    for target, result in results.items():
        output[target] = result.value

    verdict = analyzer.analyze(results, env_anomalies)

    detailed_evidence = {}
    for target, result in results.items():
        entry = {
            "value": result.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }
        if result.evidence:
            if "code" in result.evidence:
                entry["code"] = result.evidence["code"]
            if "output" in result.evidence:
                entry["execution_output"] = result.evidence["output"]
            if "ncu_validation" in result.evidence:
                entry["ncu_validation"] = result.evidence["ncu_validation"]
            if "attempts" in result.evidence:
                entry["attempts"] = result.evidence["attempts"]
        detailed_evidence[target] = entry

    full_output = {
        "results": output,
        "evidence_summary": evidence_summary,
        "detailed_evidence": detailed_evidence,
        "environment_anomalies": env_anomalies if env_anomalies else None,
        "cross_validation_notes": verdict.cross_validation_notes if verdict.cross_validation_notes else None,
        "gpu_info": gpu_info,
    }

    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = Path(__file__).resolve().parent / output_path
    with open(output_file, "w") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Results written to {output_file}")
    logger.info(f"Final results: {json.dumps(output, indent=2)}")

    return output


if __name__ == "__main__":
    spec_path = sys.argv[1] if len(sys.argv) > 1 else "target_spec.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results.json"
    run(spec_path, out_path)
