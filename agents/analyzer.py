import json
from dataclasses import dataclass
from typing import Optional

from core.llm import LLMClient
from core.agent_base import ProbeResult
from core import tools


@dataclass
class AnalysisVerdict:
    all_valid: bool
    retries: list[dict]
    env_anomalies: dict
    cross_validation_notes: list[str]


class AnalyzerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, results: dict[str, ProbeResult], env_anomalies: dict) -> AnalysisVerdict:
        retries = []
        cross_validation_notes = []

        single_issues = self._check_individual_results(results)
        retries.extend(single_issues)

        cross_issues = self._cross_validate(results)
        cross_validation_notes.extend(cross_issues)
        for issue in cross_issues:
            if issue.get("severity") == "high":
                retries.append({
                    "target": issue["target"],
                    "reason": issue["reason"],
                })

        env_issues = self._check_env_consistency(results, env_anomalies)
        for issue in env_issues:
            if issue.get("severity") == "high":
                retries.append({
                    "target": issue["target"],
                    "reason": issue["reason"],
                })

        return AnalysisVerdict(
            all_valid=len(retries) == 0,
            retries=retries,
            env_anomalies=env_anomalies,
            cross_validation_notes=cross_validation_notes,
        )

    def _check_individual_results(self, results: dict[str, ProbeResult]) -> list[dict]:
        issues = []
        for target, result in results.items():
            if result.value < 0:
                issues.append({
                    "target": target,
                    "reason": f"Measurement failed (value={result.value})",
                })
            elif result.confidence < 0.5:
                issues.append({
                    "target": target,
                    "reason": f"Low confidence ({result.confidence}): {result.reasoning}",
                })
        return issues

    def _cross_validate(self, results: dict[str, ProbeResult]) -> list[dict]:
        issues = []
        result_values = {t: r.value for t, r in results.items()}

        latency_targets = [t for t in results if "latency" in t.lower()]
        if len(latency_targets) >= 2:
            latency_vals = {t: results[t].value for t in latency_targets}
            sorted_latencies = sorted(latency_vals.items(), key=lambda x: x[1])
            for i in range(len(sorted_latencies) - 1):
                name_lower = sorted_latencies[i][0].lower()
                name_next_lower = sorted_latencies[i + 1][0].lower()
                if "l1" in name_lower and "l2" in name_next_lower:
                    if sorted_latencies[i + 1][1] < sorted_latencies[i][1]:
                        issues.append({
                            "target": sorted_latencies[i + 1][0],
                            "reason": f"L2 latency ({sorted_latencies[i+1][1]}) should be > L1 latency ({sorted_latencies[i][1]})",
                            "severity": "high",
                        })
                if "l2" in name_lower and "dram" in name_next_lower:
                    if sorted_latencies[i + 1][1] < sorted_latencies[i][1]:
                        issues.append({
                            "target": sorted_latencies[i + 1][0],
                            "reason": f"DRAM latency ({sorted_latencies[i+1][1]}) should be > L2 latency ({sorted_latencies[i][1]})",
                            "severity": "high",
                        })

        if any("vram_bandwidth" in t.lower() or "dram_bandwidth" in t.lower() for t in results):
            if any("boost_clock" in t.lower() or "frequency" in t.lower() or "mhz" in t.lower() for t in results):
                freq_target = next(t for t in results if "clock" in t.lower() or "frequency" in t.lower() or "mhz" in t.lower())
                bw_target = next(t for t in results if "bandwidth" in t.lower())
                freq_mhz = results[freq_target].value
                bw_gbps = results[bw_target].value
                if freq_mhz > 0:
                    theoretical_bw = freq_mhz * 384 * 2 / 8 / 1000
                    if theoretical_bw > 0 and abs(bw_gbps - theoretical_bw) / theoretical_bw > 0.5:
                        issues.append({
                            "target": bw_target,
                            "reason": f"Bandwidth {bw_gbps} GB/s seems inconsistent with frequency {freq_mhz} MHz (theoretical ~{theoretical_bw:.0f} GB/s for 384-bit bus)",
                            "severity": "medium",
                        })

        return issues

    def _check_env_consistency(self, results: dict[str, ProbeResult], env_anomalies: dict) -> list[dict]:
        issues = []
        if "frequency_locked" in env_anomalies:
            locked_freq = env_anomalies["frequency_locked"].get("current_mhz", 0)
            for target, result in results.items():
                if "clock" in target.lower() or "frequency" in target.lower() or "mhz" in target.lower():
                    if abs(result.value - locked_freq) / max(locked_freq, 1) > 0.15:
                        issues.append({
                            "target": target,
                            "reason": f"Measured frequency {result.value} MHz doesn't match nvidia-smi reported {locked_freq} MHz. Environment may have frequency locking.",
                            "severity": "medium",
                        })
        return issues

    def generate_evidence_summary(self, results: dict[str, ProbeResult], env_anomalies: dict) -> str:
        summary_parts = []
        summary_parts.append("=== GPU Hardware Probing Evidence Summary ===\n")
        summary_parts.append("== Environment Anomalies Detected ==")
        if env_anomalies:
            for key, val in env_anomalies.items():
                summary_parts.append(f"  {key}: {val}")
        else:
            summary_parts.append("  None detected")
        summary_parts.append("")

        for target, result in results.items():
            summary_parts.append(f"== Target: {target} ==")
            summary_parts.append(f"  Measured value: {result.value}")
            summary_parts.append(f"  Confidence: {result.confidence}")
            summary_parts.append(f"  Reasoning: {result.reasoning}")
            if result.evidence:
                if "code" in result.evidence:
                    summary_parts.append(f"  Code length: {len(result.evidence['code'])} chars")
                if "ncu_validation" in result.evidence:
                    summary_parts.append(f"  NCU validation: {result.evidence['ncu_validation'][:200]}")
                if "attempts" in result.evidence:
                    summary_parts.append(f"  Attempts: {result.evidence['attempts']}")
            summary_parts.append("")

        return "\n".join(summary_parts)
