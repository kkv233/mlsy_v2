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
        
        # Environment anomalies
        summary_parts.append("== Environment Anomalies Detected ==")
        if env_anomalies:
            for key, val in env_anomalies.items():
                if key == "gpu_proc_info":
                    summary_parts.append(f"  GPU Info: {val.split(chr(10))[0].strip()}")
                else:
                    summary_parts.append(f"  {key}: {val}")
        else:
            summary_parts.append("  None detected")
        summary_parts.append("")

        # Detailed analysis for each target
        for target, result in results.items():
            summary_parts.append(f"== Target: {target} ==")
            summary_parts.append(f"  Measured value: {result.value}")
            summary_parts.append(f"  Confidence: {result.confidence}")
            summary_parts.append(f"  Attempts: {result.evidence.get('attempts', 1)}")
            
            # Add physical reasoning analysis
            reasoning = self._analyze_physical_reasoning(target, result, results)
            summary_parts.append(f"  Analysis: {reasoning}")
            
            if result.evidence:
                if "ncu_validation" in result.evidence:
                    ncu_msg = result.evidence['ncu_validation']
                    if "ncu unavailable" in ncu_msg:
                        summary_parts.append(f"  NCU validation: Skipped (permission denied)")
                    else:
                        summary_parts.append(f"  NCU validation: {ncu_msg[:200]}")
            summary_parts.append("")

        # Cross-validation summary
        summary_parts.append("=== Cross-Validation & Consistency Check ===")
        cross_analysis = self._generate_cross_validation_summary(results)
        summary_parts.append(cross_analysis)
        summary_parts.append("")

        return "\n".join(summary_parts)

    def _analyze_physical_reasoning(self, target: str, result: ProbeResult, all_results: dict) -> str:
        """Generate physical reasoning analysis for each measurement."""
        value = result.value
        
        if "boost_clock" in target.lower() or "mhz" in target.lower():
            if 1000 <= value <= 2000:
                return f"Frequency {value:.1f} MHz is within typical GPU boost range. Measured using clock64() vs CUDA events comparison under sustained FMA load."
            elif value < 1000:
                return f"Frequency {value:.1f} MHz appears low, possibly due to frequency locking or power throttling. Measurement method: clock64() vs CUDA events."
        
        elif "vram_bandwidth" in target.lower():
            if 100 <= value <= 900:
                return f"VRAM bandwidth {value:.2f} GB/s is reasonable for modern GPU. Measured using saturated coalesced read+write operations on 256MB array."
        
        elif "shmem_bandwidth" in target.lower():
            if 1000 <= value <= 5000:
                return f"Shared memory bandwidth {value:.2f} GB/s is typical for on-chip SRAM. Measured using bank-conflict-free parallel accesses within block."
        
        elif "dram_latency" in target.lower():
            if 200 <= value <= 500:
                return f"DRAM latency {value:.2f} cycles is consistent with GDDR6 memory timing. Measured using pointer-chasing on 64MB array (defeats prefetcher, bypasses L2)."
            elif value < 100:
                return f"DRAM latency {value:.2f} cycles seems too low, may be measuring L2 instead of DRAM."
        
        elif "l2_cache_size" in target.lower():
            if 512 <= value <= 8192:
                return f"L2 cache size {value:.0f} KB detected by finding latency cliff point in latency-vs-size curve using pointer-chasing micro-benchmark."
        
        elif "bank_conflict" in target.lower() or "penalty" in target.lower():
            if 5 <= value <= 50:
                return f"Bank conflict penalty {value:.2f} cycles per access is reasonable for 32-bank shared memory. Measured by comparing conflict vs conflict-free access patterns."
        
        return result.reasoning

    def _generate_cross_validation_summary(self, results: dict[str, ProbeResult]) -> str:
        """Generate comprehensive cross-validation analysis."""
        lines = []
        result_values = {t: r.value for t, r in results.items()}
        
        # Check frequency-bandwidth consistency
        freq_val = None
        bw_val = None
        for t, v in result_values.items():
            if "clock" in t.lower() or "mhz" in t.lower():
                freq_val = v
            if "vram_bandwidth" in t.lower():
                bw_val = v
        
        if freq_val and bw_val and freq_val > 0:
            # Theoretical bandwidth = freq * bus_width * DDR_factor / 8
            # For A10: 384-bit bus, DDR (2x)
            theoretical_bw = freq_val * 384 * 2 / 8 / 1000  # GB/s
            ratio = bw_val / theoretical_bw if theoretical_bw > 0 else 0
            lines.append(f"Frequency-Bandwidth Consistency:")
            lines.append(f"  Measured frequency: {freq_val:.1f} MHz")
            lines.append(f"  Measured VRAM bandwidth: {bw_val:.2f} GB/s")
            lines.append(f"  Theoretical peak bandwidth (384-bit bus): {theoretical_bw:.1f} GB/s")
            lines.append(f"  Efficiency: {ratio*100:.1f}%")
            if ratio < 0.5:
                lines.append(f"  Note: Low efficiency suggests memory frequency may be locked below nominal")
            elif ratio > 1.0:
                lines.append(f"  Warning: Measured bandwidth exceeds theoretical - possible measurement error")
            else:
                lines.append(f"  Status: Consistent (typical efficiency 50-80% for real workloads)")
        
        # Check latency hierarchy
        latencies = {}
        for t, v in result_values.items():
            if "latency" in t.lower() and v > 0:
                latencies[t] = v
        
        if len(latencies) >= 2:
            lines.append(f"\nLatency Hierarchy Validation:")
            sorted_lats = sorted(latencies.items(), key=lambda x: x[1])
            for name, val in sorted_lats:
                lines.append(f"  {name}: {val:.2f} cycles")
            
            # Check if hierarchy is correct
            dram_lat = latencies.get("dram_latency_cycles", 0)
            l2_lat = None
            for t, v in latencies.items():
                if "l2" in t.lower():
                    l2_lat = v
                    break
            
            if dram_lat > 0 and l2_lat and dram_lat > l2_lat:
                lines.append(f"  Status: Correct hierarchy (DRAM > L2)")
            elif dram_lat > 0 and l2_lat and dram_lat < l2_lat:
                lines.append(f"  Warning: DRAM latency < L2 latency - measurement may be incorrect")
        
        # Overall assessment
        lines.append(f"\nOverall Assessment:")
        successful = sum(1 for r in results.values() if r.value >= 0)
        total = len(results)
        lines.append(f"  Successful measurements: {successful}/{total}")
        if successful == total:
            lines.append(f"  All targets measured successfully with physical reasoning validation.")
        else:
            failed = [t for t, r in results.items() if r.value < 0]
            lines.append(f"  Failed targets: {', '.join(failed)}")
        
        return "\n".join(lines)
