import os
import subprocess
import tempfile
import shutil
import re
from pathlib import Path


WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/root/autodl-tmp/mlsy_v2/workspace"))


def _run_command(cmd: str, cwd: str = None, timeout: int = 120) -> tuple[bool, str, str]:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def compile_cuda(source_code: str, output_name: str, extra_flags: str = "") -> tuple[bool, str, str]:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    src_path = WORKSPACE / f"{output_name}.cu"
    bin_path = WORKSPACE / output_name
    src_path.write_text(source_code)
    cmd = f"nvcc -o {bin_path} {src_path} -arch=sm_89 {extra_flags} 2>&1"
    ok, stdout, stderr = _run_command(cmd)
    combined = stdout + stderr
    if not ok:
        return False, combined, ""
    return True, str(bin_path), ""


def execute_binary(binary_path: str, args: str = "", timeout: int = 60) -> tuple[bool, str, str]:
    cmd = f"{binary_path} {args}"
    ok, stdout, stderr = _run_command(cmd, timeout=timeout)
    return ok, stdout, stderr


def run_ncu(binary_path: str, metrics: list[str], args: str = "", timeout: int = 120) -> tuple[bool, str, str]:
    metrics_str = ",".join(metrics)
    cmd = f"ncu --metrics {metrics_str} {binary_path} {args} 2>&1"
    ok, stdout, stderr = _run_command(cmd, timeout=timeout)
    combined = stdout + stderr
    return ok, combined, ""


def parse_ncu_output(ncu_output: str) -> dict[str, float]:
    results = {}
    pattern = r"(\S+)\s+(?:\d+[.,]?\d*)\s+(?:%\s+)?(\d+[.,]?\d*)"
    for line in ncu_output.split("\n"):
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-") or line.startswith("=="):
            continue
        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0]
            for part in parts[1:]:
                part = part.replace(",", "").replace("%", "")
                try:
                    results[metric_name] = float(part)
                    break
                except ValueError:
                    continue
    return results


def run_ncu_section(binary_path: str, section: str = "SpeedOfLight", args: str = "", timeout: int = 120) -> tuple[bool, str, str]:
    cmd = f"ncu --section {section} {binary_path} {args} 2>&1"
    ok, stdout, stderr = _run_command(cmd, timeout=timeout)
    combined = stdout + stderr
    return ok, combined, ""


def get_gpu_info() -> dict:
    info = {}
    ok, stdout, _ = _run_command("nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader 2>&1")
    if ok and stdout.strip():
        parts = [p.strip() for p in stdout.strip().split(",")]
        if len(parts) >= 4:
            info["gpu_name"] = parts[0]
            info["memory_total_mb"] = parts[1]
            info["max_sm_clock_mhz"] = parts[2]
            info["max_mem_clock_mhz"] = parts[3]
    ok2, stdout2, _ = _run_command("nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw --format=csv,noheader 2>&1")
    if ok2 and stdout2.strip():
        parts2 = [p.strip() for p in stdout2.strip().split(",")]
        if len(parts2) >= 2:
            info["current_sm_clock_mhz"] = parts2[0]
            info["current_mem_clock_mhz"] = parts2[1]
    return info


def check_environment_tampering() -> dict:
    anomalies = {}
    ok, stdout, _ = _run_command("nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader 2>&1")
    if ok and stdout.strip():
        parts = [p.strip() for p in stdout.strip().split(",")]
        if len(parts) >= 2:
            try:
                current = float(parts[0])
                max_clock = float(parts[1])
                if current < max_clock * 0.7:
                    anomalies["frequency_locked"] = {
                        "current_mhz": current,
                        "max_mhz": max_clock,
                        "note": "GPU frequency appears to be locked below 70% of max"
                    }
            except ValueError:
                pass
    ok2, stdout2, _ = _run_command("cat /proc/driver/nvidia/gpus/0000:*/information 2>&1 | head -20")
    if ok2:
        anomalies["gpu_proc_info"] = stdout2
    ok3, stdout3, _ = _run_command("nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader 2>&1")
    if ok3 and stdout3.strip():
        parts3 = [p.strip().replace("W", "") for p in stdout3.strip().split(",")]
        if len(parts3) >= 2:
            try:
                limit = float(parts3[0])
                default = float(parts3[1])
                if limit < default * 0.8:
                    anomalies["power_limited"] = {
                        "limit_w": limit,
                        "default_w": default,
                        "note": "Power limit appears to be reduced"
                    }
            except ValueError:
                pass
    return anomalies


def cleanup_workspace():
    if WORKSPACE.exists():
        for f in WORKSPACE.iterdir():
            if f.is_file():
                f.unlink()
