import subprocess
import os
import sys
import time

WORKSPACE = "/root/mlsy_v2/workspace"

def run_cmd(cmd, timeout=60):
    print(f"\n>>> Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout[:2000]}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr[:2000]}")
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT!")
        return -1, "", "timeout"

def compile_test_kernel():
    src = """
#include <stdio.h>
__global__ void test_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = val * 1.00001f + 0.00001f;
        }
        data[idx] = val;
    }
}
int main() {
    int n = 1024 * 1024;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemset(d_data, 0, n * sizeof(float));
    test_kernel<<<256, 256>>>(d_data, n);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    printf("0\\n");
    return 0;
}
"""
    src_path = os.path.join(WORKSPACE, "ncu_test_kernel.cu")
    bin_path = os.path.join(WORKSPACE, "ncu_test_kernel")
    with open(src_path, "w") as f:
        f.write(src)
    rc, out, err = run_cmd(f"nvcc -arch=sm_86 -o {bin_path} {src_path} 2>&1")
    if rc == 0:
        print(f"Compilation OK, binary: {bin_path}")
        return bin_path
    else:
        print("Compilation FAILED!")
        return None

def test_ncu_methods(binary_path):
    methods = [
        {
            "name": "Method 1: Basic ncu with metrics",
            "cmd": f"ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 2: ncu with --target-processes all",
            "cmd": f"ncu --target-processes all --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 3: ncu with --section SpeedOfLight",
            "cmd": f"ncu --section SpeedOfLight {binary_path} 2>&1",
        },
        {
            "name": "Method 4: ncu with --set full",
            "cmd": f"ncu --set full {binary_path} 2>&1",
        },
        {
            "name": "Method 5: ncu with --set baseline",
            "cmd": f"ncu --set baseline {binary_path} 2>&1",
        },
        {
            "name": "Method 6: ncu with --page raw",
            "cmd": f"ncu --page raw --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 7: ncu with --print-summary",
            "cmd": f"ncu --print-summary --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 8: ncu with sudo",
            "cmd": f"sudo ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 9: ncu with --force-overwrite",
            "cmd": f"ncu --force-overwrite --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 10: ncu with -o report",
            "cmd": f"ncu -o /tmp/ncu_report --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 11: ncu with --clock-control none",
            "cmd": f"ncu --clock-control none --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 12: ncu with --cache-control none",
            "cmd": f"ncu --cache-control none --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 13: ncu with both --cache-control and --clock-control none",
            "cmd": f"ncu --cache-control none --clock-control none --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 14: ncu with --launch-skip and --launch-count",
            "cmd": f"ncu --launch-skip 0 --launch-count 1 --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
        {
            "name": "Method 15: ncu with --target-processes all and sudo",
            "cmd": f"sudo ncu --target-processes all --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed {binary_path} 2>&1",
        },
    ]

    success_methods = []
    for method in methods:
        rc, out, err = run_cmd(method["cmd"], timeout=120)
        has_err_nvgpuctrperm = "ERR_NVGPUCTRPERM" in out or "ERR_NVGPUCTRPERM" in err
        has_profiled = "profiled" in out.lower() or "profiled" in err.lower()
        has_metrics = any(c.isdigit() for c in out if c != '\n')
        if rc == 0 and not has_err_nvgpuctrperm:
            print(f"  *** SUCCESS: {method['name']} ***")
            success_methods.append(method["name"])
        elif has_err_nvgpuctrperm:
            print(f"  FAILED: ERR_NVGPUCTRPERM")
        else:
            print(f"  FAILED: other error (rc={rc})")

    print("\n\n=== SUMMARY ===")
    if success_methods:
        print("Working methods:")
        for m in success_methods:
            print(f"  - {m}")
    else:
        print("No working methods found!")

def test_env_modifications():
    print("\n\n=== Testing environment modifications ===")

    print("\n1. Check current RmProfilingAdminOnly:")
    run_cmd("cat /proc/driver/nvidia/params | grep Profiling")

    print("\n2. Try setting RmProfilingAdminOnly=0 via params file:")
    run_cmd("echo 'RmProfilingAdminOnly 0' > /proc/driver/nvidia/params 2>&1")
    run_cmd("cat /proc/driver/nvidia/params | grep Profiling")

    print("\n3. Try creating modprobe.d config:")
    run_cmd("echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf 2>&1")
    run_cmd("cat /etc/modprobe.d/nvidia-profiling.conf 2>&1")

    print("\n4. Check capabilities:")
    run_cmd("capsh --print 2>&1 | grep -i sys_admin")

    print("\n5. Try setcap on ncu binary:")
    run_cmd("setcap cap_sys_admin+ep /usr/local/cuda/bin/ncu 2>&1")
    run_cmd("getcap /usr/local/cuda/bin/ncu 2>&1")

    print("\n6. Try setcap on the test binary:")
    binary_path = os.path.join(WORKSPACE, "ncu_test_kernel")
    if os.path.exists(binary_path):
        run_cmd(f"setcap cap_sys_admin+ep {binary_path} 2>&1")
        run_cmd(f"getcap {binary_path} 2>&1")

def test_cupti_python():
    print("\n\n=== Testing CUPTI Python approach ===")
    try:
        cupti_code = """
import sys
try:
    from cuda import cudart
    print("cuda.cudart available")
except ImportError:
    print("cuda.cudart NOT available")

try:
    import pycupti
    print("pycupti available")
except ImportError:
    print("pycupti NOT available")

try:
    from PyCUPTI import CUPTI
    print("PyCUPTI available")
except ImportError:
    print("PyCUPTI NOT available")
"""
        run_cmd(f"python3 -c '{cupti_code}'")
    except:
        pass

if __name__ == "__main__":
    print("=" * 60)
    print("NCU Permission Test Script")
    print("=" * 60)

    binary_path = compile_test_kernel()
    if binary_path:
        test_env_modifications()
        test_ncu_methods(binary_path)
        test_cupti_python()
    else:
        print("Cannot compile test kernel, aborting!")