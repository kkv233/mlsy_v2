# GPU Hardware Probing Multi-Agent System 实现计划

## 架构设计

```
┌─────────────────────────────────────────────┐
│              main.py (入口)                  │
│  读取 target_spec.json → Orchestrator       │
│  输出 results.json                          │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│         Orchestrator Agent                   │
│  - 解析探测目标                              │
│  - 分配给 Specialist Agents                  │
│  - 汇总结果、交叉验证                        │
└────┬───┬───┬───┬───┬───┬────────────────────┘
     │   │   │   │   │   │
     ▼   ▼   ▼   ▼   ▼   ▼
   ┌─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┐
   │ML │BW │L2 │BF │RP │NCU│  (6个Specialist Agents)
   └───┴───┴───┴───┴───┴───┘
```

## 核心原则（严格遵守）

1. **Agent自主生成CUDA代码**：提示词中只提供设计思路，不提供源代码模板。Agent根据思路自行编写CUDA C++代码
2. **禁止硬编码/查表**：所有数值必须通过实际微基准测试获得
3. **API Key通过环境变量读取**：`os.environ["API_KEY"]`
4. **Multi-Strategy Fusion**：结合微基准测试 + ncu分析 + 交叉验证

## 文件结构

```
mlsy_v2/
├── main.py                    # 入口：读取target_spec.json，运行Orchestrator，输出results.json
├── core/
│   ├── __init__.py
│   ├── llm.py                 # LLM调用封装（OpenAI兼容接口）
│   ├── tools.py               # 工具定义：编译、执行、ncu分析
│   └── agent_base.py          # Agent基类
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py        # 编排Agent
│   ├── memory_latency.py      # Memory Latency探测
│   ├── bandwidth.py           # 带宽探测
│   ├── l2_capacity.py         # L2 Cache容量探测
│   ├── boost_frequency.py     # 实际频率探测
│   ├── resource_penalty.py    # Bank Conflict等资源惩罚
│   └── ncu_profiler.py        # ncu瓶颈分析
├── workspace/                 # Agent生成代码的工作目录
└── requirement.txt
```

## 各Specialist Agent设计思路

### 1. Memory Latency Agent

* **设计思路**：生成Pointer Chasing kernel，通过链表指针追踪绕过预取器，测量不同stride下的访问延迟，区分L1/L2/DRAM延迟

* **交叉验证**：用ncu的`l1tex__t_sectors`和`l2__throughput`指标验证

### 2. Bandwidth Agent

* **设计思路**：生成大块连续读/写kernel，分别测试Shared Memory和Global Memory的吞吐上限

* **交叉验证**：用ncu的`dram__throughput`和`l1tex__data_pipe_lsu_wavefronts`验证

### 3. L2 Capacity Agent

* **设计思路**：生成不同大小数组的随机访问kernel，绘制latency-vs-size曲线，找到延迟跳变的"悬崖"点

* **交叉验证**：与`cudaDeviceProp::l2CacheSize`对比（注意API可能被篡改）

### 4. Boost Frequency Agent

* **设计思路**：生成持续满载计算kernel，通过CUDA event计时+已知FLOPS反推实际频率；同时读取`/proc/driver/nvidia/gpus/`信息

* **交叉验证**：nvidia-smi查询 + ncu的`sm__clock_cycle`指标

### 5. Resource Penalty Agent

* **设计思路**：生成有/无bank conflict的Shared Memory访问kernel，对比两者延迟差

* **交叉验证**：ncu的`l1tex__data_bank_conflicts_pipe_lsu.sum`指标

### 6. NCU Profiler Agent

* **设计思路**：对给定可执行文件运行ncu，收集1.1-1.6中所有关键指标，按Roofline→Memory/Compute→Occupancy→异常的流程分析

## 实现步骤

1. **核心基础设施**：LLM调用、工具系统（编译/执行/ncu）、Agent基类
2. **Orchestrator**：解析target\_spec.json，调度Specialist Agents
3. **6个Specialist Agents**：每个Agent包含设计思路提示词 + 工具调用 + 迭代优化逻辑
4. **结果汇总**：交叉验证、输出results.json
5. **端到端测试**

