import json
from core.llm import LLMClient
from core.methodology import match_methodology, METHODOLOGY_KB
from core.agent_base import ProbeTask


class PlannerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, target_spec: dict) -> list[ProbeTask]:
        targets = target_spec.get("targets", [])
        run_executable = target_spec.get("run", None)
        tasks = []
        for target in targets:
            methodology = match_methodology(target)
            if methodology["matched_category"] == "unknown":
                methodology = self._infer_methodology(target)
            dependencies = self._infer_dependencies(target, targets)
            context = {}
            if run_executable:
                context["run_executable"] = run_executable
                context["has_reference_program"] = True
            tasks.append(ProbeTask(
                target=target,
                methodology=methodology,
                dependencies=dependencies,
                context=context,
            ))
        tasks = self._topological_sort(tasks)
        return tasks

    def _infer_methodology(self, target: str) -> dict:
        prompt = f"""Analyze this GPU hardware metric name and provide a measurement methodology.

Target metric: {target}

Available methodology categories: {list(METHODOLOGY_KB.keys())}

Respond in JSON format:
{{
    "matched_category": "best matching category or 'unknown'",
    "principle": "What this metric measures and the general approach to measure it",
    "key_challenges": ["list of challenges"],
    "approach": "Step-by-step approach to write a CUDA micro-benchmark for this metric",
    "ncu_metrics": ["list of relevant ncu metrics for validation"],
    "validation": "How to validate the measurement using ncu"
}}"""
        response = self.llm.chat_with_system(
            system_prompt="You are a GPU hardware expert. Respond only in JSON.",
            user_prompt=prompt,
            temperature=0.2,
        )
        result = self.llm.extract_json(response)
        if not result:
            return match_methodology(target)
        return result

    def _infer_dependencies(self, target: str, all_targets: list[str]) -> list[str]:
        deps = []
        target_lower = target.lower()
        if "bandwidth" in target_lower and "vram" in target_lower:
            if any("clock" in t.lower() or "frequency" in t.lower() or "mhz" in t.lower() for t in all_targets):
                for t in all_targets:
                    if "clock" in t.lower() or "frequency" in t.lower() or "mhz" in t.lower():
                        deps.append(t)
        if "l2_cache_size" in target_lower or "l2_capacity" in target_lower:
            if any("latency" in t.lower() for t in all_targets):
                for t in all_targets:
                    if "dram_latency" in t.lower() or "l2_latency" in t.lower():
                        deps.append(t)
        return deps

    def _topological_sort(self, tasks: list[ProbeTask]) -> list[ProbeTask]:
        task_map = {t.target: t for t in tasks}
        visited = set()
        result = []
        visiting = set()

        def visit(target: str):
            if target in visited:
                return
            if target in visiting:
                return
            visiting.add(target)
            if target in task_map:
                for dep in task_map[target].dependencies:
                    if dep in task_map:
                        visit(dep)
            visiting.discard(target)
            visited.add(target)
            result.append(task_map[target])

        for task in tasks:
            visit(task.target)
        return result
