from typing import List, Optional, Callable
from agent.schemas import AgentPlan, ToolResult, PlanStep
from tools.registry import registry
from security.policy import SecurityPolicy
from security.confirmation import ConfirmationHandler
from observability.logging import get_logger
from observability.tracing import trace_span

logger = get_logger("CyberJarvis.Executor")


class PlanExecutor:
    """Executes planned tool steps, checks security guardrails, and records observations."""

    def __init__(
        self,
        policy: Optional[SecurityPolicy] = None,
        confirmation_handler: Optional[ConfirmationHandler] = None,
        step_callback: Optional[Callable[[str], None]] = None,
    ):
        self.policy = policy or SecurityPolicy()
        self.confirmation_handler = confirmation_handler or ConfirmationHandler()
        self.step_callback = step_callback

    def execute_plan(self, plan: AgentPlan) -> List[ToolResult]:
        results: List[ToolResult] = []
        executed_tools = set()

        for step in plan.steps:
            if self.step_callback:
                self.step_callback(step.tool)

            logger.info(f"Executing step: {step.tool} with params: {step.parameters}")

            with trace_span(f"Tool:{step.tool}", metadata=step.parameters):
                result = registry.execute(
                    tool_name=step.tool,
                    parameters=step.parameters,
                    policy=self.policy,
                    confirmation_handler=self.confirmation_handler,
                )

            results.append(result)
            executed_tools.add(step.tool)

            # Dynamic multi-step expansion:
            # If memory or CPU is high from get_system_status/get_memory_usage, and get_top_processes wasn't planned,
            # dynamically inspect the processes!
            if result.success and isinstance(result.output, dict):
                mem_pct = result.output.get("percent") or (result.output.get("memory", {}).get("percent") if isinstance(result.output.get("memory"), dict) else 0)
                if mem_pct and mem_pct >= 75.0 and "get_top_processes" not in executed_tools and "diagnose_system_bottleneck" not in executed_tools:
                    logger.info(f"Dynamic reasoning: High memory ({mem_pct}%) detected. Adding top processes inspection.")
                    if self.step_callback:
                        self.step_callback("get_top_processes")
                    proc_res = registry.execute(
                        tool_name="get_top_processes",
                        parameters={"limit": 3, "sort_by": "memory"},
                        policy=self.policy,
                        confirmation_handler=self.confirmation_handler,
                    )
                    results.append(proc_res)
                    executed_tools.add("get_top_processes")

        return results
