"""
Plan Executor for CyberJarvis.
Coordinates security policy enforcement, confirmation prompts, parallel execution of independent read-only diagnostics,
and dynamic multi-step expansion.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable, Dict, Any

from agent.schemas import AgentPlan, ToolResult, PlanStep
from tools.registry import registry
from security.policy import SecurityPolicy, RiskLevel, PermissionLevel
from security.confirmation import ConfirmationHandler
from observability.logging import get_logger
from observability.tracing import trace_span

logger = get_logger("CyberJarvis.Executor")


class PlanExecutor:
    """Executes planned tool steps with parallel optimization for independent read-only diagnostics."""

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
        if not plan.steps:
            return []

        results: List[ToolResult] = []
        executed_tools = set()

        # Separate independent read-only steps from mutating/dependent steps
        readonly_steps: List[PlanStep] = []
        mutating_steps: List[PlanStep] = []

        for step in plan.steps:
            tool = registry.get(step.tool)
            if tool and tool.definition.is_read_only and tool.definition.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
                readonly_steps.append(step)
            else:
                mutating_steps.append(step)

        user_req = getattr(plan, "user_request", None) or plan.goal or plan.intent

        # 1. Parallel execution for multiple independent read-only tools
        if len(readonly_steps) > 1:
            tool_names = [s.tool for s in readonly_steps]
            logger.info(f"Parallel Execution: Running {len(readonly_steps)} diagnostic tools concurrently: {tool_names}")
            if self.step_callback:
                self.step_callback(f"Running parallel diagnostics: {', '.join(tool_names)}")

            def _execute_single(step: PlanStep) -> ToolResult:
                logger.debug(f"[Parallel] Starting: {step.tool}")
                with trace_span(f"Tool:{step.tool}", metadata=step.parameters):
                    return registry.execute(
                        tool_name=step.tool,
                        parameters=step.parameters,
                        policy=self.policy,
                        confirmation_handler=self.confirmation_handler,
                        user_request=user_req,
                    )

            max_workers = min(len(readonly_steps), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_step = {executor.submit(_execute_single, step): step for step in readonly_steps}
                for future in as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        res = future.result()
                        results.append(res)
                        executed_tools.add(step.tool)
                    except Exception as ex:
                        logger.error(f"Parallel tool {step.tool} failed: {ex}")
                        results.append(ToolResult(tool_name=step.tool, success=False, error=str(ex)))

        elif len(readonly_steps) == 1:
            step = readonly_steps[0]
            if self.step_callback:
                self.step_callback(f"Running: {step.tool}")
            with trace_span(f"Tool:{step.tool}", metadata=step.parameters):
                res = registry.execute(
                    tool_name=step.tool,
                    parameters=step.parameters,
                    policy=self.policy,
                    confirmation_handler=self.confirmation_handler,
                    user_request=user_req,
                )
                results.append(res)
                executed_tools.add(step.tool)

        # 2. Sequential execution for mutating / high-risk operations with confirmation
        for step in mutating_steps:
            if self.step_callback:
                self.step_callback(f"Running (confirmed): {step.tool}")
            logger.info(f"[Sequential] Executing mutating step: {step.tool}")
            with trace_span(f"Tool:{step.tool}", metadata=step.parameters):
                res = registry.execute(
                    tool_name=step.tool,
                    parameters=step.parameters,
                    policy=self.policy,
                    confirmation_handler=self.confirmation_handler,
                    user_request=user_req,
                )
                results.append(res)
                executed_tools.add(step.tool)

        # 3. Dynamic multi-step expansion:
        # If memory or CPU is high from diagnostic metrics, and top processes wasn't checked, add inspection
        for r in results:
            if r.success and isinstance(r.output, dict):
                mem_pct = r.output.get("percent") or (
                    r.output.get("memory", {}).get("percent")
                    if isinstance(r.output.get("memory"), dict)
                    else 0
                )
                if (
                    mem_pct
                    and mem_pct >= 75.0
                    and "get_top_processes" not in executed_tools
                    and "diagnose_system_bottleneck" not in executed_tools
                ):
                    logger.info(f"Dynamic reasoning: High memory ({mem_pct}%) detected. Adding top processes inspection.")
                    if self.step_callback:
                        self.step_callback("Inspecting top memory processes...")
                    proc_res = registry.execute(
                        tool_name="get_top_processes",
                        parameters={"limit": 3, "sort_by": "memory"},
                        policy=self.policy,
                        confirmation_handler=self.confirmation_handler,
                    )
                    results.append(proc_res)
                    executed_tools.add("get_top_processes")
                    break

        return results
