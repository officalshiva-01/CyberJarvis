"""
Tier 2 Diagnostic Path Handler for CyberJarvis.
Coordinates:
- Predefined diagnostic plan for standard system diagnostics
- Parallel tool execution across independent read-only telemetry tools
- Measurement of sequential vs parallel wall-clock execution time
- Deterministic diagnostic analysis with zero LLM calls
- Direct generation of grounded Markdown responses and concise spoken summaries
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Callable

from agent.schemas import AgentResponse, AgentPlan, PlanStep
from agent.diagnostic_analyzer import DiagnosticAnalyzer, DiagnosticReport
from tools.schemas import ToolResult
from tools.registry import registry
from security.policy import SecurityPolicy
from security.confirmation import ConfirmationHandler
from observability.logging import get_logger

logger = get_logger("CyberJarvis.DiagnosticPath")


class DiagnosticPathHandler:
    """Executes Tier 2 system diagnostics deterministically with parallel tool execution."""

    def __init__(
        self,
        analyzer: Optional[DiagnosticAnalyzer] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self.analyzer = analyzer or DiagnosticAnalyzer()
        self.on_status = on_status_change or (lambda s: None)
        self.policy = SecurityPolicy()
        self.confirmation_handler = ConfirmationHandler()

    def handle(self, user_input: str) -> Tuple[AgentResponse, float, float, float]:
        """
        Execute diagnostic query.
        Returns:
            (response, parallel_tools_ms, sequential_tools_ms, analysis_ms)
        """
        clean_input = user_input.strip()
        user_lower = clean_input.lower()

        # 1. Determine tool steps based on inquiry
        tool_specs = self._select_diagnostic_tools(user_lower)
        logger.info(f"Diagnostic Path: Running {len(tool_specs)} independent tools in parallel: {[t[0] for t in tool_specs]}")
        self.on_status(f"🔧 Running parallel diagnostics ({len(tool_specs)} checks)...")

        # 2. Parallel Execution
        results, parallel_ms, sequential_ms = self._execute_tools_parallel(tool_specs)

        # 3. Deterministic Diagnostic Analysis (Zero LLM)
        self.on_status("📊 Analyzing telemetry deterministically...")
        analysis_start = time.perf_counter()
        report, display_markdown = self.analyzer.analyze(results, clean_input)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000

        self.on_status("✅ Diagnosis complete")

        # 4. Build Plan representation
        plan_steps = [
            PlanStep(tool=tool_name, parameters=params, explanation="Diagnostic telemetry probe")
            for tool_name, params in tool_specs
        ]
        intent = "investigate_processes" if any(w in user_lower for w in ["consuming", "using", "taking", "process"]) else "system_diagnosis"
        plan = AgentPlan(
            intent=intent,
            goal=clean_input,
            steps=plan_steps,
        )

        response = AgentResponse(
            answer=display_markdown,
            spoken_summary=report.spoken_summary,
            plan=plan,
            tool_results=results,
            completed=True,
            planning_ms=0.0,
            evaluator_ms=0.0,
            diagnostic_report=report,
        )

        logger.info(
            f"Diagnostic Path completed in {parallel_ms + analysis_ms:.1f}ms "
            f"(tools_parallel={parallel_ms:.1f}ms, tools_sequential={sequential_ms:.1f}ms, "
            f"saved={sequential_ms - parallel_ms:.1f}ms, analysis={analysis_ms:.1f}ms)"
        )

        return response, parallel_ms, sequential_ms, analysis_ms

    def _select_diagnostic_tools(self, user_lower: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Select predefined diagnostic plan based on query specificity."""
        # Network specific query
        if any(w in user_lower for w in ["check my network", "check network", "mera internet check karo", "network check"]):
            return [
                ("check_network_connectivity", {}),
                ("check_dns_resolution", {"domain": "github.com"}),
            ]

        # Top memory query
        if any(w in user_lower for w in [
            "consuming the most memory", "most memory", "consuming memory", "using memory",
            "which process is consuming", "what is using the most memory", "what is taking memory",
            "taking memory", "ram kaun use kar raha hai", "kaun memory use kar raha hai",
            "kaun ram use kar raha hai"
        ]):
            if "cpu" not in user_lower:
                return [
                    ("get_top_processes", {"limit": 5, "sort_by": "memory"}),
                    ("get_memory_usage", {}),
                ]

        # Top CPU query
        if any(w in user_lower for w in ["consuming the most cpu", "most cpu", "consuming cpu", "using cpu", "top cpu"]):
            if "memory" not in user_lower and "ram" not in user_lower:
                return [
                    ("get_top_processes", {"limit": 5, "sort_by": "cpu"}),
                    ("get_cpu_usage", {}),
                ]

        # Standard comprehensive diagnosis
        return [
            ("get_cpu_usage", {}),
            ("get_memory_usage", {}),
            ("get_swap_usage", {}),
            ("get_disk_usage", {}),
            ("get_load_average", {}),
            ("get_top_processes", {"limit": 5, "sort_by": "memory"}),
            ("get_top_processes", {"limit": 5, "sort_by": "cpu"}),
            ("check_network_connectivity", {}),
            ("check_dns_resolution", {"domain": "github.com"}),
        ]

    def _execute_tools_parallel(
        self, tool_specs: List[Tuple[str, Dict[str, Any]]]
    ) -> Tuple[List[ToolResult], float, float]:
        """Execute independent read-only diagnostic tools concurrently."""
        results: List[ToolResult] = []

        def _run_single(tool_name: str, params: Dict[str, Any]) -> ToolResult:
            return registry.execute(
                tool_name=tool_name,
                parameters=params,
                policy=self.policy,
                confirmation_handler=self.confirmation_handler,
            )

        start_wall = time.perf_counter()
        max_workers = min(len(tool_specs), 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_single, name, params) for name, params in tool_specs]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    results.append(res)
                except Exception as ex:
                    logger.error(f"Parallel diagnostic tool failed: {ex}")
                    results.append(ToolResult(tool_name="unknown", success=False, error=str(ex)))

        parallel_ms = (time.perf_counter() - start_wall) * 1000
        sequential_ms = sum(r.duration_ms for r in results)

        return results, round(parallel_ms, 2), round(sequential_ms, 2)
