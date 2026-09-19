"""
Agent Gateway for CyberJarvis.
Unified entry point for Terminal UI, Voice UI, and future Web UI.
Coordinates:
- Transcript normalization (wake word removal, whitespace, lowercase)
- Single authoritative entry point: process_user_request()
- Three-tier execution architecture:
    * Tier 1: Fast Path (Deterministic facts, time, date, single metrics)
    * Tier 2: Diagnostic Path (Predefined parallel plan + deterministic analyzer, zero LLM)
    * Tier 3: True Agent Path (Ollama LLM Planner -> Parallel Executor -> Grounded Evaluator)
- End-to-end granular latency measurement across all pipeline stages
"""
import time
from typing import Optional, Callable, Dict, Any
from pydantic import BaseModel, Field

from agent.schemas import AgentResponse, AgentPlan
from agent.router import ExecutionRouter, RouteType, normalize_transcript, is_incomplete_transcript
from agent.fast_path import FastPathHandler
from agent.diagnostic_path import DiagnosticPathHandler
from agent.information_path import InformationPathHandler
from agent.agent_core import AgentCore
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Gateway")


class LatencyBreakdown(BaseModel):
    mic_capture_ms: float = 0.0
    vad_ms: float = 0.0
    stt_ms: float = 0.0
    normalization_ms: float = 0.0
    router_ms: float = 0.0
    fast_path_ms: float = 0.0
    planning_ms: float = 0.0
    tools_ms: float = 0.0
    sequential_tools_ms: float = 0.0
    parallel_tools_ms: float = 0.0
    analysis_ms: float = 0.0
    tool_breakdown: Dict[str, float] = Field(default_factory=dict)
    evaluator_ms: float = 0.0
    evaluation_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    execution_path: str = "FAST"

    def get_largest_source(self) -> tuple[str, float]:
        """Identify the single largest source of latency."""
        stages = [
            ("STT", self.stt_ms / 1000.0),
            ("Planner", self.planning_ms / 1000.0),
            ("Tools", self.tools_ms / 1000.0),
            ("Evaluator", self.evaluator_ms / 1000.0),
            ("TTS", self.tts_ms / 1000.0),
        ]
        if self.fast_path_ms > self.planning_ms:
            stages.append(("Fast-Path", self.fast_path_ms / 1000.0))
        if self.analysis_ms > 0 and (self.analysis_ms / 1000.0) > (self.planning_ms / 1000.0):
            stages.append(("Analysis", self.analysis_ms / 1000.0))
        return max(stages, key=lambda s: s[1])

    def format_timing_report(self) -> str:
        stt_s = f"{self.stt_ms / 1000.0:.1f}s"
        router_s = f"{self.router_ms / 1000.0:.1f}s" if self.router_ms >= 100 else (f"{self.router_ms:.1f}ms" if self.router_ms > 0 else "0.0s")
        planner_s = f"{self.planning_ms / 1000.0:.1f}s" if self.planning_ms > 0 else "SKIPPED"
        tools_s = f"{self.tools_ms / 1000.0:.1f}s" if self.tools_ms > 0 else "0.0s"
        analysis_s = f"{self.analysis_ms / 1000.0:.1f}s" if self.analysis_ms >= 100 else (f"{self.analysis_ms:.1f}ms" if self.analysis_ms > 0 else ("0.0s" if self.planning_ms == 0 else "SKIPPED"))
        evaluator_s = f"{self.evaluator_ms / 1000.0:.1f}s" if self.evaluator_ms > 0 else "SKIPPED"
        tts_s = f"{self.tts_ms / 1000.0:.1f}s"
        total_s = f"{self.total_ms / 1000.0:.1f}s"

        largest_name, largest_sec = self.get_largest_source()

        return (
            f"STT: {stt_s}\n"
            f"Router: {router_s}\n"
            f"Planner: {planner_s}\n"
            f"Tools: {tools_s}\n"
            f"Analysis: {analysis_s}\n"
            f"Evaluator: {evaluator_s}\n"
            f"TTS: {tts_s}\n"
            f"Total: {total_s}\n\n"
            f"Largest latency source: {largest_name} ({largest_sec:.1f}s)"
        )


class GatewayResult(BaseModel):
    response: AgentResponse
    route: str  # "fast_path", "diagnostic_path", "information_path", "agent_path"
    latency: LatencyBreakdown
    source: str = "terminal"
    execution_path: str = "FAST"  # "FAST", "DIAGNOSTIC", "INFORMATION", "AGENT", "CLARIFICATION"


class AgentGateway:
    """Unified API Gateway decoupling UI interfaces from core agent and tool logic."""

    def __init__(
        self,
        agent_core: Optional[AgentCore] = None,
        fast_path: Optional[FastPathHandler] = None,
        diagnostic_path: Optional[DiagnosticPathHandler] = None,
        information_path: Optional[InformationPathHandler] = None,
        router: Optional[ExecutionRouter] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self.on_status = on_status_change or (lambda s: None)
        self.router = router or ExecutionRouter()
        self.fast_path = fast_path or FastPathHandler()
        self.diagnostic_path = diagnostic_path or DiagnosticPathHandler(on_status_change=self.on_status)
        self.information_path = information_path or InformationPathHandler()
        self.agent_core = agent_core or AgentCore(on_status_change=self.on_status)

    def process_user_request(
        self,
        user_input: str,
        source: str = "terminal",
        stt_latency_ms: float = 0.0,
        mic_capture_ms: float = 0.0,
        vad_ms: float = 0.0,
    ) -> GatewayResult:
        """
        Authoritative execution entry point for all interfaces (Terminal, Voice, API).
        Coordinates transcript normalization, three-tier routing, and granular latency measurement.
        """
        start_time = time.perf_counter()

        # 0a. Normalize input (lowercase, trim, collapse whitespace, strip harmless wake words & conjunctions)
        norm_start = time.perf_counter()
        clean_input = normalize_transcript(user_input)
        if not clean_input:
            clean_input = user_input.strip()
        norm_ms = (time.perf_counter() - norm_start) * 1000

        logger.info(f"Gateway process_user_request [{source}]: raw='{user_input}' -> normalized='{clean_input}'")
        breakdown = LatencyBreakdown(
            stt_ms=round(stt_latency_ms, 2),
            mic_capture_ms=round(mic_capture_ms, 2),
            vad_ms=round(vad_ms, 2),
            normalization_ms=round(norm_ms, 2),
        )

        # 0b. Route Classification (Deterministic & executed BEFORE any LLM call)
        router_start = time.perf_counter()
        route_decision = self.router.classify(clean_input)
        router_ms = (time.perf_counter() - router_start) * 1000
        breakdown.router_ms = round(router_ms, 2)

        # 0c. Incomplete Transcript Guard (Section 3, 5, 7, 8, 9 - Zero LLM call)
        if (
            route_decision == RouteType.CLARIFICATION
            or is_incomplete_transcript(clean_input)
            or is_incomplete_transcript(user_input)
        ):
            logger.info(f"Incomplete command detected: '{user_input}' -> '{clean_input}'")
            clarification_response = AgentResponse(
                answer="🤔 I didn't catch the full command.\n🎤 Please repeat.",
                spoken_summary="I didn't catch the full command. Please repeat.",
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
                plan=AgentPlan(intent="clarification", goal="Request user repeat incomplete command", steps=[]),
            )
            total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
            breakdown.total_ms = round(total_duration, 2)
            breakdown.execution_path = "CLARIFICATION"
            breakdown.planning_ms = 0.0
            breakdown.evaluator_ms = 0.0
            breakdown.tools_ms = 0.0
            return GatewayResult(
                response=clarification_response,
                route="clarification",
                latency=breakdown,
                source=source,
                execution_path="CLARIFICATION",
            )

        # 1. Tier 1 — Fast Path Check
        if route_decision == RouteType.FAST:
            fast_start = time.perf_counter()
            fast_response = self.fast_path.try_handle(clean_input)
            fast_duration = (time.perf_counter() - fast_start) * 1000
            breakdown.fast_path_ms = round(fast_duration, 2)

            if fast_response:
                total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
                breakdown.total_ms = round(total_duration, 2)
                tools_duration = sum(r.duration_ms for r in fast_response.tool_results)
                breakdown.tools_ms = round(tools_duration, 2)
                breakdown.tool_breakdown = {r.tool_name: r.duration_ms for r in fast_response.tool_results}
                breakdown.execution_path = "FAST"
                logger.info(f"Command handled via Fast Path in {breakdown.total_ms}ms")
                return GatewayResult(
                    response=fast_response,
                    route="fast_path",
                    latency=breakdown,
                    source=source,
                    execution_path="FAST",
                )

        # 2. Tier 2 — Diagnostic Path Check (Predefined plan, parallel execution, zero LLM)
        if route_decision == RouteType.DIAGNOSTIC:
            logger.info("Routing command to Tier 2 Diagnostic Path...")
            diag_response, parallel_ms, seq_ms, analysis_ms = self.diagnostic_path.handle(clean_input)

            total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
            breakdown.total_ms = round(total_duration, 2)
            breakdown.tools_ms = round(parallel_ms, 2)
            breakdown.parallel_tools_ms = round(parallel_ms, 2)
            breakdown.sequential_tools_ms = round(seq_ms, 2)
            breakdown.analysis_ms = round(analysis_ms, 2)
            breakdown.planning_ms = 0.0
            breakdown.evaluator_ms = 0.0
            breakdown.tool_breakdown = {r.tool_name: r.duration_ms for r in diag_response.tool_results}
            breakdown.execution_path = "DIAGNOSTIC"

            logger.info(f"Command handled via Diagnostic Path in {breakdown.total_ms}ms (parallel_tools={parallel_ms}ms)")
            return GatewayResult(
                response=diag_response,
                route="diagnostic_path",
                latency=breakdown,
                source=source,
                execution_path="DIAGNOSTIC",
            )

        # 3. Tier 3b — Information / Direct LLM Path (Single focused response, 0 tools, 0 planner, 0 evaluator)
        if route_decision == RouteType.INFORMATION:
            logger.info("Routing command to Tier 3b Information Path...")
            info_response, llm_ms = self.information_path.handle(clean_input)

            total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
            breakdown.total_ms = round(total_duration, 2)
            breakdown.tools_ms = 0.0
            breakdown.planning_ms = 0.0
            breakdown.evaluator_ms = 0.0
            breakdown.analysis_ms = round(llm_ms, 2)
            breakdown.execution_path = "INFORMATION"

            logger.info(f"Command handled via Information Path in {breakdown.total_ms}ms (llm={llm_ms:.1f}ms)")
            return GatewayResult(
                response=info_response,
                route="information_path",
                latency=breakdown,
                source=source,
                execution_path="INFORMATION",
            )

        # 4. Tier 3a — True Agent Path (Ollama LLM Planner -> Parallel Executor -> Grounded Evaluator)
        logger.info("Routing command to Tier 3 True Agent Path...")
        agent_response = self.agent_core.process(clean_input)

        total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
        breakdown.total_ms = round(total_duration, 2)

        breakdown.planning_ms = round(agent_response.planning_ms, 2)
        breakdown.evaluator_ms = round(agent_response.evaluator_ms, 2)
        breakdown.evaluation_ms = round(agent_response.evaluator_ms, 2)

        tools_duration = sum(r.duration_ms for r in agent_response.tool_results)
        breakdown.tools_ms = round(tools_duration, 2)
        breakdown.tool_breakdown = {r.tool_name: r.duration_ms for r in agent_response.tool_results}
        breakdown.execution_path = "AGENT"

        return GatewayResult(
            response=agent_response,
            route="agent_path",
            latency=breakdown,
            source=source,
            execution_path="AGENT",
        )

    def process_command(
        self,
        user_input: str,
        source: str = "terminal",
        stt_latency_ms: float = 0.0,
        mic_capture_ms: float = 0.0,
        vad_ms: float = 0.0,
    ) -> GatewayResult:
        """Backward-compatible alias for process_user_request."""
        return self.process_user_request(
            user_input=user_input,
            source=source,
            stt_latency_ms=stt_latency_ms,
            mic_capture_ms=mic_capture_ms,
            vad_ms=vad_ms,
        )


_default_gateway: Optional[AgentGateway] = None


def get_default_gateway() -> AgentGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = AgentGateway()
    return _default_gateway


def process_user_request(
    text: str,
    source: str = "terminal",
    gateway: Optional[AgentGateway] = None,
    **kwargs,
) -> GatewayResult:
    """Module-level authoritative execution entry point."""
    gw = gateway or get_default_gateway()
    return gw.process_user_request(user_input=text, source=source, **kwargs)
