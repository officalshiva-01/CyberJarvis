"""
Agent Gateway for CyberJarvis.
Serves as the unified entry point for all interfaces (Terminal UI, Voice UI, and future Web UI).
Coordinates:
- Input adaptation
- Fast Path routing for deterministic commands
- Agent Path routing for reasoning & multi-tool workflows
- Granular latency measurement across all pipeline stages
- Separation of UI from Agent logic
"""
import time
from typing import Optional, Callable, Dict, Any
from pydantic import BaseModel, Field

from agent.schemas import AgentResponse, AgentPlan
from agent.fast_path import FastPathHandler
from agent.agent_core import AgentCore
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Gateway")


class LatencyBreakdown(BaseModel):
    mic_capture_ms: float = 0.0
    vad_ms: float = 0.0
    stt_ms: float = 0.0
    fast_path_ms: float = 0.0
    planning_ms: float = 0.0
    tools_ms: float = 0.0
    tool_breakdown: Dict[str, float] = Field(default_factory=dict)
    evaluator_ms: float = 0.0
    evaluation_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0

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
        return max(stages, key=lambda s: s[1])

    def format_timing_report(self) -> str:
        stt_s = self.stt_ms / 1000.0
        planner_s = self.planning_ms / 1000.0
        tools_s = self.tools_ms / 1000.0
        evaluator_s = self.evaluator_ms / 1000.0
        tts_s = self.tts_ms / 1000.0
        total_s = self.total_ms / 1000.0

        largest_name, largest_sec = self.get_largest_source()

        return (
            f"STT: {stt_s:.1f}s\n"
            f"Planner: {planner_s:.1f}s\n"
            f"Tools: {tools_s:.1f}s\n"
            f"Evaluator: {evaluator_s:.1f}s\n"
            f"TTS: {tts_s:.1f}s\n"
            f"Total: {total_s:.1f}s\n\n"
            f"Largest latency source: {largest_name} ({largest_sec:.1f}s)"
        )


class GatewayResult(BaseModel):
    response: AgentResponse
    route: str  # "fast_path" or "agent_path"
    latency: LatencyBreakdown
    source: str = "terminal"


class AgentGateway:
    """Unified API Gateway decoupling UI interfaces from core agent and tool logic."""

    def __init__(
        self,
        agent_core: Optional[AgentCore] = None,
        fast_path: Optional[FastPathHandler] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self.fast_path = fast_path or FastPathHandler()
        self.on_status = on_status_change or (lambda s: None)
        self.agent_core = agent_core or AgentCore(on_status_change=self.on_status)

    def process_command(
        self,
        user_input: str,
        source: str = "terminal",
        stt_latency_ms: float = 0.0,
        mic_capture_ms: float = 0.0,
        vad_ms: float = 0.0,
    ) -> GatewayResult:
        """
        Process a user instruction from any interface with end-to-end timing.
        """
        start_time = time.perf_counter()
        clean_input = user_input.strip()

        logger.info(f"Gateway received command from [{source}]: '{clean_input}'")
        breakdown = LatencyBreakdown(
            stt_ms=round(stt_latency_ms, 2),
            mic_capture_ms=round(mic_capture_ms, 2),
            vad_ms=round(vad_ms, 2),
        )

        # 1. Fast Path Check (Deterministic facts, time, date, files, simple git/connectivity)
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
            logger.info(f"Command handled via Fast Path in {breakdown.total_ms}ms")
            return GatewayResult(
                response=fast_response,
                route="fast_path",
                latency=breakdown,
                source=source,
            )

        # 2. Agent Path (LLM Planner -> Parallel Executor -> Grounded Evaluator)
        logger.info("Routing command to Agent Path...")
        agent_response = self.agent_core.process(clean_input)

        total_duration = breakdown.stt_ms + (time.perf_counter() - start_time) * 1000
        breakdown.total_ms = round(total_duration, 2)

        # Latency assignments from agent response
        breakdown.planning_ms = round(agent_response.planning_ms, 2)
        breakdown.evaluator_ms = round(agent_response.evaluator_ms, 2)
        breakdown.evaluation_ms = round(agent_response.evaluator_ms, 2)

        # Calculate tool execution time from tool_results
        tools_duration = sum(r.duration_ms for r in agent_response.tool_results)
        breakdown.tools_ms = round(tools_duration, 2)
        breakdown.tool_breakdown = {r.tool_name: r.duration_ms for r in agent_response.tool_results}

        return GatewayResult(
            response=agent_response,
            route="agent_path",
            latency=breakdown,
            source=source,
        )
