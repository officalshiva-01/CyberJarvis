"""
Professional Terminal Dashboard for CyberJarvis.
Renders clean, structured command-center views for both Terminal and Voice modes.
"""
from typing import List, Dict, Any, Optional
from agent.gateway import GatewayResult
from tools.schemas import ToolResult
from config import DEFAULT_MODEL


class TerminalDashboard:
    """Renders formatted ASCII/Unicode command center interface."""

    @staticmethod
    def print_banner(mode: str = "TERMINAL", model: str = DEFAULT_MODEL, mic_info: Optional[str] = None):
        print("\n" + "=" * 62)
        print("                   🤖 CYBERJARVIS")
        print("             LOCAL AI DIAGNOSTIC AGENT")
        print("=" * 62)
        print(f"  🎤 MODE    : {mode}")
        print(f"  🧠 MODEL   : Ollama / {model}")
        print(f"  🟢 STATUS  : ONLINE & READY")
        if mic_info:
            print(f"  🎙️ AUDIO   : {mic_info}")
        print("=" * 62 + "\n")

    @staticmethod
    def render_agent_turn(user_input: str, result: GatewayResult):
        resp = result.response
        lat = result.latency

        print("\n" + "─" * 62)
        print(f"👤 USER:  {user_input}")
        print("─" * 62)

        # Agent Path / Route indication
        route_label = "⚡ FAST PATH (Deterministic)" if result.route == "fast_path" else "🧠 AGENT PATH (LLM Reasoning)"
        print(f"⚙️ ROUTE: {route_label}")

        # Tools summary if tools were executed
        if resp.tool_results:
            print("\n🔧 TOOL EXECUTION:")
            for r in resp.tool_results:
                icon = "✓" if r.success else "❌"
                dur = f"({r.duration_ms}ms)" if r.duration_ms > 0 else ""
                val_summary = TerminalDashboard._summarize_tool_output(r)
                print(f"  {icon} {r.tool_name:<26} {val_summary:<20} {dur}")

        # Main Response / Diagnosis Card
        print("\n📊 AGENT RESPONSE:")
        # Indent response lines for clean layout
        for line in resp.answer.splitlines():
            print(f"  {line}")

        # Spoken Summary
        if resp.spoken_summary:
            print("\n🔊 SPOKEN:")
            print(f'  "{resp.spoken_summary}"')

        # Compact timing report and single largest source of latency
        print("\n⏱️ TIMING REPORT:")
        for line in lat.format_timing_report().splitlines():
            print(f"  {line}")
        print("─" * 62 + "\n")

    @staticmethod
    def _summarize_tool_output(r: ToolResult) -> str:
        if not r.success:
            return "Failed"
        out = r.output
        if r.tool_name == "get_cpu_usage":
            return f"{out}% CPU"
        if r.tool_name == "get_memory_usage" and isinstance(out, dict):
            return f"{out.get('percent')}% RAM"
        if r.tool_name == "get_swap_usage" and isinstance(out, dict):
            return f"{out.get('percent')}% Swap"
        if r.tool_name == "get_disk_usage" and isinstance(out, dict):
            return f"{out.get('percent')}% Disk"
        if r.tool_name == "check_network_connectivity" and isinstance(out, dict):
            return "Connected" if out.get("connected") else "Offline"
        if r.tool_name == "check_dns_resolution" and isinstance(out, dict):
            return "Resolved" if out.get("success") else "Failed"
        if r.tool_name == "get_git_status" and isinstance(out, dict):
            return "Clean" if out.get("is_clean") else f"{out.get('modified_count', 0)} changes"
        if r.tool_name == "get_top_processes" and isinstance(out, list):
            return f"{len(out)} procs checked"
        if r.tool_name == "diagnose_system_bottleneck" and isinstance(out, dict):
            return f"Verdict: {out.get('verdict', 'OK')}"
        if isinstance(out, str) and len(out) <= 20:
            return out
        return "Complete"
