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
    def print_banner(
        mode: str = "TERMINAL",
        model: str = DEFAULT_MODEL,
        mic_info: Optional[str] = None,
        audio_details: Optional[Dict[str, Any]] = None,
        stt_info: Optional[str] = None,
    ):
        print("\n" + "=" * 62)
        print("                   🤖 CYBERJARVIS")
        print("             LOCAL AI DIAGNOSTIC AGENT")
        print("=" * 62)
        print(f"  🎤 MODE    : {mode}")
        print(f"  🧠 MODEL   : Ollama / {model}")
        if stt_info:
            print(f"  🎧 STT     : {stt_info}")
        print(f"  🟢 STATUS  : ONLINE & READY")
        if audio_details and isinstance(audio_details, dict):
            sr_val = audio_details.get("samplerate", 44100)
            try:
                sr_display = f"{int(sr_val)} Hz"
            except (ValueError, TypeError):
                sr_display = f"{sr_val} Hz"
            print("  🎙️ AUDIO:")
            print(f"     Device: {audio_details.get('name', 'Unknown')}")
            print(f"     ID: {audio_details.get('index', 0)}")
            print(f"     Sample Rate: {sr_display}")
        elif mic_info:
            print(f"  🎙️ AUDIO   : {mic_info}")
        print("=" * 62 + "\n")

    @staticmethod
    def render_agent_turn(user_input: str, result: GatewayResult):
        resp = result.response
        lat = result.latency

        print("\n" + "─" * 62)
        print(f"👤 USER:  {user_input}")
        print("─" * 62)

        # Execution Path & Route indication
        exec_path = getattr(result, "execution_path", None)
        if not exec_path:
            exec_path = (
                "FAST"
                if result.route == "fast_path"
                else (
                    "DIAGNOSTIC"
                    if result.route == "diagnostic_path"
                    else (
                        "INFORMATION"
                        if result.route == "information_path"
                        else ("CLARIFICATION" if result.route == "clarification" else "AGENT")
                    )
                )
            )
        profile_name = getattr(result, "project_profile", None) or getattr(resp, "project_profile", None)
        if profile_name:
            print(f"Execution Path:\n{exec_path}\n\nProject Profile: {profile_name}\n")
        else:
            print(f"Execution Path:\n{exec_path}\n")

        if exec_path == "FAST":
            route_label = "⚡ FAST PATH (Deterministic)"
        elif exec_path == "DIAGNOSTIC":
            route_label = "🔍 DIAGNOSTIC PATH (Deterministic Parallel)"
        elif exec_path == "INFORMATION":
            route_label = "💡 INFORMATION PATH (Direct LLM Response)"
        elif exec_path == "CLARIFICATION":
            route_label = "🤔 CLARIFICATION (Incomplete Command Guard)"
        else:
            route_label = "🧠 AGENT PATH (LLM Reasoning)"
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
        print("\n⏱️ TIMING")
        for line in lat.format_timing_report().splitlines():
            print(f"  {line}")
        print("─" * 62 + "\n")

    @staticmethod
    def _summarize_tool_output(r: ToolResult) -> str:
        if not r.success:
            return "Failed"
        out = r.output
        if r.tool_name == "list_project_files" and isinstance(out, list):
            return f"{len(out)} items"
        if r.tool_name == "read_project_file" and isinstance(out, dict):
            fn = out.get("filename", "file")
            return f"Read {fn}"
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
        if r.tool_name in ("get_git_status", "git_status") and isinstance(out, dict):
            return "Clean" if out.get("is_clean") else f"{out.get('modified_count', 0)} changes"
        if r.tool_name in ("get_git_recent_commits", "git_log") and isinstance(out, list):
            return f"{len(out)} commits"
        if r.tool_name in ("get_test_summary", "test_summary") and isinstance(out, dict):
            return f"{out.get('test_file_count', 0)} test files"
        if r.tool_name in ("run_project_tests", "test_execution") and isinstance(out, dict):
            return "Tests Passed" if out.get("success") else f"Exit {out.get('exit_code', -1)}"
        if r.tool_name == "read_project_log" and isinstance(out, dict):
            errs = len(out.get('errors', []))
            return f"{out.get('line_count', 0)} lines ({errs} errs)"
        if r.tool_name == "get_top_processes" and isinstance(out, list):
            return f"{len(out)} procs checked"
        if r.tool_name == "diagnose_system_bottleneck" and isinstance(out, dict):
            return f"Verdict: {out.get('verdict', 'OK')}"
        if isinstance(out, str) and len(out) <= 20:
            return out
        return "Complete"
