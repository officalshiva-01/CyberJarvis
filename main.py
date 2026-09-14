"""
CyberJarvis: Production Voice-Controlled Local AI Diagnostic Agent.

Combines:
- Physical Microphone Audio Capture (PipeWire / ALSA)
- Speech-to-Text & Text-to-Speech (spd-say)
- Unified Agent Gateway (Fast Path & Agent Path)
- Local LLM Planning (Ollama / phi3:mini)
- Parallel Diagnostic Execution (psutil / network / git)
- Evidence-Grounded Evaluator (Strict anti-hallucination guardrails)
- Persistent SQLite Memory (Facts, Typed Memories, Relevance Ranking)
- Professional Terminal Dashboard UI
"""
import sys
import time
import argparse
from pathlib import Path

from config import DEFAULT_MODEL, DB_PATH
from memory import init_db
from agent.gateway import AgentGateway
from voice.text_to_speech import TextToSpeech
from voice.voice_loop import VoiceAssistantLoop
from ui.dashboard import TerminalDashboard
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Main")

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "/bye", "band karo"}


def run_terminal_mode(model: str = DEFAULT_MODEL, enable_voice_output: bool = True):
    """Interactive terminal agent mode with real-time dashboard and optional TTS."""
    init_db()
    tts = TextToSpeech() if enable_voice_output else None

    def on_status_change(status_text: str):
        print(f"  ⏳ {status_text}")

    gateway = AgentGateway(on_status_change=on_status_change)
    TerminalDashboard.print_banner(mode="TERMINAL INTERACTIVE", model=model)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCyberJarvis: Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            farewell = "Goodbye! Shutting down CyberJarvis."
            print(f"\nCyberJarvis: {farewell}\n")
            if tts:
                tts.speak(farewell, blocking=False)
            break

        try:
            result = gateway.process_user_request(user_input, source="terminal")

            if tts and result.response.spoken_summary:
                tts_start = time.perf_counter()
                tts.speak(result.response.spoken_summary, blocking=False)
                tts_duration_ms = (time.perf_counter() - tts_start) * 1000
                result.latency.tts_ms = round(tts_duration_ms, 2)
                result.latency.total_ms = round(result.latency.total_ms + tts_duration_ms, 2)

            TerminalDashboard.render_agent_turn(user_input, result)

        except Exception as error:
            logger.error(f"Error processing command: {error}", exc_info=True)
            print(f"\n❌ Error: {error}\n")


def run_one_shot_diagnosis():
    """Direct one-shot system diagnosis command."""
    from tools.diagnostic_tools import diagnose_system_bottleneck
    print("\n🔍 Running Comprehensive System Diagnosis...\n")
    diag = diagnose_system_bottleneck()

    verdict_emoji = "🟢" if diag["verdict"] == "HEALTHY" else "🔴"
    print(f"Diagnosis Verdict: {verdict_emoji} {diag['verdict']}")
    print(f"Summary: {diag['summary']}\n")
    print("Metrics:")
    for k, v in diag["metrics"].items():
        print(f"  - {k}: {v}")

    if diag["top_memory_consumers"]:
        print("\nTop Memory Processes:")
        for p in diag["top_memory_consumers"][:3]:
            print(f"  - PID {p['pid']} ({p['name']}): {p['memory_percent']}% RAM")

    if diag["recommendations"]:
        print("\nRecommendations:")
        for r in diag["recommendations"]:
            print(f"  - {r}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CyberJarvis Local AI Diagnostic Agent")
    parser.add_argument("--voice", "-v", action="store_true", help="Start in voice-controlled interactive mode")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output in terminal mode")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run immediate hardware & process diagnosis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if args.diagnose:
        run_one_shot_diagnosis()
    elif args.voice:
        loop = VoiceAssistantLoop(model=args.model)
        loop.run()
    else:
        run_terminal_mode(model=args.model, enable_voice_output=not args.no_tts)


if __name__ == "__main__":
    main()
