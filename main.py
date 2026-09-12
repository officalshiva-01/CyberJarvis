"""
CyberJarvis: Production-Oriented Voice-Controlled Local AI Agent.

Combines:
- Local LLM Planning (Ollama / phi3:mini)
- Local System Diagnostics (CPU, RAM, Swap, Disk, Top Processes, Network, Git)
- Unified Tool Registry & Execution Engine
- Security Policy & Confirmation Guardrails
- Persistent SQLite Memory (Facts, Typed Memories, Relevance Ranking)
- Speech-to-Text (STT) and Text-to-Speech (TTS) Voice Pipeline
- Structured Observability & Tracing
"""
import sys
import argparse
from pathlib import Path

from config import DEFAULT_MODEL, DB_PATH
from memory import init_db
from agent.agent_core import AgentCore
from voice.text_to_speech import TextToSpeech
from voice.voice_loop import VoiceAssistantLoop
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Main")

EXIT_COMMANDS = {"exit", "quit", "bye", "/bye"}


def run_terminal_mode(model: str = DEFAULT_MODEL, enable_voice_output: bool = True):
    """Interactive terminal agent mode with real-time status and optional TTS."""
    init_db()
    tts = TextToSpeech() if enable_voice_output else None

    def on_status_change(status_text: str):
        print(status_text)

    agent = AgentCore(on_status_change=on_status_change)

    print("\n" + "=" * 55)
    print("🤖 CyberJarvis AI Agent")
    print("Persistent memory: ON (SQLite)")
    print(f"LLM Engine: Ollama ({model})")
    print("Tools: Hardware Diagnostics, Network, Git, Files, Actions")
    print("Type 'exit', 'quit', or 'bye' to quit.")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCyberJarvis: Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("CyberJarvis: Goodbye.")
            if tts:
                tts.speak("Goodbye.", blocking=False)
            break

        try:
            response = agent.process(user_input)
            print(f"\nCyberJarvis:\n{response.answer}\n")

            if tts and response.spoken_summary:
                tts.speak(response.spoken_summary, blocking=False)

        except Exception as error:
            logger.error(f"Error processing command: {error}", exc_info=True)
            print(f"\n❌ Error: {error}\n")


def run_one_shot_diagnosis():
    """Direct one-shot system diagnosis command."""
    from tools.diagnostic_tools import diagnose_system_bottleneck
    print("\n🔍 Running Comprehensive System Diagnosis...\n")
    diag = diagnose_system_bottleneck()

    print(f"Diagnosis Verdict: {diag['verdict']}")
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
    parser = argparse.ArgumentParser(description="CyberJarvis Local AI Agent")
    parser.add_argument("--voice", "-v", action="store_true", help="Start in voice-controlled interactive mode")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output in terminal mode")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run immediate hardware & process diagnosis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if args.diagnose:
        run_one_shot_diagnosis()
    elif args.voice:
        loop = VoiceAssistantLoop()
        loop.run()
    else:
        run_terminal_mode(model=args.model, enable_voice_output=not args.no_tts)


if __name__ == "__main__":
    main()
