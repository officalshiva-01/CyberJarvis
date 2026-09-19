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


def run_audio_test(keep_audio: bool = False):
    """
    Diagnostic raw audio capture test (Phase 4).
    Records 3 seconds from active microphone, calculates RMS and peak,
    and saves a temporary WAV for debugging.
    """
    import wave
    import tempfile
    from voice.audio_capture import (
        AudioDeviceManager,
        calculate_pcm_rms,
        calculate_pcm_peak,
        convert_stereo_to_mono,
        DEFAULT_DTYPE,
    )
    import sounddevice as sd

    dev = AudioDeviceManager.select_input_device()
    if not dev:
        print("Microphone: None detected")
        print("Signal detected: NO")
        print("RMS: 0")
        print("Peak: 0")
        print("Capture: FAIL")
        return

    dev_name = dev["name"]
    dev_idx = dev["index"]
    sr = int(dev.get("default_samplerate", 44100))
    ch = min(dev.get("channels", 1), 2)

    print(f"Microphone: {dev_name} (Device ID: {dev_idx})")
    print("Recording 3 seconds...")

    chunks = []
    def _cb(indata, frames, time_info, status):
        chunks.append(bytes(indata))

    try:
        try:
            with sd.RawInputStream(
                samplerate=sr,
                channels=ch,
                dtype=DEFAULT_DTYPE,
                device=dev_idx,
                callback=_cb,
            ):
                time.sleep(3.0)
        except Exception:
            chunks = []
            ch = 1
            with sd.RawInputStream(
                samplerate=sr,
                channels=1,
                dtype=DEFAULT_DTYPE,
                device=dev_idx,
                callback=_cb,
            ):
                time.sleep(3.0)
    except Exception as ex:
        print("Signal detected: NO")
        print("RMS: 0")
        print("Peak: 0")
        print(f"Capture: FAIL ({ex})")
        return

    raw_pcm = b"".join(chunks)
    if ch == 2:
        pcm = convert_stereo_to_mono(raw_pcm)
    else:
        pcm = raw_pcm

    rms = calculate_pcm_rms(pcm)
    peak = calculate_pcm_peak(pcm)
    signal_detected = "YES" if (peak > 50 and rms > 10) else "NO"
    capture_status = "PASS" if (signal_detected == "YES" and len(pcm) > 0) else "FAIL"

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix="cyberjarvis_test_", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)

        print(f"Signal detected: {signal_detected}")
        print(f"RMS: {rms:.2f}")
        print(f"Peak: {peak}")
        print(f"Capture: {capture_status}")

        if keep_audio:
            print(f"Audio saved to: {tmp_path}")
    finally:
        if not keep_audio and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def run_device_inspection():
    """Inspect and test all available input devices (Phase 1)."""
    from voice.audio_capture import AudioDeviceManager
    print("=== AVAILABLE INPUT DEVICES INSPECTION ===\n")
    AudioDeviceManager.inspect_all_devices(record_duration=2.0)


def run_single_command(command: str, model: str = DEFAULT_MODEL, enable_voice_output: bool = False):
    """Run a single command non-interactively, render dashboard, and exit."""
    init_db()
    tts = TextToSpeech() if enable_voice_output else None

    def on_status_change(status_text: str):
        print(f"  ⏳ {status_text}")

    gateway = AgentGateway(on_status_change=on_status_change)
    TerminalDashboard.print_banner(mode="ONE-SHOT COMMAND", model=model)

    try:
        result = gateway.process_user_request(command, source="terminal")
        if tts and result.response.spoken_summary:
            tts_start = time.perf_counter()
            tts.speak(result.response.spoken_summary, blocking=False)
            tts_duration_ms = (time.perf_counter() - tts_start) * 1000
            result.latency.tts_ms = round(tts_duration_ms, 2)
            result.latency.total_ms = round(result.latency.total_ms + tts_duration_ms, 2)
        TerminalDashboard.render_agent_turn(command, result)
    except Exception as error:
        logger.error(f"Error processing command: {error}", exc_info=True)
        print(f"\n❌ Error: {error}\n")


def main():
    parser = argparse.ArgumentParser(description="CyberJarvis Local AI Diagnostic Agent")
    parser.add_argument("--voice", "-v", action="store_true", help="Start in voice-controlled interactive mode")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output in terminal mode")
    parser.add_argument("--diagnose", "-d", action="store_true", help="Run immediate hardware & process diagnosis")
    parser.add_argument("--command", "-c", type=str, default=None, help="Execute a single command non-interactively")
    parser.add_argument("--audio-test", action="store_true", help="Run 3-second raw audio capture diagnostic test")
    parser.add_argument("--keep-audio", action="store_true", help="Keep temporary WAV audio file from --audio-test")
    parser.add_argument("--inspect-devices", action="store_true", help="Inspect and probe all available input devices")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")

    args = parser.parse_args()

    if args.inspect_devices:
        run_device_inspection()
    elif args.audio_test:
        run_audio_test(keep_audio=args.keep_audio)
    elif args.diagnose:
        run_one_shot_diagnosis()
    elif args.command:
        run_single_command(command=args.command, model=args.model, enable_voice_output=not args.no_tts)
    elif args.voice:
        loop = VoiceAssistantLoop(model=args.model)
        loop.run()
    else:
        run_terminal_mode(model=args.model, enable_voice_output=not args.no_tts)


if __name__ == "__main__":
    main()
