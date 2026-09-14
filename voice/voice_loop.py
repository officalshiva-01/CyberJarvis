import re
import time
from typing import Optional
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech
from agent.gateway import AgentGateway
from ui.dashboard import TerminalDashboard
from config import DEFAULT_MODEL
from observability.logging import get_logger

logger = get_logger("CyberJarvis.VoiceLoop")

VOICE_EXIT_COMMANDS = {"exit", "quit", "goodbye", "bye", "stop", "terminate", "band karo"}


class VoiceAssistantLoop:
    """Production voice-driven conversation loop."""

    def __init__(self, gateway: Optional[AgentGateway] = None, model: str = DEFAULT_MODEL):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.model = model

        def on_agent_status(status_msg: str):
            print(f"  ⏳ {status_msg}")

        self.gateway = gateway or AgentGateway(on_status_change=on_agent_status)

    def run(self):
        mic_info = self.stt.get_microphone_info()
        TerminalDashboard.print_banner(mode="VOICE INTERACTIVE", model=self.model, mic_info=mic_info)

        welcome_phrase = "CyberJarvis is online and ready for voice commands."
        print(f"🔊 [Spoken]: {welcome_phrase}\n")
        self.tts.speak(welcome_phrase, blocking=False)

        consecutive_failures = 0

        while True:
            try:
                listen_start = time.perf_counter()
                user_command = self.stt.listen_and_transcribe()
                stt_duration_ms = (time.perf_counter() - listen_start) * 1000

                if not user_command:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("\nCyberJarvis: Stopping voice loop after repeated input failures.")
                        break
                    continue

                consecutive_failures = 0
                clean_cmd = user_command.lower().strip()
                clean_tokens = set(re.sub(r"[^\w\s]", "", clean_cmd).split())

                # Rule 10: Clear voice exit command: exit / quit / bye / goodbye
                if clean_cmd in VOICE_EXIT_COMMANDS or bool(clean_tokens.intersection(VOICE_EXIT_COMMANDS)):
                    farewell = "Goodbye! Shutting down CyberJarvis."
                    print(f"\n🤖 CyberJarvis: {farewell}")
                    self.tts.speak(farewell, blocking=True)
                    break

                # Process command through Agent Gateway with latency measurement
                gateway_result = self.gateway.process_user_request(
                    user_input=user_command,
                    source="voice",
                    stt_latency_ms=stt_duration_ms,
                    mic_capture_ms=getattr(self.stt, "last_mic_ms", 0.0),
                    vad_ms=getattr(self.stt, "last_vad_ms", 0.0),
                )

                # Speak natural summary and measure TTS latency
                spoken_text = gateway_result.response.spoken_summary or gateway_result.response.answer
                if spoken_text:
                    tts_start = time.perf_counter()
                    self.tts.speak(spoken_text, blocking=True)
                    tts_duration_ms = (time.perf_counter() - tts_start) * 1000
                    gateway_result.latency.tts_ms = round(tts_duration_ms, 2)
                    gateway_result.latency.total_ms = round(gateway_result.latency.total_ms + tts_duration_ms, 2)

                # Render turn in Terminal Dashboard (including full timing report)
                TerminalDashboard.render_agent_turn(user_command, gateway_result)

            except (KeyboardInterrupt, EOFError):
                print("\nShutting down CyberJarvis voice loop...")
                break
            except Exception as ex:
                logger.error(f"Voice loop exception: {ex}", exc_info=True)
                print(f"\n❌ Error in voice processing: {ex}\n")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
