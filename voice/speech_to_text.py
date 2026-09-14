import time
from typing import Optional
from voice.audio_capture import AudioCapture, AudioDeviceManager
from voice.stt_engine import RobustSTTEngine, BaseSTTEngine
from observability.logging import get_logger

logger = get_logger("CyberJarvis.STT")


class SpeechToText:
    """Production STT interface with physical microphone capture and robust fallbacks."""

    def __init__(
        self,
        audio_capture: Optional[AudioCapture] = None,
        stt_engine: Optional[BaseSTTEngine] = None,
    ):
        self.capture = audio_capture or AudioCapture()
        self.engine = stt_engine or RobustSTTEngine()
        self.last_mic_ms: float = 0.0
        self.last_vad_ms: float = 0.0
        self.last_transcribe_ms: float = 0.0
        self.last_total_stt_ms: float = 0.0

    def is_voice_input_available(self) -> bool:
        """Verify that a working microphone is detected on the system."""
        return self.capture.is_available()

    def get_microphone_info(self) -> str:
        """Return human-readable diagnostic description of the active microphone."""
        dev = AudioDeviceManager.get_default_input_device()
        if dev:
            return f"Microphone: {dev['name']} (Device ID: {dev['index']})"
        return "No microphone detected"

    def listen_and_transcribe(
        self,
        timeout: float = 7.0,
        phrase_time_limit: float = 10.0,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        Listen from physical microphone and transcribe speech to text.
        Retries at most twice upon failure (Rules 8 & 9), then provides clean keyboard fallback.
        """
        if not self.is_voice_input_available():
            try:
                print(f"\n🎤 ({self.get_microphone_info()}) Enter command: ", end="")
                text = input().strip()
                return text if text else None
            except (KeyboardInterrupt, EOFError):
                return None

        # Retry loop: attempt 0 is initial try, followed by at most max_retries retries
        for attempt in range(max_retries + 1):
            try:
                attempt_label = f" (Retry {attempt}/{max_retries})" if attempt > 0 else ""
                print(f"\n🎤 Listening...{attempt_label} (Speak your command)")
                pcm_data, status = self.capture.record_phrase(
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

                self.last_mic_ms = getattr(self.capture, "last_mic_capture_ms", 0.0)
                self.last_vad_ms = getattr(self.capture, "last_vad_ms", 0.0)

                if not pcm_data:
                    logger.debug(f"Audio capture ended: {status}")
                    if attempt < max_retries:
                        print(f"❓ No clear speech detected. Retrying ({attempt + 1}/{max_retries})...")
                        continue
                    else:
                        break

                print("🎧 Transcribing...")
                transcribe_start = time.perf_counter()
                transcript = self.engine.transcribe(pcm_data, sample_rate=self.capture.sample_rate)
                transcribe_ms = (time.perf_counter() - transcribe_start) * 1000
                self.last_transcribe_ms = round(transcribe_ms, 2)
                self.last_total_stt_ms = round(self.last_mic_ms + self.last_transcribe_ms, 2)

                if transcript:
                    logger.info(f"Transcribed audio: '{transcript}'")
                    return transcript
                else:
                    if attempt < max_retries:
                        print(f"❓ Could not transcribe speech. Retrying ({attempt + 1}/{max_retries})...")
                    else:
                        print(f"❓ Could not transcribe speech after {max_retries} retries.")

            except Exception as ex:
                logger.warning(f"Voice capture exception: {ex}")
                if attempt < max_retries:
                    print(f"⚠️ Voice error: {ex}. Retrying ({attempt + 1}/{max_retries})...")
                else:
                    break

        # Clean fallback after at most 2 retries (Rule 8 & 9)
        print("\n⚠️ Voice recognition unavailable after retries. Switching to clean text fallback.")
        try:
            fallback_text = input("⌨️ Enter command (or 'exit' to quit): ").strip()
            return fallback_text if fallback_text else None
        except (KeyboardInterrupt, EOFError):
            return None
