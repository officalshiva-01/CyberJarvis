import time
from typing import Optional
from voice.audio_capture import AudioCapture, AudioDeviceManager, resample_pcm16_mono, DEFAULT_SAMPLE_RATE
from voice.stt_engine import RobustSTTEngine, BaseSTTEngine
from agent.router import is_incomplete_transcript
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
        if hasattr(self.capture, "device_info") and self.capture.device_info:
            dev = self.capture.device_info
            return f"Microphone: {dev['name']} (Device ID: {dev['index']})"
        dev = AudioDeviceManager.get_default_input_device()
        if dev:
            return f"Microphone: {dev['name']} (Device ID: {dev['index']})"
        return "No microphone detected"

    def get_audio_device_details(self) -> Optional[dict]:
        """Return structured device information for formatted banner display."""
        dev = getattr(self.capture, "device_info", None) or AudioDeviceManager.get_default_input_device()
        if dev:
            sr = getattr(self.capture, "sample_rate", int(dev.get("default_samplerate", 44100)))
            return {
                "name": dev.get("name", "Unknown"),
                "index": dev.get("index", 0),
                "samplerate": sr,
            }
        return None

    def listen_and_transcribe(
        self,
        timeout: float = 3.5,
        phrase_time_limit: float = 6.0,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        Listen from physical microphone and transcribe speech to text.
        Retries at most twice upon failure, distinguishing:
        - No speech detected
        - Speech detected but transcription failed
        - Transcription succeeded but result is too short / incomplete
        """
        if not self.is_voice_input_available():
            try:
                print(f"🎤 ({self.get_microphone_info()})")
                fallback_text = input("⌨️ Voice unavailable. Enter command: ").strip()
                return fallback_text if fallback_text else None
            except (KeyboardInterrupt, EOFError):
                return None

        # Retry loop: attempt 0 is initial try, followed by at most max_retries retries
        for attempt in range(max_retries + 1):
            try:
                print("\n🎤 Listening...")
                pcm_data, status = self.capture.record_phrase(
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

                self.last_mic_ms = getattr(self.capture, "last_mic_capture_ms", 0.0)
                self.last_vad_ms = getattr(self.capture, "last_vad_ms", 0.0)

                # Case A: No speech detected
                if not pcm_data:
                    logger.debug(f"Audio capture ended: {status}")
                    print("❌ No speech detected")
                    if attempt < max_retries:
                        print(f"🔁 Retry {attempt + 1}/{max_retries}")
                        continue
                    else:
                        break

                # Cleanly resample to 16000 Hz if hardware captured at another rate
                capture_sr = getattr(self.capture, "sample_rate", DEFAULT_SAMPLE_RATE)
                stt_target_sr = 16000
                if capture_sr != stt_target_sr:
                    stt_pcm = resample_pcm16_mono(pcm_data, capture_sr, stt_target_sr)
                    stt_sr = stt_target_sr
                else:
                    stt_pcm = pcm_data
                    stt_sr = capture_sr

                print("🎧 Transcribing...")
                transcribe_start = time.perf_counter()
                transcript = self.engine.transcribe(stt_pcm, sample_rate=stt_sr)
                transcribe_ms = (time.perf_counter() - transcribe_start) * 1000
                self.last_transcribe_ms = round(transcribe_ms, 2)
                self.last_total_stt_ms = round(self.last_mic_ms + self.last_transcribe_ms, 2)

                # Case B: Speech detected but transcription failed
                if not transcript:
                    print("❌ Speech not recognized")
                    if attempt < max_retries:
                        print(f"🔁 Retry {attempt + 1}/{max_retries}")
                        continue
                    else:
                        break

                # Case C: Transcription succeeded but result is too short / incomplete
                if is_incomplete_transcript(transcript):
                    logger.info(f"Incomplete transcript from STT: '{transcript}'")
                    print("\n🤔 Incomplete command detected.\n🎤 Please repeat.")
                    if attempt < max_retries:
                        print(f"🔁 Retry {attempt + 1}/{max_retries}")
                        continue
                    else:
                        return transcript

                logger.info(f"Transcribed audio: '{transcript}'")
                return transcript

            except Exception as ex:
                logger.warning(f"Voice capture exception: {ex}")
                print("❌ Speech not recognized")
                if attempt < max_retries:
                    print(f"🔁 Retry {attempt + 1}/{max_retries}")
                else:
                    break

        # Keyboard fallback after failed retries (Section 6)
        try:
            fallback_text = input("\n⌨️ Voice unavailable. Enter command: ").strip()
            return fallback_text if fallback_text else None
        except (KeyboardInterrupt, EOFError):
            return None
