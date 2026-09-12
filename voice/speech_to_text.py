import sys
from typing import Optional
from observability.logging import get_logger

logger = get_logger("CyberJarvis.STT")


class SpeechToText:
    """Speech-to-Text interface with microphone management and graceful terminal fallback."""

    def __init__(self):
        self.sr = None
        self.recognizer = None
        self.microphone = None

        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            # Test if a microphone device exists
            try:
                self.microphone = sr.Microphone()
            except Exception as ex:
                logger.warning(f"No microphone hardware found: {ex}")
                self.microphone = None
        except ImportError:
            logger.info("speech_recognition library not installed. Terminal fallback active.")

    def is_voice_input_available(self) -> bool:
        return self.recognizer is not None and self.microphone is not None

    def listen_and_transcribe(self, timeout: float = 8.0, phrase_time_limit: float = 12.0) -> Optional[str]:
        """
        Listen from microphone and transcribe speech to text.
        Returns transcribed string, or None on silence/timeout.
        """
        if not self.is_voice_input_available():
            # Graceful fallback to console input
            try:
                text = input("\n🎤 (Mic unavailable) Speak via keyboard: ").strip()
                return text if text else None
            except (KeyboardInterrupt, EOFError):
                return None

        try:
            with self.microphone as source:
                print("\n🎤 Listening... (Speak your command)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            print("🎧 Transcribing...")
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Transcribed audio: '{text}'")
            return text
        except self.sr.WaitTimeoutError:
            logger.debug("Voice listen timed out with no audio.")
            return None
        except self.sr.UnknownValueError:
            print("❓ Could not understand audio. Please repeat.")
            return None
        except Exception as ex:
            logger.warning(f"STT error ({ex}). Falling back to text.")
            try:
                return input("\n🎤 Voice error. Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                return None
