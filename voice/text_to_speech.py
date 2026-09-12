import shutil
import subprocess
import threading
from typing import Optional
from config import TTS_ENGINE
from observability.logging import get_logger

logger = get_logger("CyberJarvis.TTS")


class TextToSpeech:
    """Text-to-Speech engine supporting Linux spd-say, pyttsx3, and safe text fallback."""

    def __init__(self, engine: str = TTS_ENGINE):
        self.engine = engine
        self.spd_say_available = shutil.which("spd-say") is not None
        self._pyttsx3_engine = None

        if self.engine == "pyttsx3":
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
            except Exception as ex:
                logger.warning(f"pyttsx3 unavailable: {ex}")

    def speak(self, text: str, blocking: bool = False):
        if not text or not text.strip():
            return

        clean_text = text.strip()
        logger.info(f"Speaking: '{clean_text[:60]}...'")

        if blocking:
            self._speak_internal(clean_text)
        else:
            thread = threading.Thread(target=self._speak_internal, args=(clean_text,), daemon=True)
            thread.start()

    def _speak_internal(self, text: str):
        # 1. Try spd-say on Linux if configured or available
        if self.engine == "spd-say" and self.spd_say_available:
            try:
                subprocess.run(
                    ["spd-say", "-r", "0", "-p", "0", text],
                    check=False,
                    timeout=15.0,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception as ex:
                logger.warning(f"spd-say speech output failed: {ex}")

        # 2. Try pyttsx3
        if self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return
            except Exception as ex:
                logger.warning(f"pyttsx3 speech output failed: {ex}")

        # 3. Fallback: print to console
        print(f"\n🔊 [Spoken]: {text}\n")
