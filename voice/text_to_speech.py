"""
Text-to-Speech Engine for CyberJarvis.
Supports Linux speech-dispatcher (spd-say), pyttsx3, and console text fallback.
Includes speech sanitization to ensure code, markdown, and JSON are never spoken aloud.
"""
import re
import shutil
import subprocess
import threading
from typing import Optional
from config import TTS_ENGINE
from observability.logging import get_logger

logger = get_logger("CyberJarvis.TTS")


def clean_for_speech(text: str) -> str:
    """
    Sanitize text for natural voice output.
    Removes Markdown code blocks, JSON structures, tables, and special syntax.
    """
    if not text:
        return ""

    # 1. Remove markdown code blocks (both closed and unclosed)
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
    # 2. Remove inline code and dangling backticks
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    cleaned = re.sub(r"`", "", cleaned)
    # 3. Remove raw JSON keys if whole response leaked JSON prefix
    cleaned = re.sub(r"['\"]?(?:answer|spoken_summary)['\"]?\s*:\s*", "", cleaned)
    # 4. Remove markdown headers (#, ##, etc.)
    cleaned = re.sub(r"#+\s*", "", cleaned)
    # 5. Remove bold/italics markers (* and _)
    cleaned = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", cleaned)
    # 6. Remove markdown links [text](url) -> text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # 7. Remove bullet lists and dashes
    cleaned = re.sub(r"^\s*[-*•]\s*", "", cleaned, flags=re.MULTILINE)
    # 8. Remove raw JSON brackets
    cleaned = re.sub(r"[\{\}\[\]]", "", cleaned)
    # 9. Collapse whitespace and strip dangling quotes
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("'\"")

    # Rule 7: Voice responses must be concise (max 2 sentences, max 200 chars)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2])
    elif sentences:
        cleaned = " ".join(sentences)

    return cleaned[:200].strip()


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

        sanitized_text = clean_for_speech(text)
        if not sanitized_text:
            return

        logger.info(f"Speaking: '{sanitized_text[:60]}...'")

        if blocking:
            self._speak_internal(sanitized_text)
        else:
            thread = threading.Thread(target=self._speak_internal, args=(sanitized_text,), daemon=True)
            thread.start()

    def _speak_internal(self, text: str):
        # 1. Try spd-say on Linux if available
        if (self.engine == "spd-say" or self.engine == "auto") and self.spd_say_available:
            try:
                subprocess.run(
                    ["spd-say", "-r", "0", "-p", "0", text],
                    check=False,
                    timeout=10.0,
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

        # 3. Fallback: print spoken notification
        print(f"\n🔊 [Spoken]: {text}\n")
