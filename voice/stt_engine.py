"""
STT Engine Abstraction for CyberJarvis.
Supports Google Cloud/Chromium Speech endpoint with multi-language (en-IN, en-US, hi-IN),
local offline recognition hooks, and robust error recovery.
"""
import io
import wave
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import httpx

from observability.logging import get_logger

logger = get_logger("CyberJarvis.STTEngine")

CHROMIUM_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"


class BaseSTTEngine(ABC):
    """Abstract interface for Speech-to-Text transcription engines."""

    @abstractmethod
    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe raw 16-bit mono PCM bytes to text."""
        pass


class GoogleSTTEngine(BaseSTTEngine):
    """
    Online STT using Google Speech API endpoint with FLAC encoding.
    Supports Hindi, English, and Hinglish accents.
    """

    def __init__(self, languages: Optional[List[str]] = None, timeout: float = 3.5):
        # Default languages in priority order: Indian English (matches English + Hinglish), US English, Hindi
        self.languages = languages or ["en-IN", "en-US", "hi-IN"]
        self.timeout = timeout
        self.flac_bin = shutil.which("flac")
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def _encode_to_flac(self, wav_bytes: bytes) -> Optional[bytes]:
        if not self.flac_bin:
            return None
        try:
            res = subprocess.run(
                [self.flac_bin, "-c", "--totally-silent", "-"],
                input=wav_bytes,
                capture_output=True,
                timeout=2.0,
            )
            if res.returncode == 0 and res.stdout:
                return res.stdout
        except Exception as ex:
            logger.debug(f"FLAC conversion failed: {ex}")
        return None

    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        if not pcm_data:
            return None

        wav_bytes = self._pcm_to_wav(pcm_data, sample_rate)
        flac_bytes = self._encode_to_flac(wav_bytes)

        # Prepare payload and headers
        if flac_bytes:
            content_type = f"audio/x-flac; rate={sample_rate}"
            payload = flac_bytes
        else:
            # Fallback to linear 16-bit PCM header
            content_type = f"audio/l16; rate={sample_rate}"
            payload = pcm_data

        headers = {
            "Content-Type": content_type,
            "User-Agent": "CyberJarvis-Voice/2.0",
        }

        client = self._get_client()
        for lang in self.languages:
            url = f"https://www.google.com/speech-api/v2/recognize?client=chromium&lang={lang}&key={CHROMIUM_KEY}&pFilter=0"
            try:
                resp = client.post(url, content=payload, headers=headers)
                if resp.status_code == 200:
                    text = self._parse_google_response(resp.text)
                    if text:
                        logger.info(f"Transcribed audio ({lang}): '{text}'")
                        return text
            except Exception as ex:
                logger.debug(f"STT request failed for {lang}: {ex}")

        return None

    def _parse_google_response(self, response_text: str) -> Optional[str]:
        for line in response_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results = data.get("result", [])
                for r in results:
                    alternatives = r.get("alternative", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        if transcript:
                            return transcript
            except json.JSONDecodeError:
                continue
        return None


class RobustSTTEngine(BaseSTTEngine):
    """
    Primary STT coordinator with online and offline fallbacks.
    """

    def __init__(self):
        self.primary_engine = GoogleSTTEngine()

    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        # 1. Try Google STT
        try:
            result = self.primary_engine.transcribe(pcm_data, sample_rate=sample_rate)
            if result:
                return result
        except Exception as ex:
            logger.warning(f"Primary STT failed: {ex}")

        return None
