"""
STT Engine Abstraction for CyberJarvis.
Supports:
- Local offline STT via faster-whisper (CTranslate2 INT8 on CPU, zero network, zero API keys)
- Network STT via Google Speech API v2 (optional fallback if explicitly configured)
- Granular telemetry: Backend, Model, Load Time, Inference Time, Total STT
- Robust failure handling and strict offline boundary enforcement
"""
import io
import os
import time
import wave
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import httpx
import numpy as np

from config import STT_BACKEND, STT_MODEL, STT_LANGUAGE, STT_NETWORK_FALLBACK
from voice.audio_capture import resample_pcm16_mono
from observability.logging import get_logger

logger = get_logger("CyberJarvis.STTEngine")

CHROMIUM_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"


class STTInitializationError(Exception):
    """Raised when an STT engine fails to initialize."""
    pass


class LocalSTTUnavailableError(STTInitializationError):
    """Raised when the local offline STT model is missing or cannot be loaded."""
    pass


class BaseSTTEngine(ABC):
    """Abstract interface for Speech-to-Text transcription engines."""

    backend_name: str = "UNKNOWN"
    model_name: str = "unknown"
    load_time_ms: float = 0.0
    last_inference_ms: float = 0.0

    @abstractmethod
    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe raw 16-bit mono PCM bytes to text."""
        pass


class LocalWhisperSTTEngine(BaseSTTEngine):
    """
    Fully local, offline STT engine using faster-whisper (CTranslate2 INT8).
    - Runs entirely offline without network access or external APIs.
    - Zero telemetry or audio sent to external servers.
    - High accuracy on English, Indian-accented English, and Hindi/Hinglish.
    - Sub-second inference latency on standard CPUs.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        language: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        local_files_only: bool = True,
    ):
        self.backend_name = "LOCAL"
        self.model_name = model_size or STT_MODEL or "base"
        self.language = language or STT_LANGUAGE or "en"
        if self.language.lower() in ("auto", "none", ""):
            self.language = None
        self.device = device
        self.compute_type = compute_type
        self.local_files_only = local_files_only
        self.load_time_ms = 0.0
        self.last_inference_ms = 0.0
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        # Enforce offline mode at the environment level to prevent accidental network requests
        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"

        load_start = time.perf_counter()
        try:
            from faster_whisper import WhisperModel
        except ImportError as err:
            err_msg = (
                f"faster-whisper is not installed in the active environment ({err}). "
                "Local offline STT requires faster-whisper. Run 'pip install faster-whisper' "
                "or configure STT_BACKEND=network to explicitly enable network STT."
            )
            logger.error(err_msg)
            raise LocalSTTUnavailableError(err_msg) from err

        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=self.local_files_only,
            )
            self.load_time_ms = round((time.perf_counter() - load_start) * 1000, 2)
            logger.info(
                f"Local Whisper model '{self.model_name}' initialized in {self.load_time_ms:.1f}ms "
                f"(device={self.device}, compute={self.compute_type}, offline={self.local_files_only})"
            )
        except Exception as ex:
            self.load_time_ms = round((time.perf_counter() - load_start) * 1000, 2)
            err_msg = (
                f"Local STT model '{self.model_name}' is unavailable or failed to initialize ({ex}). "
                "Ensure local offline model files are downloaded in the local cache, "
                "or set STT_BACKEND=network to explicitly enable network STT."
            )
            logger.error(err_msg)
            raise LocalSTTUnavailableError(err_msg) from ex

    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        if not pcm_data or len(pcm_data) < int(sample_rate * 2 * 0.2):
            self.last_inference_ms = 0.0
            return None

        # Cleanly resample to 16000 Hz if hardware captured at another rate
        if sample_rate != 16000:
            pcm_data = resample_pcm16_mono(pcm_data, sample_rate, 16000)
            sample_rate = 16000

        # Convert raw 16-bit mono PCM into float32 array normalized to [-1.0, 1.0]
        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as ex:
            logger.warning(f"Error parsing PCM buffer: {ex}")
            self.last_inference_ms = 0.0
            return None

        infer_start = time.perf_counter()
        try:
            segments, info = self.model.transcribe(
                audio_np,
                language=self.language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            transcribed_pieces = [seg.text for seg in segments]
            transcript = " ".join(transcribed_pieces).strip()
            self.last_inference_ms = round((time.perf_counter() - infer_start) * 1000, 2)

            if transcript:
                logger.info(
                    f"Local STT transcribed ({self.model_name}): '{transcript}' in {self.last_inference_ms:.1f}ms"
                )
                return transcript
        except Exception as ex:
            self.last_inference_ms = round((time.perf_counter() - infer_start) * 1000, 2)
            logger.warning(f"Local STT transcription error: {ex}")

        return None


class GoogleSTTEngine(BaseSTTEngine):
    """
    Online STT using Google Speech API endpoint with FLAC encoding.
    Supports Hindi, English, and Hinglish accents.
    Only invoked when STT_BACKEND=network is explicitly configured.
    """

    def __init__(self, languages: Optional[List[str]] = None, timeout: float = 2.5):
        self.backend_name = "NETWORK"
        self.model_name = "google-speech-v2"
        self.languages = languages or ["en-IN", "en-US", "hi-IN"]
        self.timeout = timeout
        self.flac_bin = shutil.which("flac")
        self._client: Optional[httpx.Client] = None
        self.last_stt_request_ms: float = 0.0
        self.last_flac_ms: float = 0.0
        self.load_time_ms: float = 0.0
        self.last_inference_ms: float = 0.0

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0),
            )
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
        if not pcm_data or len(pcm_data) < int(sample_rate * 2 * 0.2):
            self.last_stt_request_ms = 0.0
            self.last_flac_ms = 0.0
            self.last_inference_ms = 0.0
            return None

        # Cleanly resample to 16000 Hz if hardware captured at another rate
        if sample_rate != 16000:
            pcm_data = resample_pcm16_mono(pcm_data, sample_rate, 16000)
            sample_rate = 16000

        wav_bytes = self._pcm_to_wav(pcm_data, sample_rate)
        flac_start = time.perf_counter()
        flac_bytes = self._encode_to_flac(wav_bytes)
        self.last_flac_ms = round((time.perf_counter() - flac_start) * 1000, 2)

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

        stt_request_start = time.perf_counter()
        client = self._get_client()
        for lang in self.languages:
            url = f"https://www.google.com/speech-api/v2/recognize?client=chromium&lang={lang}&key={CHROMIUM_KEY}&pFilter=0"
            try:
                resp = client.post(url, content=payload, headers=headers)
                if resp.status_code == 200:
                    text = self._parse_google_response(resp.text)
                    if text:
                        self.last_stt_request_ms = round((time.perf_counter() - stt_request_start) * 1000, 2)
                        self.last_inference_ms = self.last_stt_request_ms
                        logger.info(f"Transcribed audio ({lang}): '{text}'")
                        return text
            except Exception as ex:
                logger.debug(f"STT request failed for {lang}: {ex}")

        self.last_stt_request_ms = round((time.perf_counter() - stt_request_start) * 1000, 2)
        self.last_inference_ms = self.last_stt_request_ms
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
    Primary STT coordinator with configurable backend (local vs network).
    Default is strictly LOCAL offline STT.
    If local STT is unavailable:
    - Reports a clear initialization error (LocalSTTUnavailableError).
    - Does NOT silently fall back to external network STT unless explicitly configured.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        allow_network_fallback: Optional[bool] = None,
    ):
        raw_backend = backend or os.getenv("CYBERJARVIS_STT_BACKEND", os.getenv("STT_BACKEND", STT_BACKEND))
        self.backend = (raw_backend or "local").strip().lower()
        self.model = model or os.getenv("CYBERJARVIS_STT_MODEL", os.getenv("STT_MODEL", STT_MODEL))
        self.language = language or os.getenv("CYBERJARVIS_STT_LANGUAGE", os.getenv("STT_LANGUAGE", STT_LANGUAGE))
        if allow_network_fallback is None:
            self.allow_network_fallback = STT_NETWORK_FALLBACK
        else:
            self.allow_network_fallback = allow_network_fallback

        self.primary_engine: BaseSTTEngine = self._build_engine()

    def _build_engine(self) -> BaseSTTEngine:
        if self.backend == "local":
            try:
                return LocalWhisperSTTEngine(
                    model_size=self.model,
                    language=self.language,
                    local_files_only=True,
                )
            except LocalSTTUnavailableError as ex:
                if self.allow_network_fallback:
                    logger.warning(
                        f"Local STT unavailable ({ex}). Controlled fallback to network STT enabled by configuration."
                    )
                    return GoogleSTTEngine()
                else:
                    logger.error(
                        f"Local STT failed to initialize: {ex}. Network fallback is disabled (STT_BACKEND=local)."
                    )
                    raise
        elif self.backend == "network":
            logger.info("Initializing network STT engine (explicitly configured).")
            return GoogleSTTEngine()
        else:
            raise ValueError(
                f"Unknown STT backend '{self.backend}'. Valid options are 'local' or 'network'."
            )

    @property
    def backend_name(self) -> str:
        return getattr(self.primary_engine, "backend_name", "LOCAL")

    @property
    def model_name(self) -> str:
        return getattr(self.primary_engine, "model_name", "base")

    @property
    def load_time_ms(self) -> float:
        return getattr(self.primary_engine, "load_time_ms", 0.0)

    @property
    def last_inference_ms(self) -> float:
        return getattr(self.primary_engine, "last_inference_ms", 0.0)

    def transcribe(self, pcm_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        try:
            return self.primary_engine.transcribe(pcm_data, sample_rate=sample_rate)
        except Exception as ex:
            logger.warning(f"Transcription error in {self.backend_name} STT engine: {ex}")
            return None


def get_default_stt_engine(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    allow_network_fallback: Optional[bool] = None,
) -> BaseSTTEngine:
    """Factory function to build the configured default STT engine."""
    return RobustSTTEngine(
        backend=backend,
        model=model,
        allow_network_fallback=allow_network_fallback,
    )
