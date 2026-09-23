import time
from typing import Optional
from voice.audio_capture import AudioCapture, AudioDeviceManager, resample_pcm16_mono, DEFAULT_SAMPLE_RATE
from voice.stt_engine import RobustSTTEngine, BaseSTTEngine, get_default_stt_engine
from agent.router import is_incomplete_transcript
from observability.logging import get_logger

logger = get_logger("CyberJarvis.STT")


class SpeechToText:
    """Production STT interface with physical microphone capture and robust fallbacks."""

    def __init__(
        self,
        audio_capture: Optional[AudioCapture] = None,
        stt_engine: Optional[BaseSTTEngine] = None,
        backend: Optional[str] = None,
    ):
        self.capture = audio_capture or AudioCapture()
        self.engine = stt_engine or get_default_stt_engine(backend=backend)
        self.last_mic_ms: float = 0.0
        self.last_vad_ms: float = 0.0
        self.last_transcribe_ms: float = 0.0
        self.last_total_stt_ms: float = 0.0
        self.telemetry: dict = {
            "backend": getattr(self.engine, "backend_name", "LOCAL"),
            "model": getattr(self.engine, "model_name", "base"),
            "load_time_ms": getattr(self.engine, "load_time_ms", 0.0),
            "inference_time_ms": 0.0,
            "capture_init_ms": 0.0,
            "calibration_ms": 0.0,
            "recording_ms": 0.0,
            "vad_silence_ms": 0.0,
            "resampling_ms": 0.0,
            "stt_ms": 0.0,
            "retry_ms": 0.0,
            "total_stt_ms": 0.0,
        }

    def get_telemetry(self) -> dict:
        """Return the most recent STT telemetry breakdown."""
        return dict(self.telemetry)

    def format_telemetry_report(self) -> str:
        """Format granular STT telemetry as specified in Phase 2 & Phase 3."""
        t = self.telemetry
        backend = str(t.get("backend", getattr(self.engine, "backend_name", "LOCAL"))).upper()
        model = str(t.get("model", getattr(self.engine, "model_name", "base")))
        load_time = float(t.get("load_time_ms", getattr(self.engine, "load_time_ms", 0.0)))
        infer_time = float(t.get("inference_time_ms", t.get("stt_ms", getattr(self.engine, "last_inference_ms", 0.0))))
        total_stt = float(t.get("total_stt_ms", 0.0))

        return (
            f"STT Backend: {backend}\n"
            f"Model: {model}\n"
            f"Load Time: {load_time:.1f} ms\n"
            f"Inference Time: {infer_time:.1f} ms\n"
            f"Capture Init: {t.get('capture_init_ms', 0.0):.1f} ms\n"
            f"Calibration: {t.get('calibration_ms', 0.0):.1f} ms\n"
            f"Recording: {t.get('recording_ms', 0.0):.1f} ms\n"
            f"VAD/Silence: {t.get('vad_silence_ms', 0.0):.1f} ms\n"
            f"Resampling: {t.get('resampling_ms', 0.0):.1f} ms\n"
            f"STT: {t.get('stt_ms', 0.0):.1f} ms\n"
            f"Retry: {t.get('retry_ms', 0.0):.1f} ms\n"
            f"Total STT: {total_stt:.1f} ms\n"
            f"TOTAL STT: {total_stt:.1f} ms"
        )

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

    def get_stt_backend_info(self) -> str:
        """Return human-readable description of active STT backend and model."""
        backend = getattr(self.engine, "backend_name", "LOCAL")
        model = getattr(self.engine, "model_name", "base")
        return f"{backend} ({model})"

    def _sync_telemetry(self, resample_ms: float, stt_ms: float, total_retry_ms: float, total_stt_ms: float):
        infer_ms = getattr(self.engine, "last_inference_ms", round(stt_ms, 2))
        self.telemetry = {
            "backend": getattr(self.engine, "backend_name", "LOCAL"),
            "model": getattr(self.engine, "model_name", "base"),
            "load_time_ms": getattr(self.engine, "load_time_ms", 0.0),
            "inference_time_ms": infer_ms,
            "capture_init_ms": round(getattr(self.capture, "last_capture_init_ms", 0.0), 2),
            "calibration_ms": round(getattr(self.capture, "last_calib_ms", 0.0), 2),
            "recording_ms": round(getattr(self.capture, "last_recording_ms", 0.0), 2),
            "vad_silence_ms": round(self.last_vad_ms, 2),
            "resampling_ms": round(resample_ms, 2),
            "stt_ms": round(stt_ms, 2),
            "retry_ms": round(total_retry_ms, 2),
            "total_stt_ms": round(total_stt_ms, 2),
        }

    def listen_and_transcribe(
        self,
        timeout: float = 2.5,
        phrase_time_limit: float = 4.0,
        max_retries: int = 2,
    ) -> Optional[str]:
        """
        Listen from physical microphone and transcribe speech to text.
        Retries at most twice upon failure with bounded duration, distinguishing:
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

        listen_start = time.perf_counter()
        retry_start_time = None
        total_retry_ms = 0.0
        resample_ms = 0.0
        stt_ms = 0.0

        # Retry loop: attempt 0 is initial try, followed by at most max_retries retries
        for attempt in range(max_retries + 1):
            if attempt > 0 and retry_start_time is None:
                retry_start_time = time.perf_counter()

            # Bounded retry parameters: user knows to speak immediately on retry
            if attempt > 0:
                cur_timeout = min(timeout, 2.0)
                cur_phrase_limit = min(phrase_time_limit, 3.0)
                cur_silence_timeout = 0.45
            else:
                cur_timeout = timeout
                cur_phrase_limit = phrase_time_limit
                cur_silence_timeout = 0.50

            try:
                print("\n🎤 Listening...")
                pcm_data, status = self.capture.record_phrase(
                    timeout=cur_timeout,
                    phrase_time_limit=cur_phrase_limit,
                    silence_timeout=cur_silence_timeout,
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
                resample_start = time.perf_counter()
                if capture_sr != stt_target_sr:
                    stt_pcm = resample_pcm16_mono(pcm_data, capture_sr, stt_target_sr)
                    stt_sr = stt_target_sr
                else:
                    stt_pcm = pcm_data
                    stt_sr = capture_sr
                resample_ms = (time.perf_counter() - resample_start) * 1000

                print("🎧 Transcribing...")
                transcribe_start = time.perf_counter()
                transcript = self.engine.transcribe(stt_pcm, sample_rate=stt_sr)
                stt_ms = (time.perf_counter() - transcribe_start) * 1000
                self.last_transcribe_ms = round(stt_ms, 2)

                if retry_start_time is not None:
                    total_retry_ms = (time.perf_counter() - retry_start_time) * 1000

                total_stt_ms = (time.perf_counter() - listen_start) * 1000
                self.last_total_stt_ms = round(total_stt_ms, 2)

                self._sync_telemetry(resample_ms, stt_ms, total_retry_ms, total_stt_ms)

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
                        print(f"\n⏱️ STT Telemetry:\n{self.format_telemetry_report()}")
                        return transcript

                logger.info(f"Transcribed audio: '{transcript}'")
                print(f"\n⏱️ STT Telemetry:\n{self.format_telemetry_report()}")
                return transcript

            except Exception as ex:
                logger.warning(f"Voice capture exception: {ex}")
                print("❌ Speech not recognized")
                if attempt < max_retries:
                    print(f"🔁 Retry {attempt + 1}/{max_retries}")
                else:
                    break

        if retry_start_time is not None:
            total_retry_ms = (time.perf_counter() - retry_start_time) * 1000
        total_stt_ms = (time.perf_counter() - listen_start) * 1000
        self.last_total_stt_ms = round(total_stt_ms, 2)
        self._sync_telemetry(resample_ms, stt_ms, total_retry_ms, total_stt_ms)

        # Keyboard fallback after failed retries
        try:
            fallback_text = input("\n⌨️ Voice unavailable. Enter command: ").strip()
            return fallback_text if fallback_text else None
        except (KeyboardInterrupt, EOFError):
            return None
