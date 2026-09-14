"""
Audio Capture Module for CyberJarvis.
Provides robust microphone enumeration, level monitoring, voice activity detection,
silence trimming, and fallback support on Linux PipeWire / ALSA.
"""
import time
import struct
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sounddevice as sd
from observability.logging import get_logger

logger = get_logger("CyberJarvis.AudioCapture")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"


def calculate_pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate the Root Mean Square (RMS) energy level of 16-bit PCM bytes."""
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0
    format_str = f"<{count}h"
    try:
        shorts = struct.unpack(format_str, pcm_bytes)
        sum_squares = sum(s * s for s in shorts)
        return int((sum_squares / count) ** 0.5)
    except Exception:
        return 0


class AudioDeviceManager:
    """Manages audio devices and queries PipeWire/ALSA input hardware."""

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        """Return all available input audio devices with indices and names."""
        try:
            devices = sd.query_devices()
            default_in = sd.default.device[0] if sd.default.device else None
            input_devs = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devs.append({
                        "index": idx,
                        "name": dev.get("name", "Unknown"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                        "is_default": (idx == default_in),
                    })
            return input_devs
        except Exception as ex:
            logger.warning(f"Failed to query sound devices: {ex}")
            return []

    @staticmethod
    def get_default_input_device() -> Optional[Dict[str, Any]]:
        """Get the system's default input recording device."""
        devices = AudioDeviceManager.get_input_devices()
        for d in devices:
            if d["is_default"]:
                return d
        return devices[0] if devices else None

    @staticmethod
    def is_microphone_available() -> bool:
        """Check if at least one recording device exists and can be queried."""
        return len(AudioDeviceManager.get_input_devices()) > 0


class AudioCapture:
    """
    Captures live audio from the physical microphone.
    Supports ambient noise calibration, voice activity detection (VAD),
    silence detection, timeout handling, and PipeWire fallback.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = device_index
        self.has_pw_record = shutil.which("pw-record") is not None
        self.last_mic_capture_ms: float = 0.0
        self.last_vad_ms: float = 0.0

    def is_available(self) -> bool:
        return AudioDeviceManager.is_microphone_available()

    def calibrate_ambient_noise(self, duration: float = 0.5) -> int:
        """Measure ambient noise RMS level for baseline threshold."""
        if not self.is_available():
            return 500

        recorded_chunks = []

        def callback(indata, frames, time_info, status):
            recorded_chunks.append(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DEFAULT_DTYPE,
                device=self.device_index,
                callback=callback,
            )
            with stream:
                time.sleep(duration)

            pcm = b"".join(recorded_chunks)
            baseline = calculate_pcm_rms(pcm)
            logger.debug(f"Calibrated ambient noise RMS: {baseline}")
            return max(baseline, 300)
        except Exception as ex:
            logger.warning(f"Ambient noise calibration failed: {ex}")
            return 600

    def record_phrase(
        self,
        timeout: float = 3.5,
        phrase_time_limit: float = 6.0,
        silence_timeout: float = 0.8,
    ) -> Tuple[Optional[bytes], str]:
        """
        Record a phrase from the microphone.
        Listens until user speaks, then continues recording until speech ceases (silence).
        Returns:
            (pcm_bytes, status_message)
        """
        if not self.is_available():
            self.last_mic_capture_ms = 0.0
            self.last_vad_ms = 0.0
            return None, "No microphone input device detected on system."

        mic_start_time = time.perf_counter()
        vad_time_acc = 0.0

        # Calibrate baseline ambient noise quickly
        vad_calib_start = time.perf_counter()
        ambient_baseline = self.calibrate_ambient_noise(duration=0.15)
        vad_time_acc += (time.perf_counter() - vad_calib_start)
        # Bounded adaptive threshold: sensitive to clear speech while filtering quiet room noise
        speech_threshold = max(int(ambient_baseline * 1.25), 350)

        chunks: List[bytes] = []
        speech_started = False
        start_time = time.time()
        last_speech_time = start_time

        def audio_callback(indata, frames, time_info, status):
            chunks.append(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=DEFAULT_DTYPE,
                device=self.device_index,
                callback=audio_callback,
            )
            with stream:
                check_interval = 0.1
                while True:
                    time.sleep(check_interval)
                    elapsed = time.time() - start_time

                    if not chunks:
                        continue

                    # Analyze the most recent ~0.2s of audio
                    vad_calc_start = time.perf_counter()
                    recent_bytes = b"".join(chunks[-3:])
                    current_rms = calculate_pcm_rms(recent_bytes)

                    if current_rms >= speech_threshold:
                        if not speech_started:
                            speech_started = True
                            logger.info(f"Speech detected (RMS={current_rms} > {speech_threshold})")
                        last_speech_time = time.time()

                    vad_time_acc += (time.perf_counter() - vad_calc_start)

                    # 1. Timeout waiting for speech to begin
                    if not speech_started and elapsed > timeout:
                        logger.debug("Listen timed out: No speech detected.")
                        self.last_mic_capture_ms = round((time.perf_counter() - mic_start_time) * 1000, 2)
                        self.last_vad_ms = round(vad_time_acc * 1000, 2)
                        return None, "Timed out waiting for speech."

                    # 2. Silence detected after speech started
                    if speech_started and (time.time() - last_speech_time) > silence_timeout:
                        logger.info("Silence detected after speech; stopping capture.")
                        break

                    # 3. Exceeded max phrase limit
                    if speech_started and elapsed > phrase_time_limit:
                        logger.info("Reached max phrase time limit.")
                        break

            self.last_mic_capture_ms = round((time.perf_counter() - mic_start_time) * 1000, 2)
            self.last_vad_ms = round(vad_time_acc * 1000, 2)

            all_pcm = b"".join(chunks)
            if not speech_started or len(all_pcm) < (self.sample_rate * 2 * 0.4):  # less than 0.4s
                return None, "No clear speech captured."

            logger.info(f"Captured audio: {len(all_pcm)} bytes PCM ({len(all_pcm) / (self.sample_rate * 2):.2f}s)")
            return all_pcm, "Success"

        except Exception as ex:
            logger.warning(f"sounddevice stream failed ({ex}). Attempting PipeWire pw-record fallback.")
            self.last_mic_capture_ms = round((time.perf_counter() - mic_start_time) * 1000, 2)
            self.last_vad_ms = round(vad_time_acc * 1000, 2)
            return self._record_pw_record_fallback(duration=4.0)

    def _record_pw_record_fallback(self, duration: float = 4.0) -> Tuple[Optional[bytes], str]:
        """Fallback recording using native PipeWire pw-record CLI."""
        if not self.has_pw_record:
            return None, "PipeWire pw-record not available."

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            cmd = [
                "pw-record",
                "--channels", str(self.channels),
                "--rate", str(self.sample_rate),
                "--format", "s16",
                tmp_path,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(duration)
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

            wav_file = Path(tmp_path)
            if wav_file.exists() and wav_file.stat().st_size > 1000:
                # Read WAV body (skip 44 bytes header)
                data = wav_file.read_bytes()
                pcm = data[44:] if len(data) > 44 else data
                return pcm, "Success (pw-record)"
            return None, "pw-record produced no audio data."
        except Exception as ex:
            logger.error(f"pw-record fallback failed: {ex}")
            return None, f"pw-record error: {ex}"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
