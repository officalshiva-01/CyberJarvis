"""
Audio Capture Module for CyberJarvis.
Provides robust physical microphone enumeration, level monitoring, adaptive voice activity detection (VAD),
silence detection, and multi-channel handling on Linux PipeWire / ALSA.
"""
import os
import time
import math
import struct
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sounddevice as sd
from config import AUDIO_DEVICE
from observability.logging import get_logger

logger = get_logger("CyberJarvis.AudioCapture")

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"


def calculate_pcm_rms(pcm_bytes: bytes) -> int:
    """Calculate Root Mean Square (RMS) energy level of 16-bit PCM bytes with DC offset removal."""
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0
    format_str = f"<{count}h"
    try:
        shorts = struct.unpack(format_str, pcm_bytes)
        mean = sum(shorts) / count
        sum_squares = sum((s - mean) ** 2 for s in shorts)
        return int((sum_squares / count) ** 0.5)
    except Exception:
        return 0


def calculate_pcm_peak(pcm_bytes: bytes) -> int:
    """Calculate peak amplitude of 16-bit PCM bytes with DC offset removal."""
    count = len(pcm_bytes) // 2
    if count == 0:
        return 0
    format_str = f"<{count}h"
    try:
        shorts = struct.unpack(format_str, pcm_bytes)
        mean = sum(shorts) / count
        return int(max(abs(s - mean) for s in shorts))
    except Exception:
        return 0


def convert_stereo_to_mono(stereo_bytes: bytes) -> bytes:
    """
    Extract active mono channel or downmix stereo 16-bit PCM bytes.
    Handles hardware setups where one channel is floating/pegged (e.g. Dell ALC3246).
    """
    count = len(stereo_bytes) // 4
    if count == 0:
        return stereo_bytes
    try:
        samples = struct.unpack(f"<{count * 2}h", stereo_bytes[:count * 4])
        c0 = samples[0::2]
        c1 = samples[1::2]
        m0 = sum(c0) / count
        m1 = sum(c1) / count
        rms0 = math.sqrt(sum((x - m0) ** 2 for x in c0) / count)
        rms1 = math.sqrt(sum((x - m1) ** 2 for x in c1) / count)
        c0_clips = sum(1 for x in c0 if abs(x) >= 32760)
        c1_clips = sum(1 for x in c1 if abs(x) >= 32760)

        # Check for single-channel saturation / open circuit
        if (c0_clips > count * 0.75 or rms0 < 5) and (c1_clips < count * 0.75 and rms1 >= 5):
            chosen = [int(x - m1) for x in c1]
        elif (c1_clips > count * 0.75 or rms1 < 5) and (c0_clips < count * 0.75 and rms0 >= 5):
            chosen = [int(x - m0) for x in c0]
        else:
            # Average both channels
            chosen = [int(((x0 - m0) + (x1 - m1)) / 2) for x0, x1 in zip(c0, c1)]

        clamped = [max(-32768, min(32767, x)) for x in chosen]
        return struct.pack(f"<{count}h", *clamped)
    except Exception:
        return stereo_bytes


def resample_pcm16_mono(pcm_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
    """
    Cleanly resample 16-bit mono PCM audio from in_rate to out_rate using linear interpolation.
    Deterministic, fast, zero-dependency.
    """
    if in_rate == out_rate or not pcm_bytes:
        return pcm_bytes
    in_samples = len(pcm_bytes) // 2
    if in_samples == 0:
        return b""
    shorts = struct.unpack(f"<{in_samples}h", pcm_bytes[:in_samples * 2])
    out_samples = int(in_samples * out_rate / in_rate)
    if out_samples == 0:
        return b""
    ratio = (in_samples - 1) / max(1, out_samples - 1)

    out = []
    for i in range(out_samples):
        src_pos = i * ratio
        src_idx = int(src_pos)
        frac = src_pos - src_idx
        s1 = shorts[src_idx]
        s2 = shorts[src_idx + 1] if src_idx + 1 < in_samples else s1
        val = int(s1 + frac * (s2 - s1))
        out.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(out)}h", *out)


class AudioDeviceManager:
    """Manages audio devices, physical discovery, and queries PipeWire/ALSA hardware."""

    VERIFIED_PHYSICAL_PATTERNS = [
        "alc3246",
        "hw:0,0",
        "hda intel pch",
    ]
    VIRTUAL_DEVICE_PATTERNS = [
        "sysdefault",
        "default",
        "dmix",
        "pulse",
        "pipewire",
        "null",
    ]

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        """Return all available input audio devices with indices and metadata."""
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
    def probe_device(dev_idx: int, duration: float = 0.5) -> Tuple[bool, float, int, bool]:
        """Perform short real capture to verify hardware works and outputs non-zero signal."""
        try:
            dev = sd.query_devices()[dev_idx]
            sr = int(dev.get("default_samplerate", 44100))
            channels = min(dev.get("max_input_channels", 1), 2)
            chunks = []

            def _cb(indata, frames, time_info, status):
                chunks.append(bytes(indata))

            try:
                with sd.RawInputStream(
                    samplerate=sr,
                    channels=channels,
                    dtype=DEFAULT_DTYPE,
                    device=dev_idx,
                    callback=_cb,
                ):
                    time.sleep(duration)
            except Exception:
                # Retry with 1 channel if 2 channels failed
                chunks = []
                with sd.RawInputStream(
                    samplerate=sr,
                    channels=1,
                    dtype=DEFAULT_DTYPE,
                    device=dev_idx,
                    callback=_cb,
                ):
                    time.sleep(duration)

            raw = b"".join(chunks)
            if not raw:
                return False, 0.0, 0, False

            pcm = convert_stereo_to_mono(raw) if channels == 2 else raw
            rms = calculate_pcm_rms(pcm)
            peak = calculate_pcm_peak(pcm)
            has_signal = (peak > 50 and rms > 10)
            return True, float(rms), int(peak), has_signal
        except Exception as ex:
            logger.debug(f"Probe failed for device {dev_idx}: {ex}")
            return False, 0.0, 0, False

    @staticmethod
    def select_input_device() -> Optional[Dict[str, Any]]:
        """
        Microphone selection strategy (Strict 4-tier priority):
        1. CYBERJARVIS_AUDIO_DEVICE environment override
        2. Previously verified physical input device (e.g. ALC3246, hw:0,0)
        3. Physical ALSA device with 1-2 input channels (not virtual wrappers)
        4. System default only as final fallback
        Never blindly choose sd.default.device[0].
        """
        input_devs = AudioDeviceManager.get_input_devices()
        if not input_devs:
            return None

        # 1. Configured device override (CYBERJARVIS_AUDIO_DEVICE or config.AUDIO_DEVICE)
        env_val = os.getenv("CYBERJARVIS_AUDIO_DEVICE")
        cfg_val = AUDIO_DEVICE
        override = env_val if env_val is not None else cfg_val
        if override is not None:
            try:
                target_idx = int(override)
                for d in input_devs:
                    if d["index"] == target_idx:
                        logger.info(f"Selected audio device {target_idx} via explicit configuration override.")
                        return d
            except (ValueError, TypeError):
                pass

        # 2. Previously verified physical input device
        for d in input_devs:
            name_lower = d.get("name", "").lower()
            is_virtual = any(v in name_lower for v in AudioDeviceManager.VIRTUAL_DEVICE_PATTERNS)
            if not is_virtual and any(p in name_lower for p in AudioDeviceManager.VERIFIED_PHYSICAL_PATTERNS):
                ok, rms, peak, has_signal = AudioDeviceManager.probe_device(d["index"], duration=0.2)
                if ok:
                    logger.info(f"Selected verified physical microphone: {d['name']} (Device ID: {d['index']})")
                    return d

        # 3. Physical ALSA device with 1-2 input channels
        physical_candidates = []
        for d in input_devs:
            name_lower = d.get("name", "").lower()
            ch = d.get("channels", 0)
            is_virtual = any(v in name_lower for v in AudioDeviceManager.VIRTUAL_DEVICE_PATTERNS)
            if 1 <= ch <= 2 and not is_virtual:
                physical_candidates.append(d)

        # First pass: prefer physical candidate with signal
        for d in physical_candidates:
            ok, rms, peak, has_signal = AudioDeviceManager.probe_device(d["index"], duration=0.3)
            if ok and has_signal:
                logger.info(f"Auto-selected responsive physical microphone: {d['name']} (Device ID: {d['index']})")
                return d

        # Second pass: accept any physical candidate that opened successfully
        for d in physical_candidates:
            ok, rms, peak, has_signal = AudioDeviceManager.probe_device(d["index"], duration=0.2)
            if ok:
                logger.info(f"Auto-selected physical ALSA microphone: {d['name']} (Device ID: {d['index']})")
                return d

        # 4. System default only as final fallback
        for d in input_devs:
            if d.get("is_default"):
                logger.warning(f"Falling back to system default audio device: {d['name']} (Device ID: {d['index']})")
                return d
        return input_devs[0]

    @staticmethod
    def get_default_input_device() -> Optional[Dict[str, Any]]:
        """Get the preferred active input recording device."""
        return AudioDeviceManager.select_input_device()

    @staticmethod
    def is_microphone_available() -> bool:
        """Check if at least one recording device exists."""
        return len(AudioDeviceManager.get_input_devices()) > 0

    @staticmethod
    def inspect_all_devices(record_duration: float = 2.0) -> List[Dict[str, Any]]:
        """
        Inspect every available input device and perform short real recording test (Phase 1).
        Prints:
        DEVICE <index>
        name: ...
        samplerate: ...
        rms: ...
        peak: ...
        signal: YES/NO
        """
        devices = AudioDeviceManager.get_input_devices()
        results = []
        for dev in devices:
            idx = dev["index"]
            name = dev["name"]
            sr = int(dev["default_samplerate"])
            chunks = []
            err = None

            try:
                # Record 2 seconds, 1 input channel, default samplerate initially
                try:
                    with sd.RawInputStream(
                        samplerate=sr,
                        channels=1,
                        dtype=DEFAULT_DTYPE,
                        device=idx,
                        callback=lambda indata, f, t, s: chunks.append(bytes(indata)),
                    ):
                        time.sleep(record_duration)
                except Exception:
                    # Retry with device native channels if 1-channel rejected by ALSA hardware
                    ch = min(dev.get("channels", 1), 2)
                    with sd.RawInputStream(
                        samplerate=sr,
                        channels=ch,
                        dtype=DEFAULT_DTYPE,
                        device=idx,
                        callback=lambda indata, f, t, s: chunks.append(bytes(indata)),
                    ):
                        time.sleep(record_duration)
            except Exception as ex:
                err = str(ex)

            if err or not chunks:
                res = {
                    "index": idx,
                    "name": name,
                    "samplerate": sr,
                    "rms": 0.0,
                    "peak": 0,
                    "near_zero_pct": 100.0,
                    "signal": "NO",
                    "error": err,
                }
            else:
                raw_pcm = b"".join(chunks)
                count = len(raw_pcm) // 2
                shorts = struct.unpack(f"<{count}h", raw_pcm[:count * 2])
                sum_sq = sum(s * s for s in shorts)
                peak = max(abs(s) for s in shorts) if shorts else 0
                rms = math.sqrt(sum_sq / count) if count > 0 else 0.0
                nz = sum(1 for s in shorts if abs(s) <= 10)
                nz_pct = (nz / count) * 100.0 if count > 0 else 100.0
                has_signal = "YES" if (peak > 50 and rms > 10) else "NO"
                res = {
                    "index": idx,
                    "name": name,
                    "samplerate": sr,
                    "rms": round(rms, 2),
                    "peak": peak,
                    "near_zero_pct": round(nz_pct, 2),
                    "signal": has_signal,
                    "error": None,
                }

            results.append(res)
            print(f"DEVICE {res['index']}")
            print(f"name: {res['name']}")
            print(f"samplerate: {res['samplerate']}")
            print(f"rms: {res['rms']}")
            print(f"peak: {res['peak']}")
            print(f"signal: {res['signal']}\n")

        return results


class AudioCapture:
    """
    Captures live audio from physical microphone with adaptive VAD.
    Supports dynamic energy thresholds, dual RMS + peak detection,
    and PipeWire fallback.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        debug: bool = False,
    ):
        if device_index is not None:
            # Query specific device
            try:
                d = sd.query_devices(device_index)
                self.device_info = {
                    "index": device_index,
                    "name": d.get("name", "Unknown"),
                    "channels": d.get("max_input_channels", 1),
                    "default_samplerate": d.get("default_samplerate", DEFAULT_SAMPLE_RATE),
                }
            except Exception:
                self.device_info = AudioDeviceManager.select_input_device()
        else:
            self.device_info = AudioDeviceManager.select_input_device()

        self.device_index = self.device_info["index"] if self.device_info else device_index
        self.sample_rate = sample_rate or (
            int(self.device_info.get("default_samplerate", DEFAULT_SAMPLE_RATE))
            if self.device_info else DEFAULT_SAMPLE_RATE
        )
        self.channels = channels or (
            min(self.device_info.get("channels", 1), 2)
            if self.device_info else DEFAULT_CHANNELS
        )
        self.debug = debug or (os.getenv("CYBERJARVIS_DEBUG") == "1")
        self.has_pw_record = shutil.which("pw-record") is not None
        self.last_mic_capture_ms: float = 0.0
        self.last_vad_ms: float = 0.0

    def is_available(self) -> bool:
        return AudioDeviceManager.is_microphone_available()

    def calibrate_ambient_noise(self, duration: float = 0.3) -> Tuple[int, int]:
        """Measure ambient noise RMS and peak levels for baseline threshold."""
        if not self.is_available():
            return 100, 200

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

            raw_pcm = b"".join(recorded_chunks)
            pcm = convert_stereo_to_mono(raw_pcm) if self.channels == 2 else raw_pcm
            baseline_rms = calculate_pcm_rms(pcm)
            baseline_peak = calculate_pcm_peak(pcm)
            return max(baseline_rms, 80), max(baseline_peak, 150)
        except Exception as ex:
            logger.warning(f"Ambient noise calibration failed: {ex}")
            return 120, 250

    def record_phrase(
        self,
        timeout: float = 3.5,
        phrase_time_limit: float = 6.0,
        silence_timeout: float = 0.8,
    ) -> Tuple[Optional[bytes], str]:
        """
        Record a phrase from the microphone with adaptive VAD.
        Detects speech using both RMS and peak energy.
        """
        if not self.is_available():
            self.last_mic_capture_ms = 0.0
            self.last_vad_ms = 0.0
            return None, "No microphone input device detected on system."

        mic_start_time = time.perf_counter()
        vad_time_acc = 0.0
        chunks: List[bytes] = []

        def audio_callback(indata, frames, time_info, status):
            chunks.append(bytes(indata))

        stream = None
        sr = self.sample_rate
        ch = self.channels

        try:
            stream = sd.RawInputStream(
                samplerate=sr,
                channels=ch,
                dtype=DEFAULT_DTYPE,
                device=self.device_index,
                callback=audio_callback,
            )
        except Exception as ex_open:
            # Automatic fallback to device native sample rate or channel configuration
            logger.warning(f"Stream open failed ({ex_open}), attempting parameter fallback.")
            if self.device_info:
                sr = int(self.device_info.get("default_samplerate", DEFAULT_SAMPLE_RATE))
                ch = min(self.device_info.get("channels", 1), 2)
                try:
                    stream = sd.RawInputStream(
                        samplerate=sr,
                        channels=ch,
                        dtype=DEFAULT_DTYPE,
                        device=self.device_index,
                        callback=audio_callback,
                    )
                    self.sample_rate = sr
                    self.channels = ch
                except Exception as ex_stream:
                    logger.warning(f"Audio stream open retry failed: {ex_stream}. Falling back to pw-record.")
                    self.last_mic_capture_ms = round((time.perf_counter() - mic_start_time) * 1000, 2)
                    return self._record_pw_record_fallback(duration=4.0)

        speech_started = False
        speech_detected = False
        start_time = time.time()
        last_speech_time = start_time

        ambient_rms = 100
        ambient_peak = 200
        speech_threshold = 200
        peak_threshold = 350
        peak_rms = 0
        calibrated = False

        # Codec startup delay (ALC3246 hardware power-up transient)
        startup_grace_period = 0.35

        try:
            with stream:
                check_interval = 0.08
                while True:
                    time.sleep(check_interval)
                    elapsed = time.time() - start_time

                    if not chunks:
                        continue

                    # Calibration after codec has powered up and stabilized
                    if not calibrated and elapsed >= (startup_grace_period + 0.15):
                        vad_calib_start = time.perf_counter()
                        # Use audio collected after grace period for baseline
                        raw_calib = b"".join(chunks[int(len(chunks) * 0.5):])
                        calib_pcm = convert_stereo_to_mono(raw_calib) if ch == 2 else raw_calib
                        ambient_rms = calculate_pcm_rms(calib_pcm)
                        ambient_peak = calculate_pcm_peak(calib_pcm)

                        # Dynamic bounded speech threshold
                        speech_threshold = max(int(ambient_rms * 1.3), int(ambient_rms + 80), 120)
                        speech_threshold = min(speech_threshold, 1500)
                        peak_threshold = max(int(ambient_peak * 1.4), int(ambient_peak + 150), 250)
                        peak_threshold = min(peak_threshold, 3000)

                        calibrated = True
                        vad_time_acc += (time.perf_counter() - vad_calib_start)

                    # Periodic VAD check
                    if calibrated:
                        vad_calc_start = time.perf_counter()
                        recent_raw = b"".join(chunks[-3:])
                        recent_pcm = convert_stereo_to_mono(recent_raw) if ch == 2 else recent_raw
                        current_rms = calculate_pcm_rms(recent_pcm)
                        current_peak = calculate_pcm_peak(recent_pcm)

                        if current_rms > peak_rms:
                            peak_rms = current_rms

                        # Dual detection: RMS or Peak energy
                        if current_rms >= speech_threshold or current_peak >= peak_threshold:
                            if not speech_started:
                                speech_started = True
                                speech_detected = True
                                logger.info(
                                    f"Speech detected (RMS={current_rms} >= {speech_threshold} "
                                    f"or Peak={current_peak} >= {peak_threshold})"
                                )
                            last_speech_time = time.time()

                        vad_time_acc += (time.perf_counter() - vad_calc_start)

                    # 1. Timeout waiting for speech to begin
                    if not speech_started and elapsed > (timeout + startup_grace_period):
                        logger.debug("Listen timed out: No speech detected.")
                        break

                    # 2. Silence detected after speech began
                    if speech_started and (time.time() - last_speech_time) > silence_timeout:
                        logger.info("Silence detected after speech; stopping capture.")
                        break

                    # 3. Exceeded max phrase limit
                    if speech_started and elapsed > (phrase_time_limit + startup_grace_period):
                        logger.info("Reached max phrase time limit.")
                        break

            capture_duration = time.perf_counter() - mic_start_time
            self.last_mic_capture_ms = round(capture_duration * 1000, 2)
            self.last_vad_ms = round(vad_time_acc * 1000, 2)

            # Debug output (Phase 3 requirement)
            if self.debug:
                print(f"ambient_rms: {ambient_rms}")
                print(f"speech_threshold: {speech_threshold}")
                print(f"peak_rms: {peak_rms}")
                print(f"speech_detected: {speech_detected}")
                print(f"capture_duration: {capture_duration:.2f}s")

            all_raw = b"".join(chunks)
            if not all_raw:
                return None, "No audio captured."

            # Discard initial startup transient samples from recording
            grace_samples = int(self.sample_rate * startup_grace_period)
            bytes_per_sample = 2 * ch
            start_byte = min(grace_samples * bytes_per_sample, len(all_raw))
            active_raw = all_raw[start_byte:]

            if ch == 2:
                final_pcm = convert_stereo_to_mono(active_raw)
            else:
                final_pcm = active_raw

            min_valid_samples = int(self.sample_rate * 0.3)
            if not speech_started or len(final_pcm) < (min_valid_samples * 2):
                return None, "No clear speech captured."

            return final_pcm, "Success"

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
                "--channels", "1",
                "--rate", str(DEFAULT_SAMPLE_RATE),
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
                data = wav_file.read_bytes()
                pcm = data[44:] if len(data) > 44 else data
                self.sample_rate = DEFAULT_SAMPLE_RATE
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
