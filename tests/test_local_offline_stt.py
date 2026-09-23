"""
Comprehensive Unit & Integration Tests for CyberJarvis Phase 3:
Fully Local Offline STT Architecture.

Validates:
1. Local backend selection by default (STT_BACKEND=local)
2. Explicit network backend configuration (STT_BACKEND=network)
3. Zero network calls in local mode (strictly offline boundary)
4. Transcript normalization (wake words, diagnostic speech artifacts, Hinglish)
5. Retry bounds (strictly bounded to at most 2 retries)
6. Adaptive VAD behavior and silence detection
7. Device 0 microphone selection priority and env override
8. Offline failure handling: clear initialization error & NO silent network fallback
9. Controlled network fallback only when explicitly enabled
10. Explicit telemetry reporting (STT Backend, Model, Load Time, Inference Time, Total STT)
"""
import os
import time
import struct
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from agent.router import normalize_transcript
from voice.audio_capture import (
    AudioDeviceManager,
    AudioCapture,
    resample_pcm16_mono,
    calculate_pcm_rms,
    calculate_pcm_peak,
)
from voice.speech_to_text import SpeechToText
from voice.stt_engine import (
    BaseSTTEngine,
    LocalWhisperSTTEngine,
    GoogleSTTEngine,
    RobustSTTEngine,
    LocalSTTUnavailableError,
    get_default_stt_engine,
)


class TestLocalOfflineSTT(unittest.TestCase):
    def setUp(self):
        self.orig_backend = os.environ.get("CYBERJARVIS_STT_BACKEND")
        self.orig_stt_backend = os.environ.get("STT_BACKEND")
        self.orig_audio_dev = os.environ.get("CYBERJARVIS_AUDIO_DEVICE")
        self.orig_fallback = os.environ.get("CYBERJARVIS_STT_NETWORK_FALLBACK")

    def tearDown(self):
        for key, val in [
            ("CYBERJARVIS_STT_BACKEND", self.orig_backend),
            ("STT_BACKEND", self.orig_stt_backend),
            ("CYBERJARVIS_AUDIO_DEVICE", self.orig_audio_dev),
            ("CYBERJARVIS_STT_NETWORK_FALLBACK", self.orig_fallback),
        ]:
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    # 1. LOCAL BACKEND SELECTION
    def test_local_backend_selection_by_default(self):
        """Default STT backend must be strictly LOCAL using faster-whisper without network."""
        os.environ.pop("CYBERJARVIS_STT_BACKEND", None)
        os.environ.pop("STT_BACKEND", None)

        engine = RobustSTTEngine()
        self.assertEqual(engine.backend, "local")
        self.assertEqual(engine.backend_name, "LOCAL")
        self.assertEqual(engine.model_name, "base")
        self.assertIsInstance(engine.primary_engine, LocalWhisperSTTEngine)
        self.assertGreater(engine.load_time_ms, 0.0)

    # 2. EXPLICIT NETWORK BACKEND CONFIGURATION
    def test_network_backend_explicit_configuration(self):
        """Network STT must ONLY be initialized when intentionally configured via STT_BACKEND=network."""
        os.environ["STT_BACKEND"] = "network"
        engine = RobustSTTEngine()
        self.assertEqual(engine.backend, "network")
        self.assertEqual(engine.backend_name, "NETWORK")
        self.assertIsInstance(engine.primary_engine, GoogleSTTEngine)

    # 3. NO NETWORK CALL IN LOCAL MODE
    @patch("httpx.Client.post")
    def test_no_network_call_in_local_mode(self, mock_httpx_post):
        """Local mode must never make any outbound HTTP or API requests."""
        engine = LocalWhisperSTTEngine(model_size="base", local_files_only=True)
        # Transcribe 1 second of silence
        pcm_silence = b"\x00\x00" * 16000
        engine.transcribe(pcm_silence, sample_rate=16000)

        # Assert no network call was made
        self.assertEqual(mock_httpx_post.call_count, 0)
        self.assertGreater(engine.last_inference_ms, 0.0)

    # 4. TRANSCRIPT NORMALIZATION
    def test_transcript_normalization_local_stt_outputs(self):
        """Normalize local Whisper transcripts across English, accents, and Hinglish."""
        test_pairs = [
            ("CyberJarvis, what is my name?", "what is my name"),
            ("Hey CyberJarvis, diagnosis by computer.", "diagnose my computer"),
            ("okay cyberjarvis how to become a developer", "how to become a developer"),
            ("CyberJarvis band karo", "band karo"),
            ("  CYBERJARVIS!   check my ram   ", "check my ram"),
            ("hey jarvis, what time is it?", "what time is it"),
        ]
        for raw, expected in test_pairs:
            norm = normalize_transcript(raw)
            self.assertEqual(norm, expected, f"Failed normalizing '{raw}'")

    # 5. RETRY BOUNDS
    def test_retry_bounds_in_local_mode(self):
        """SpeechToText must retry at most 2 times with bounded parameters and fallback cleanly."""
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        mock_capture.record_phrase.return_value = (None, "No speech detected")

        mock_engine = MagicMock(spec=BaseSTTEngine)
        mock_engine.backend_name = "LOCAL"
        mock_engine.model_name = "base"
        mock_engine.load_time_ms = 400.0
        mock_engine.last_inference_ms = 0.0

        stt = SpeechToText(audio_capture=mock_capture, stt_engine=mock_engine)
        with patch("builtins.input", return_value="fallback command") as mock_input:
            res = stt.listen_and_transcribe(max_retries=2)
            # attempt 0 + 2 retries = 3 calls
            self.assertEqual(mock_capture.record_phrase.call_count, 3)
            self.assertEqual(res, "fallback command")
            self.assertTrue(mock_input.called)

    # 6. VAD BEHAVIOR PRESERVATION
    def test_vad_behavior_preservation(self):
        """Adaptive VAD must detect silence and stop capture promptly."""
        import threading
        cap = AudioCapture()
        cap.sample_rate = 16000
        cap.channels = 1

        speech = struct.pack("<800h", *([4000 * (1 if i % 2 == 0 else -1) for i in range(800)]))
        silence = struct.pack("<800h", *([40] * 800))

        with patch("sounddevice.RawInputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            stop_evt = threading.Event()

            def feeder(callback):
                t_start = time.time()
                while not stop_evt.is_set():
                    elapsed = time.time() - t_start
                    if elapsed < 0.2:
                        callback(silence, 800, None, None)
                    elif elapsed < 0.6:
                        callback(speech, 800, None, None)
                    else:
                        callback(silence, 800, None, None)
                    time.sleep(0.04)

            def enter_cb(self_mock):
                cb = mock_stream_cls.call_args[1].get("callback")
                t = threading.Thread(target=feeder, args=(cb,), daemon=True)
                t.start()
                return mock_stream

            mock_stream.__enter__ = enter_cb
            mock_stream.__exit__ = lambda self, *args: stop_evt.set()

            t0 = time.perf_counter()
            pcm, status = cap.record_phrase(timeout=1.5, phrase_time_limit=3.0, silence_timeout=0.4)
            dur = time.perf_counter() - t0

            self.assertEqual(status, "Success")
            self.assertIsNotNone(pcm)
            self.assertGreater(len(pcm), 0)
            self.assertLess(dur, 2.0)

    # 7. MICROPHONE SELECTION PRESERVATION
    @patch("sounddevice.query_devices")
    @patch("voice.audio_capture.AudioDeviceManager.probe_device")
    def test_microphone_selection_device_0_priority(self, mock_probe, mock_query):
        """Microphone selection must strictly prioritize Device 0 (ALC3246)."""
        mock_query.return_value = [
            {"name": "HDA Intel PCH: ALC3246 Analog (hw:0,0)", "index": 0, "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "sysdefault", "index": 4, "max_input_channels": 128, "default_samplerate": 48000},
        ]
        mock_probe.return_value = (True, 400.0, 800, True)

        dev = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(dev)
        self.assertEqual(dev["index"], 0)
        self.assertIn("ALC3246", dev["name"])

    # 8. STT FAILURE HANDLING & NO SILENT NETWORK FALLBACK
    def test_stt_failure_handling_and_no_silent_fallback(self):
        """When local model fails to load, raise LocalSTTUnavailableError and DO NOT silently fall back to network."""
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("Model file missing")):
            with self.assertRaises(LocalSTTUnavailableError) as ctx:
                RobustSTTEngine(backend="local", allow_network_fallback=False)
            self.assertIn("Local STT model", str(ctx.exception))
            self.assertIn("unavailable or failed to initialize", str(ctx.exception))

    # 9. CONTROLLED NETWORK FALLBACK
    def test_controlled_network_fallback_when_explicitly_configured(self):
        """Network fallback must only engage if allow_network_fallback is explicitly True."""
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("Model corrupted")):
            engine = RobustSTTEngine(backend="local", allow_network_fallback=True)
            self.assertEqual(engine.backend_name, "NETWORK")
            self.assertIsInstance(engine.primary_engine, GoogleSTTEngine)

    # 10. EXPLICIT TELEMETRY REPORTING
    def test_explicit_telemetry_reporting(self):
        """Telemetry must include STT Backend, Model, Load Time, Inference Time, and Total STT."""
        mock_engine = MagicMock(spec=BaseSTTEngine)
        mock_engine.backend_name = "LOCAL"
        mock_engine.model_name = "base"
        mock_engine.load_time_ms = 437.5
        mock_engine.last_inference_ms = 512.4

        stt = SpeechToText(stt_engine=mock_engine)
        stt.telemetry = {
            "backend": "LOCAL",
            "model": "base",
            "load_time_ms": 437.5,
            "inference_time_ms": 512.4,
            "capture_init_ms": 12.0,
            "calibration_ms": 105.0,
            "recording_ms": 1100.0,
            "vad_silence_ms": 22.0,
            "resampling_ms": 3.5,
            "stt_ms": 512.4,
            "retry_ms": 0.0,
            "total_stt_ms": 1754.9,
        }

        report = stt.format_telemetry_report()
        self.assertIn("STT Backend: LOCAL", report)
        self.assertIn("Model: base", report)
        self.assertIn("Load Time: 437.5 ms", report)
        self.assertIn("Inference Time: 512.4 ms", report)
        self.assertIn("Total STT: 1754.9 ms", report)
        self.assertIn("TOTAL STT: 1754.9 ms", report)
        self.assertIn("Capture Init: 12.0 ms", report)
        self.assertIn("Calibration: 105.0 ms", report)
        self.assertIn("Recording: 1100.0 ms", report)
        self.assertIn("VAD/Silence: 22.0 ms", report)
        self.assertIn("Resampling: 3.5 ms", report)
        self.assertIn("STT: 512.4 ms", report)
        self.assertIn("Retry: 0.0 ms", report)


if __name__ == "__main__":
    unittest.main()
