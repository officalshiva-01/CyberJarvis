"""
Unit and regression tests for CyberJarvis Phase 2 STT Latency Optimization.
Validates:
- Microphone selection (Device ID 0 ALC3246 preference and env override)
- Adaptive VAD stop condition and silence timeout
- Native 44.1kHz -> 16kHz resampling
- Strict retry duration bounding and retry limits
- STT failure handling and keyboard fallback
- Transcript normalization
- Comprehensive Phase 2 telemetry breakdown
"""
import io
import os
import sys
import time
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import normalize_transcript
from voice.audio_capture import (
    AudioDeviceManager,
    AudioCapture,
    resample_pcm16_mono,
    calculate_pcm_rms,
    calculate_pcm_peak,
    convert_stereo_to_mono,
)
from voice.speech_to_text import SpeechToText
from voice.stt_engine import BaseSTTEngine, GoogleSTTEngine


class TestVoiceLatencyAndVAD(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.get("CYBERJARVIS_AUDIO_DEVICE")

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["CYBERJARVIS_AUDIO_DEVICE"] = self.orig_env
        else:
            os.environ.pop("CYBERJARVIS_AUDIO_DEVICE", None)

    # 1. MICROPHONE SELECTION
    @patch("sounddevice.query_devices")
    @patch("voice.audio_capture.AudioDeviceManager.probe_device")
    def test_microphone_selection_alc3246_priority(self, mock_probe, mock_query):
        """Microphone selection must strictly choose verified physical device ALC3246 Device ID 0."""
        mock_query.return_value = [
            {"name": "HDA Intel PCH: ALC3246 Analog (hw:0,0)", "index": 0, "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "sysdefault", "index": 4, "max_input_channels": 128, "default_samplerate": 48000},
            {"name": "default", "index": 10, "max_input_channels": 128, "default_samplerate": 48000},
        ]
        mock_probe.return_value = (True, 500.0, 1000, True)

        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 0)
        self.assertIn("ALC3246", selected["name"])

    @patch("sounddevice.query_devices")
    def test_microphone_selection_env_override(self, mock_query):
        """CYBERJARVIS_AUDIO_DEVICE must explicitly override default selection."""
        mock_query.return_value = [
            {"name": f"Output {i}", "max_input_channels": 0, "default_samplerate": 48000}
            for i in range(4)
        ] + [
            {"name": "CI test microphone", "max_input_channels": 1, "default_samplerate": 16000}
        ]

        os.environ["CYBERJARVIS_AUDIO_DEVICE"] = "4"
        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 4)

    # 2. VAD STOP CONDITION & SILENCE TIMEOUT
    def test_vad_silence_detection_stops_promptly(self):
        """VAD must stop recording promptly after silence_timeout following speech."""
        import threading
        cap = AudioCapture()
        cap.sample_rate = 16000
        cap.channels = 1

        speech_chunk = struct.pack("<800h", *([3000 * (1 if i % 2 == 0 else -1) for i in range(800)]))
        silence_chunk = struct.pack("<800h", *([50] * 800))

        with patch("sounddevice.RawInputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            stop_event = threading.Event()

            def stream_worker(cb):
                t0 = time.time()
                while not stop_event.is_set():
                    now = time.time() - t0
                    if now < 0.3:
                        cb(silence_chunk, 800, None, None)
                    elif now < 0.7:
                        cb(speech_chunk, 800, None, None)
                    else:
                        cb(silence_chunk, 800, None, None)
                    time.sleep(0.04)

            def mock_enter(self_mock):
                cb = mock_stream_cls.call_args[1].get("callback")
                t = threading.Thread(target=stream_worker, args=(cb,))
                t.daemon = True
                t.start()
                return mock_stream

            mock_stream.__enter__ = mock_enter
            mock_stream.__exit__ = lambda self, *args: stop_event.set()

            t_start = time.perf_counter()
            data, status = cap.record_phrase(timeout=1.5, phrase_time_limit=3.0, silence_timeout=0.4)
            dur = time.perf_counter() - t_start

            self.assertEqual(status, "Success")
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            # Must stop promptly after speech (~1.1 - 1.4s), not wait for phrase limit (3.0s)
            self.assertLess(dur, 2.0)

    # 3. RESAMPLING
    def test_resample_pcm16_mono_44100_to_16000(self):
        """Audio captured at 44.1kHz must be correctly resampled to 16kHz with exact sample ratio."""
        in_rate = 44100
        out_rate = 16000
        num_samples = 44100  # 1 second
        samples = [int(2000 * (1 if i % 2 == 0 else -1)) for i in range(num_samples)]
        in_bytes = struct.pack(f"<{num_samples}h", *samples)

        out_bytes = resample_pcm16_mono(in_bytes, in_rate, out_rate)
        expected_samples = int(num_samples * out_rate / in_rate)
        self.assertEqual(len(out_bytes), expected_samples * 2)

        # Same rate returns original unmodified
        same = resample_pcm16_mono(in_bytes, 16000, 16000)
        self.assertEqual(same, in_bytes)

    # 4. RETRY BOUNDS & STT FAILURE HANDLING
    def test_stt_retries_bounded_at_most_twice(self):
        """SpeechToText must retry at most 2 times with bounded parameters and fallback cleanly."""
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        mock_capture.record_phrase.return_value = (None, "No clear speech captured.")

        stt = SpeechToText(audio_capture=mock_capture)
        with patch("builtins.input", return_value="diagnose computer") as mock_input:
            result = stt.listen_and_transcribe(max_retries=2)
            # Call count: attempt 0 + 2 retries = 3 calls
            self.assertEqual(mock_capture.record_phrase.call_count, 3)
            self.assertEqual(result, "diagnose computer")
            self.assertTrue(mock_input.called)

    def test_stt_failure_handling_and_retry_telemetry(self):
        """Failed recognition attempts must be tracked in retry_ms and bounded."""
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        mock_capture.record_phrase.return_value = (b"\x00\x01" * 8000, "Success")
        mock_capture.last_mic_capture_ms = 400.0
        mock_capture.last_calib_ms = 50.0
        mock_capture.last_recording_ms = 350.0
        mock_capture.last_vad_ms = 20.0

        mock_engine = MagicMock(spec=BaseSTTEngine)
        # Attempt 0: fails (None), Attempt 1: succeeds ("what is my name")
        mock_engine.transcribe.side_effect = [None, "what is my name"]

        stt = SpeechToText(audio_capture=mock_capture, stt_engine=mock_engine)
        res = stt.listen_and_transcribe(max_retries=2)

        self.assertEqual(res, "what is my name")
        self.assertEqual(mock_engine.transcribe.call_count, 2)
        telem = stt.get_telemetry()
        self.assertIn("retry_ms", telem)
        self.assertGreater(telem["total_stt_ms"], 0)

    # 5. TELEMETRY BREAKDOWN FORMATTING
    def test_telemetry_breakdown_all_8_metrics(self):
        """Telemetry must include all 8 required Stage 1 metrics formatted correctly."""
        stt = SpeechToText()
        stt.telemetry = {
            "capture_init_ms": 15.2,
            "calibration_ms": 110.5,
            "recording_ms": 1250.0,
            "vad_silence_ms": 25.4,
            "resampling_ms": 4.1,
            "stt_ms": 520.3,
            "retry_ms": 0.0,
            "total_stt_ms": 1925.5,
        }
        report = stt.format_telemetry_report()
        self.assertIn("Capture Init: 15.2 ms", report)
        self.assertIn("Calibration: 110.5 ms", report)
        self.assertIn("Recording: 1250.0 ms", report)
        self.assertIn("VAD/Silence: 25.4 ms", report)
        self.assertIn("Resampling: 4.1 ms", report)
        self.assertIn("STT: 520.3 ms", report)
        self.assertIn("Retry: 0.0 ms", report)
        self.assertIn("TOTAL STT: 1925.5 ms", report)

    # 6. TRANSCRIPT NORMALIZATION
    def test_transcript_normalization_voice_commands(self):
        """Normalization must handle wake words, punctuation, and artifacts."""
        cases = [
            ("CyberJarvis, what is my name?", "what is my name"),
            ("hey cyberjarvis diagnose my computer", "diagnose my computer"),
            ("and diagnosis my computer", "diagnose my computer"),
            ("okay cyberjarvis how to become a developer", "how to become a developer"),
        ]
        for raw, expected in cases:
            norm = normalize_transcript(raw)
            self.assertEqual(norm, expected)


if __name__ == "__main__":
    unittest.main()
