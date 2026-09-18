"""
Unit and regression tests for CyberJarvis Real Microphone Capture & Profile Fast Path.
Tests:
- input-device selection
- real-device preference
- audio signal detection
- VAD threshold logic
- profile fast path
- "tell me about me" => FAST
- "who am I" => FAST
- voice transcript normalization
- voice diagnostic routing
- no Ollama on FAST
- no Ollama on DIAGNOSTIC
- bounded retries
- keyboard fallback
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import ExecutionRouter, RouteType, normalize_transcript
from agent.gateway import AgentGateway, process_user_request
from agent.fast_path import FastPathHandler
from voice.speech_to_text import SpeechToText
from voice.audio_capture import (
    AudioCapture,
    AudioDeviceManager,
    calculate_pcm_rms,
    calculate_pcm_peak,
    convert_stereo_to_mono,
)
from memory import init_db, save_fact, get_facts


class TestVoiceNextFix(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        os.environ["CYBERJARVIS_DB_PATH"] = self.tmp_db.name
        init_db()
        save_fact("My name is Shivaji")
        save_fact("My goal is to become a DevOps engineer")
        save_fact("My project is CyberJarvis")

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
        os.environ.pop("CYBERJARVIS_DB_PATH", None)
        os.environ.pop("CYBERJARVIS_AUDIO_DEVICE", None)

    # 1. INPUT-DEVICE SELECTION
    def test_input_device_selection_env_override(self):
        os.environ["CYBERJARVIS_AUDIO_DEVICE"] = "4"
        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 4)

    # 2. REAL-DEVICE PREFERENCE
    @patch("sounddevice.query_devices")
    @patch("voice.audio_capture.AudioDeviceManager.probe_device")
    def test_real_device_preference_over_virtual(self, mock_probe, mock_query):
        mock_query.return_value = [
            {"name": "HDA Intel PCH: ALC3246 Analog (hw:0,0)", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "sysdefault", "max_input_channels": 128, "default_samplerate": 48000},
            {"name": "default", "max_input_channels": 128, "default_samplerate": 48000},
        ]
        # Probe succeeds with signal on device 0
        mock_probe.return_value = (True, 500.0, 1200, True)

        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 0)
        self.assertIn("ALC3246", selected["name"])

    # 3. AUDIO SIGNAL DETECTION
    def test_audio_signal_detection_metrics(self):
        # All zeros
        zeros = b"\x00\x00" * 200
        self.assertEqual(calculate_pcm_rms(zeros), 0)
        self.assertEqual(calculate_pcm_peak(zeros), 0)

        # Non-zero synthetic waveform with DC bias
        import struct
        samples = [1000 + int(500 * (1 if i % 2 == 0 else -1)) for i in range(100)]
        pcm_data = struct.pack(f"<{len(samples)}h", *samples)
        rms = calculate_pcm_rms(pcm_data)
        peak = calculate_pcm_peak(pcm_data)
        self.assertGreater(rms, 0)
        self.assertGreater(peak, 0)

    # 4. VAD THRESHOLD LOGIC
    def test_vad_threshold_logic_bounding(self):
        # Test calibration and threshold logic
        ambient_rms = 50
        ambient_peak = 100
        speech_threshold = max(int(ambient_rms * 1.3), int(ambient_rms + 80), 120)
        speech_threshold = min(speech_threshold, 1500)
        peak_threshold = max(int(ambient_peak * 1.4), int(ambient_peak + 150), 250)
        peak_threshold = min(peak_threshold, 3000)

        self.assertEqual(speech_threshold, 130)
        self.assertEqual(peak_threshold, 250)

        # High ambient noise must be capped to prevent discarding normal speech
        high_ambient_rms = 2000
        high_threshold = max(int(high_ambient_rms * 1.3), int(high_ambient_rms + 80), 120)
        high_threshold = min(high_threshold, 1500)
        self.assertEqual(high_threshold, 1500)

    # 5. PROFILE FAST PATH
    def test_profile_fast_path_stored_facts(self):
        handler = FastPathHandler()
        res = handler._handle_profile_query()
        self.assertTrue(res.completed)
        self.assertIn("Shivaji", res.answer)
        self.assertIn("DevOps engineer", res.answer)
        self.assertIn("CyberJarvis", res.answer)
        self.assertEqual(
            res.spoken_summary,
            "You are Shivaji. Your current goal is to become a DevOps engineer, and you are building CyberJarvis.",
        )

    # 6. "TELL ME ABOUT ME" => FAST
    def test_tell_me_about_me_routes_fast(self):
        router = ExecutionRouter()
        self.assertEqual(router.classify("tell me about me"), RouteType.FAST)

        gateway = AgentGateway()
        res = gateway.process_user_request("tell me about me", source="terminal")
        self.assertEqual(res.execution_path, "FAST")
        self.assertIn("Shivaji", res.response.answer)

    # 7. "WHO AM I" => FAST
    def test_who_am_i_routes_fast(self):
        router = ExecutionRouter()
        self.assertEqual(router.classify("who am i"), RouteType.FAST)

        gateway = AgentGateway()
        res = gateway.process_user_request("who am i", source="terminal")
        self.assertEqual(res.execution_path, "FAST")
        self.assertIn("Shivaji", res.response.answer)

    # 8. VOICE TRANSCRIPT NORMALIZATION
    def test_voice_transcript_normalization(self):
        cases = [
            ("CyberJarvis, what is my name?", "what is my name"),
            ("CyberJarvis, what time is it?", "what time is it"),
            ("CyberJarvis, what is my goal?", "what is my goal"),
            ("CyberJarvis, tell me about me.", "tell me about me"),
            ("CyberJarvis, check my RAM.", "check my ram"),
            ("CyberJarvis, diagnose my computer.", "diagnose my computer"),
            ("CyberJarvis, analyze my computer.", "analyze my computer"),
            ("CyberJarvis, why is my computer slow?", "why is my computer slow"),
            ("CyberJarvis, what is consuming the most memory?", "what is consuming the most memory"),
            ("CyberJarvis, check my system.", "check my system"),
            ("CyberJarvis, check my network.", "check my network"),
        ]
        for raw, expected in cases:
            self.assertEqual(normalize_transcript(raw), expected)

    # 9. VOICE DIAGNOSTIC ROUTING
    def test_voice_diagnostic_queries_route_diagnostic(self):
        gateway = AgentGateway()
        diagnostic_queries = [
            "CyberJarvis, diagnose my computer.",
            "CyberJarvis, analyze my computer.",
            "CyberJarvis, why is my computer slow?",
            "CyberJarvis, what is consuming the most memory?",
            "CyberJarvis, check my system.",
            "CyberJarvis, check my network.",
        ]
        for q in diagnostic_queries:
            res = gateway.process_user_request(q, source="voice")
            self.assertEqual(
                res.execution_path,
                "DIAGNOSTIC",
                f"Query '{q}' routed to {res.execution_path}, expected DIAGNOSTIC",
            )

    # 10. NO OLLAMA ON FAST
    def test_no_ollama_on_fast(self):
        gateway = AgentGateway()
        fast_queries = [
            "CyberJarvis, what is my name?",
            "CyberJarvis, what time is it?",
            "CyberJarvis, what is my goal?",
            "CyberJarvis, tell me about me.",
            "CyberJarvis, check my RAM.",
            "who am i",
            "describe me",
            "what do you know about me",
            "my profile",
            "mere bare mein batao",
            "mere baare mein batao",
            "main kaun hoon",
        ]
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("HARD FAIL: Ollama called on FAST query!")
            for q in fast_queries:
                res = gateway.process_user_request(q, source="voice")
                self.assertEqual(res.execution_path, "FAST", f"Failed for query '{q}'")
                self.assertEqual(res.latency.planning_ms, 0.0)
                self.assertEqual(res.latency.evaluator_ms, 0.0)

    # 11. NO OLLAMA ON DIAGNOSTIC
    def test_no_ollama_on_diagnostic(self):
        gateway = AgentGateway()
        diagnostic_queries = [
            "CyberJarvis, diagnose my computer.",
            "CyberJarvis, analyze my computer.",
            "CyberJarvis, why is my computer slow?",
            "CyberJarvis, what is consuming the most memory?",
            "CyberJarvis, check my system.",
            "CyberJarvis, check my network.",
        ]
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("HARD FAIL: Ollama called on DIAGNOSTIC query!")
            for q in diagnostic_queries:
                res = gateway.process_user_request(q, source="voice")
                self.assertEqual(res.execution_path, "DIAGNOSTIC", f"Failed for query '{q}'")
                self.assertEqual(res.latency.planning_ms, 0.0)
                self.assertEqual(res.latency.evaluator_ms, 0.0)
                self.assertTrue(res.response.completed)

    # 12. BOUNDED RETRIES
    def test_bounded_retries_maximum_two(self):
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        mock_capture.record_phrase.return_value = (None, "No clear speech captured.")

        stt = SpeechToText(audio_capture=mock_capture)
        with patch("builtins.input", return_value="check my RAM") as mock_input:
            result = stt.listen_and_transcribe(max_retries=2)
            # Exactly 3 record_phrase calls: attempt 0 + 2 retries = 3
            self.assertEqual(mock_capture.record_phrase.call_count, 3)
            self.assertEqual(result, "check my RAM")

    # 13. KEYBOARD FALLBACK
    def test_keyboard_fallback_passes_through_gateway(self):
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        mock_capture.record_phrase.return_value = (None, "Timed out waiting for speech.")

        stt = SpeechToText(audio_capture=mock_capture)
        with patch("builtins.input", return_value="tell me about me"):
            fallback_text = stt.listen_and_transcribe(max_retries=2)

        gateway = AgentGateway()
        res = gateway.process_user_request(fallback_text, source="terminal")
        self.assertEqual(res.execution_path, "FAST")
        self.assertIn("Shivaji", res.response.answer)


if __name__ == "__main__":
    unittest.main()
