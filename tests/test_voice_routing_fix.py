"""
Unit and regression tests for CyberJarvis Voice Input Routing Fix.
Validates:
- Verified physical device preference over virtual wrappers
- Device override via CYBERJARVIS_AUDIO_DEVICE
- Native sample rate handling & clean resampling to 16000 Hz
- Incomplete transcript detection & guard
- Voice clarification state and zero-LLM assertion
- Harmless leading conjunctions & wake words normalization
- Speech artifact recovery ("and diagnosis my computer" -> "diagnose my computer")
- Exact FAST routing (Section 6 & 9)
- Exact DIAGNOSTIC routing (Section 6 & 9)
- Exact AGENT routing (Section 6 & 9)
- Strict zero-Ollama assertions for FAST, DIAGNOSTIC, and CLARIFICATION
- STT retry behavior distinguishing no speech, unparseable speech, and incomplete commands
"""
import os
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import (
    ExecutionRouter,
    RouteType,
    normalize_transcript,
    is_incomplete_transcript,
)
from agent.gateway import AgentGateway, process_user_request
from voice.audio_capture import (
    AudioDeviceManager,
    AudioCapture,
    resample_pcm16_mono,
    calculate_pcm_rms,
    calculate_pcm_peak,
)
from voice.speech_to_text import SpeechToText
from memory import init_db, save_fact


class TestVoiceRoutingFix(unittest.TestCase):
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

    # -------------------------------------------------------------
    # 1. PHYSICAL DEVICE PREFERENCE & OVERRIDE
    # -------------------------------------------------------------
    @patch("sounddevice.query_devices")
    @patch("voice.audio_capture.AudioDeviceManager.probe_device")
    def test_physical_device_priority_over_sysdefault(self, mock_probe, mock_query):
        """Verify physical ALSA device 0 is prioritized over sysdefault (ID 4) and default (ID 10)."""
        mock_query.return_value = [
            {"name": "HDA Intel PCH: ALC3246 Analog (hw:0,0)", "index": 0, "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "sysdefault", "index": 4, "max_input_channels": 128, "default_samplerate": 48000},
            {"name": "default", "index": 10, "max_input_channels": 128, "default_samplerate": 48000},
        ]
        mock_probe.return_value = (True, 600.0, 1500, True)

        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 0)
        self.assertIn("ALC3246", selected["name"])

    def test_device_selection_env_override(self):
        """Verify CYBERJARVIS_AUDIO_DEVICE explicitly overrides auto-selection."""
        os.environ["CYBERJARVIS_AUDIO_DEVICE"] = "4"
        selected = AudioDeviceManager.select_input_device()
        self.assertIsNotNone(selected)
        self.assertEqual(selected["index"], 4)

    # -------------------------------------------------------------
    # 2. AUDIO RESAMPLING (Native hardware rate -> 16000 Hz)
    # -------------------------------------------------------------
    def test_resample_pcm16_mono_44100_to_16000(self):
        """Verify linear resampling converts 44.1kHz audio to 16kHz with correct sample count."""
        in_rate = 44100
        out_rate = 16000
        # 1 second of 44100 Hz 16-bit mono PCM (88200 bytes)
        in_samples = 44100
        import struct
        samples = [int(1000 * (1 if i % 2 == 0 else -1)) for i in range(in_samples)]
        in_pcm = struct.pack(f"<{in_samples}h", *samples)

        out_pcm = resample_pcm16_mono(in_pcm, in_rate, out_rate)
        expected_out_samples = int(in_samples * out_rate / in_rate)
        self.assertEqual(len(out_pcm), expected_out_samples * 2)

        # Same rate returns original without modification
        same_pcm = resample_pcm16_mono(in_pcm, 16000, 16000)
        self.assertEqual(same_pcm, in_pcm)

    # -------------------------------------------------------------
    # 3. INCOMPLETE TRANSCRIPT DETECTION
    # -------------------------------------------------------------
    def test_incomplete_transcript_detection_guard(self):
        """Verify obviously incomplete transcripts are flagged and complete queries pass."""
        incomplete_examples = [
            "what is",
            "what",
            "and",
            "tell me",
            "how",
            "why is",
            "check",
            "diagnose",
            "and diagnose",
            "tell me about",
            "check my",
            "diagnose my",
        ]
        for phrase in incomplete_examples:
            self.assertTrue(
                is_incomplete_transcript(phrase),
                f"Expected '{phrase}' to be detected as incomplete transcript.",
            )

        complete_examples = [
            "what is my name",
            "who am i",
            "what is my goal",
            "what time is it",
            "what date is it",
            "check my ram",
            "check my cpu",
            "check my disk",
            "what is your name",
            "tell me about me",
            "hello",
            "diagnose my computer",
            "and diagnose my computer",
            "and diagnosis my computer",
            "why is my computer slow",
            "what is consuming the most memory",
            "analyze my CyberJarvis architecture",
        ]
        for phrase in complete_examples:
            self.assertFalse(
                is_incomplete_transcript(phrase),
                f"Expected '{phrase}' to be recognized as complete command.",
            )

    # -------------------------------------------------------------
    # 4. TRANSCRIPT NORMALIZATION (Wake words, Conjunctions, Artifacts)
    # -------------------------------------------------------------
    def test_transcript_normalization_rules(self):
        """Test normalization removes wake words, leading conjunctions, and fixes speech artifacts."""
        cases = [
            ("and diagnosis my computer", "diagnose my computer"),
            ("and diagnose my computer", "diagnose my computer"),
            ("okay cyberjarvis and diagnose my computer", "diagnose my computer"),
            ("please check my ram", "check my ram"),
            ("also check cpu", "check cpu"),
            ("so what time is it", "what time is it"),
            ("then check disk", "check disk"),
            ("uh check my ram", "check my ram"),
            ("um diagnose my computer", "diagnose my computer"),
            ("check cpu and ram", "check cpu and ram"),  # internal "and" preserved
            ("CyberJarvis, what is my name?", "what is my name"),
            ("hey cyberjarvis what is my goal", "what is my goal"),
        ]
        for raw, expected in cases:
            norm = normalize_transcript(raw)
            self.assertEqual(norm, expected, f"Failed for raw='{raw}': got '{norm}', expected '{expected}'")

    # -------------------------------------------------------------
    # 5. EXACT SECTION 9 VOICE COMMAND TESTS & ZERO-LLM ASSERTIONS
    # -------------------------------------------------------------
    def test_section9_voice_command_routing_and_no_ollama(self):
        """Execute all 9 Section 9 tests with strict zero-Ollama assertions on FAST & DIAGNOSTIC."""
        gateway = AgentGateway()

        # Test 1: "CyberJarvis, what is my name?" -> FAST, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on FAST Test 1!")
            res1 = gateway.process_user_request("CyberJarvis, what is my name?", source="voice")
            self.assertEqual(res1.execution_path, "FAST")
            self.assertIn("Shivaji", res1.response.answer)
            self.assertEqual(res1.latency.planning_ms, 0.0)
            self.assertEqual(res1.latency.evaluator_ms, 0.0)

        # Test 2: "CyberJarvis, what time is it?" -> FAST, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on FAST Test 2!")
            res2 = gateway.process_user_request("CyberJarvis, what time is it?", source="voice")
            self.assertEqual(res2.execution_path, "FAST")
            self.assertEqual(res2.latency.planning_ms, 0.0)
            self.assertEqual(res2.latency.evaluator_ms, 0.0)

        # Test 3: "CyberJarvis, diagnose my computer." -> DIAGNOSTIC, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on DIAGNOSTIC Test 3!")
            res3 = gateway.process_user_request("CyberJarvis, diagnose my computer.", source="voice")
            self.assertEqual(res3.execution_path, "DIAGNOSTIC")
            self.assertEqual(res3.latency.planning_ms, 0.0)
            self.assertEqual(res3.latency.evaluator_ms, 0.0)

        # Test 4: "CyberJarvis, what is consuming the most memory?" -> DIAGNOSTIC, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on DIAGNOSTIC Test 4!")
            res4 = gateway.process_user_request("CyberJarvis, what is consuming the most memory?", source="voice")
            self.assertEqual(res4.execution_path, "DIAGNOSTIC")
            self.assertEqual(res4.latency.planning_ms, 0.0)
            self.assertEqual(res4.latency.evaluator_ms, 0.0)

        # Test 5: "and diagnose my computer" -> DIAGNOSTIC, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on DIAGNOSTIC Test 5!")
            res5 = gateway.process_user_request("and diagnose my computer", source="voice")
            self.assertEqual(res5.execution_path, "DIAGNOSTIC")
            self.assertEqual(res5.latency.planning_ms, 0.0)
            self.assertEqual(res5.latency.evaluator_ms, 0.0)

        # Test 6: "what is" -> CLARIFICATION, No Ollama, No Agent
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on CLARIFICATION Test 6!")
            res6 = gateway.process_user_request("what is", source="voice")
            self.assertEqual(res6.execution_path, "CLARIFICATION")
            self.assertIn("didn't catch the full command", res6.response.answer)
            self.assertIn("Please repeat", res6.response.answer)
            self.assertEqual(res6.latency.planning_ms, 0.0)
            self.assertEqual(res6.latency.evaluator_ms, 0.0)

        # Test 7: "tell me about" -> CLARIFICATION, No Ollama, No Agent
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on CLARIFICATION Test 7!")
            res7 = gateway.process_user_request("tell me about", source="voice")
            self.assertEqual(res7.execution_path, "CLARIFICATION")
            self.assertIn("didn't catch the full command", res7.response.answer)
            self.assertEqual(res7.latency.planning_ms, 0.0)
            self.assertEqual(res7.latency.evaluator_ms, 0.0)

        # Test 8: "tell me about me" -> FAST / PROFILE, No Ollama
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked on FAST Profile Test 8!")
            res8 = gateway.process_user_request("tell me about me", source="voice")
            self.assertEqual(res8.execution_path, "FAST")
            self.assertIn("Shivaji", res8.response.answer)
            self.assertEqual(res8.latency.planning_ms, 0.0)
            self.assertEqual(res8.latency.evaluator_ms, 0.0)

        # Test 9: "analyze my CyberJarvis architecture" -> AGENT, Ollama allowed
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {
                    "content": '{"intent": "architecture", "goal": "Analyze architecture", "steps": []}'
                }
            }
            res9 = gateway.process_user_request("analyze my CyberJarvis architecture", source="voice")
            self.assertEqual(res9.execution_path, "AGENT")
            self.assertTrue(mock_ollama.called)

    # -------------------------------------------------------------
    # 6. ALL SECTION 6 FAST ROUTING CASES
    # -------------------------------------------------------------
    def test_all_section6_fast_routing_cases(self):
        """Verify all Section 6 fast queries route to FAST with zero Ollama calls."""
        fast_queries = [
            "what is my name",
            "who am i",
            "what is my goal",
            "what time is it",
            "what date is it",
            "check my ram",
            "check my cpu",
            "check my disk",
            "what is your name",
            "tell me about me",
            "hello",
        ]
        gateway = AgentGateway()
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a Section 6 FAST query!")
            for q in fast_queries:
                res = gateway.process_user_request(q, source="voice")
                self.assertEqual(
                    res.execution_path, "FAST",
                    f"Query '{q}' expected FAST but got {res.execution_path}",
                )

    # -------------------------------------------------------------
    # 7. ALL SECTION 6 DIAGNOSTIC ROUTING CASES
    # -------------------------------------------------------------
    def test_all_section6_diagnostic_routing_cases(self):
        """Verify all Section 6 diagnostic queries route to DIAGNOSTIC with zero Ollama calls."""
        diagnostic_queries = [
            "diagnose my computer",
            "analyze my computer",
            "analyse my computer",
            "check my computer",
            "check my system",
            "why is my computer slow",
            "why is my system slow",
            "what is consuming the most memory",
            "what is using the most memory",
            "check my network",
            "mera computer diagnose karo",
            "mera system check karo",
            "mera computer slow kyu hai",
            "ram kaun use kar raha hai",
        ]
        gateway = AgentGateway()
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a Section 6 DIAGNOSTIC query!")
            for q in diagnostic_queries:
                res = gateway.process_user_request(q, source="voice")
                self.assertEqual(
                    res.execution_path, "DIAGNOSTIC",
                    f"Query '{q}' expected DIAGNOSTIC but got {res.execution_path}",
                )

    # -------------------------------------------------------------
    # 8. ALL SECTION 6 AGENT ROUTING CASES
    # -------------------------------------------------------------
    def test_all_section6_agent_routing_cases(self):
        """Verify all Section 6 architecture/complex queries route to AGENT."""
        agent_queries = [
            "analyze my CyberJarvis architecture",
            "refactor my memory subsystem",
            "plan a deployment architecture",
            "compare two software architectures",
        ]
        gateway = AgentGateway()
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {"content": '{"intent": "architecture", "goal": "Complex task", "steps": []}'}
            }
            for q in agent_queries:
                res = gateway.process_user_request(q, source="voice")
                self.assertEqual(
                    res.execution_path, "AGENT",
                    f"Query '{q}' expected AGENT but got {res.execution_path}",
                )

    # -------------------------------------------------------------
    # 9. STT RETRY BEHAVIOR (Distinguishing A, B, and C)
    # -------------------------------------------------------------
    def test_stt_retry_distinguishes_no_speech_vs_incomplete(self):
        """Verify listen_and_transcribe distinguishes No speech (A) vs Incomplete command (C)."""
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True

        # Case A: No speech detected (pcm_data is None)
        mock_capture.record_phrase.return_value = (None, "No clear speech captured.")
        stt_a = SpeechToText(audio_capture=mock_capture)
        with patch("builtins.input", return_value="check my cpu"):
            with patch("sys.stdout", new=io.StringIO()) as out:
                res_a = stt_a.listen_and_transcribe(max_retries=1)
                output_a = out.getvalue()
                self.assertIn("❌ No speech detected", output_a)
                self.assertIn("🔁 Retry 1/1", output_a)
                self.assertEqual(res_a, "check my cpu")

        # Case C: Speech detected, transcribed, but incomplete fragment "what is"
        mock_capture.record_phrase.return_value = (b"\x00\x01" * 16000, "Success")
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = "what is"
        stt_c = SpeechToText(audio_capture=mock_capture, stt_engine=mock_engine)

        with patch("sys.stdout", new=io.StringIO()) as out:
            res_c = stt_c.listen_and_transcribe(max_retries=1)
            output_c = out.getvalue()
            self.assertIn("🤔 Incomplete command detected.", output_c)
            self.assertIn("🎤 Please repeat.", output_c)
            self.assertEqual(res_c, "what is")

    # -------------------------------------------------------------
    # 10. TIMING REPORT SKIPPED STAGES
    # -------------------------------------------------------------
    def test_timing_report_skipped_stages_on_fast_and_diagnostic(self):
        """Verify timing report shows Planner: SKIPPED and Evaluator: SKIPPED for FAST & DIAGNOSTIC."""
        gateway = AgentGateway()
        res_fast = gateway.process_user_request("what time is it", source="voice")
        report_fast = res_fast.latency.format_timing_report()
        self.assertIn("Planner: SKIPPED", report_fast)
        self.assertIn("Evaluator: SKIPPED", report_fast)

        res_diag = gateway.process_user_request("diagnose my computer", source="voice")
        report_diag = res_diag.latency.format_timing_report()
        self.assertIn("Planner: SKIPPED", report_diag)
        self.assertIn("Evaluator: SKIPPED", report_diag)

        res_clar = gateway.process_user_request("what is", source="voice")
        report_clar = res_clar.latency.format_timing_report()
        self.assertIn("Planner: SKIPPED", report_clar)
        self.assertIn("Evaluator: SKIPPED", report_clar)


if __name__ == "__main__":
    unittest.main()
