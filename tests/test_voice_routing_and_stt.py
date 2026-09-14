"""
Dedicated regression & unit tests for:
- Wake-word normalization (lowercase, whitespace collapse, wake word prefix removal, punctuation)
- Voice transcript routing through the single authoritative execution entry point
- Fast path voice routing (Section 10 & 3)
- Diagnostic voice routing (Section 10 & 4)
- Complex reasoning routing to True Agent Path
- Strict no-Ollama assertions for FAST and DIAGNOSTIC voice queries
- STT retry limit (maximum 2 retries) and bounded timeout handling
- Clean keyboard fallback when voice recognition is unavailable
- Hinglish command routing for diagnostics and time
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import ExecutionRouter, RouteType, normalize_transcript
from agent.gateway import AgentGateway, process_user_request
from voice.speech_to_text import SpeechToText
from voice.audio_capture import AudioCapture
from memory import init_db, save_fact


class TestVoiceRoutingAndSTT(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        os.environ["CYBERJARVIS_DB_PATH"] = self.tmp_db.name
        init_db()
        save_fact("My name is Shivaji")
        save_fact("My project is CyberJarvis")
        save_fact("My goal is to build AI agents")

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
        os.environ.pop("CYBERJARVIS_DB_PATH", None)

    # -------------------------------------------------------------
    # 1. WAKE-WORD & TRANSCRIPT NORMALIZATION TESTS
    # -------------------------------------------------------------
    def test_wake_word_normalization_cases(self):
        cases = [
            ("CyberJarvis, diagnose my computer", "diagnose my computer"),
            ("hey cyberjarvis check my ram", "check my ram"),
            ("okay cyberjarvis what is my name", "what is my name"),
            ("CyberJarvis what is my name", "what is my name"),
            ("CyberJarvis, check my RAM.", "check my ram"),
            ("CyberJarvis, what time is it?", "what time is it"),
            ("CyberJarvis, what is consuming the most memory?", "what is consuming the most memory"),
            ("CyberJarvis, analyze my project architecture.", "analyze my project architecture"),
            ("   OKAY   CYBERJARVIS   Check   My   Disk!   ", "check my disk"),
            ("hey jarvis, what date is it?", "what date is it"),
            ("cyber jarvis, why is my computer slow", "why is my computer slow"),
        ]
        for raw, expected in cases:
            norm = normalize_transcript(raw)
            self.assertEqual(norm, expected, f"Failed for raw='{raw}': got '{norm}', expected '{expected}'")

    # -------------------------------------------------------------
    # 2. EXACT SECTION 10 TERMINAL TESTS
    # -------------------------------------------------------------
    def test_section10_exact_terminal_cases(self):
        gateway = AgentGateway()

        terminal_cases = [
            ("hello", "FAST"),
            ("what is my name", "FAST"),
            ("what time is it", "FAST"),
            ("check my RAM", "FAST"),
            ("diagnose my computer", "DIAGNOSTIC"),
            ("analyse my computer", "DIAGNOSTIC"),
            ("why is my computer slow", "DIAGNOSTIC"),
            ("what is consuming the most memory", "DIAGNOSTIC"),
        ]

        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a FAST/DIAGNOSTIC command!")
            for query, expected_path in terminal_cases:
                res = gateway.process_user_request(query, source="terminal")
                self.assertEqual(
                    res.execution_path,
                    expected_path,
                    f"Query '{query}' expected {expected_path} but got {res.execution_path}",
                )

        # Complex agent request
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {
                    "content": '{"intent": "architecture", "goal": "Analyze architecture", "steps": []}'
                }
            }
            res_complex = gateway.process_user_request(
                "Analyze my project architecture and suggest improvements", source="terminal"
            )
            self.assertEqual(res_complex.execution_path, "AGENT")
            self.assertTrue(mock_ollama.called)

    # -------------------------------------------------------------
    # 3. EXACT SECTION 10 VOICE TESTS
    # -------------------------------------------------------------
    def test_section10_exact_voice_cases(self):
        gateway = AgentGateway()

        voice_cases = [
            ("CyberJarvis, what is my name?", "FAST"),
            ("CyberJarvis, what time is it?", "FAST"),
            ("CyberJarvis, check my RAM.", "FAST"),
            ("CyberJarvis, diagnose my computer.", "DIAGNOSTIC"),
            ("CyberJarvis, what is consuming the most memory?", "DIAGNOSTIC"),
        ]

        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a FAST/DIAGNOSTIC voice command!")
            for query, expected_path in voice_cases:
                res = gateway.process_user_request(query, source="voice")
                self.assertEqual(
                    res.execution_path,
                    expected_path,
                    f"Voice query '{query}' expected {expected_path} but got {res.execution_path}",
                )
                self.assertEqual(res.latency.planning_ms, 0.0)
                self.assertEqual(res.latency.evaluator_ms, 0.0)

        # Complex voice agent request
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.return_value = {
                "message": {
                    "content": '{"intent": "architecture", "goal": "Analyze project architecture", "steps": []}'
                }
            }
            res_agent = gateway.process_user_request(
                "CyberJarvis, analyze my project architecture.", source="voice"
            )
            self.assertEqual(res_agent.execution_path, "AGENT")
            self.assertTrue(mock_ollama.called)

    # -------------------------------------------------------------
    # 4. HINGLISH ROUTING TESTS (Section 4 & 5)
    # -------------------------------------------------------------
    def test_hinglish_diagnostic_routing(self):
        gateway = AgentGateway()
        hinglish_cases = [
            ("mera computer diagnose karo", "DIAGNOSTIC"),
            ("mera system check karo", "DIAGNOSTIC"),
            ("mera computer slow kyu hai", "DIAGNOSTIC"),
            ("mere system ko analyse karo", "DIAGNOSTIC"),
            ("ram kaun use kar raha hai", "DIAGNOSTIC"),
            ("mera internet check karo", "DIAGNOSTIC"),
        ]
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a Hinglish diagnostic command!")
            for query, expected_path in hinglish_cases:
                res = gateway.process_user_request(query, source="voice")
                self.assertEqual(
                    res.execution_path,
                    expected_path,
                    f"Hinglish query '{query}' expected {expected_path} but got {res.execution_path}",
                )

    def test_section5_short_deterministic_time_requests(self):
        gateway = AgentGateway()
        time_cases = [
            "time which is my computer",
            "what is my computer time",
            "tell me the time",
            "computer ka time kya hai",
            "kya time hua hai",
            "time batao",
            "samay kya hai",
        ]
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama was invoked for a short time query!")
            for query in time_cases:
                res = gateway.process_user_request(query, source="terminal")
                self.assertEqual(res.execution_path, "FAST", f"Time query '{query}' did not route to FAST")
                self.assertIn("time", res.response.answer.lower())

    # -------------------------------------------------------------
    # 5. HARD PERFORMANCE REQUIREMENT (Zero Ollama calls)
    # -------------------------------------------------------------
    def test_hard_performance_requirement_zero_ollama(self):
        gateway = AgentGateway()
        strict_queries = [
            "What is my name?",
            "What time is it?",
            "Check my RAM.",
            "Diagnose my computer.",
        ]
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("HARD REQUIREMENT VIOLATION: Ollama called!")
            for q in strict_queries:
                res = gateway.process_user_request(q, source="voice")
                report = res.latency.format_timing_report()
                self.assertIn("Planner: SKIPPED", report)
                self.assertIn("Evaluator: SKIPPED", report)
                self.assertIn(res.execution_path, ["FAST", "DIAGNOSTIC"])

    # -------------------------------------------------------------
    # 6. STT RETRY LIMIT & TIMEOUT & VOICE FALLBACK (Section 6)
    # -------------------------------------------------------------
    def test_stt_retry_limit_and_voice_fallback(self):
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        # Return None simulating no speech detected / silence
        mock_capture.record_phrase.return_value = (None, "Timed out waiting for speech")

        stt = SpeechToText(audio_capture=mock_capture)

        with patch("builtins.input", return_value="diagnose my computer") as mock_input:
            result = stt.listen_and_transcribe(timeout=3.5, max_retries=2)
            # Must retry at most 2 times (attempt 0 + 2 retries = 3 calls)
            self.assertEqual(mock_capture.record_phrase.call_count, 3)
            # Returns the fallback text entered via keyboard
            self.assertEqual(result, "diagnose my computer")
            self.assertTrue(mock_input.called)

    def test_single_authoritative_gateway_entry(self):
        """Verify process_user_request is the authoritative entry for both terminal and voice."""
        res_terminal = process_user_request("what is my name", source="terminal")
        self.assertEqual(res_terminal.execution_path, "FAST")

        res_voice = process_user_request("CyberJarvis, what is my name?", source="voice")
        self.assertEqual(res_voice.execution_path, "FAST")


if __name__ == "__main__":
    unittest.main()
