import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.gateway import AgentGateway, LatencyBreakdown
from agent.fast_path import FastPathHandler
from agent.evaluator import (
    Evaluator,
    contains_hallucinated_code,
    contains_unrelated_devops_lecture,
    is_code_requested,
)
from agent.schemas import AgentPlan, PlanStep, AgentResponse
from tools.schemas import ToolResult
from voice.text_to_speech import TextToSpeech, clean_for_speech
from voice.speech_to_text import SpeechToText
from voice.audio_capture import AudioCapture
from voice.voice_loop import VoiceAssistantLoop, VOICE_EXIT_COMMANDS
from memory import init_db, save_fact


class TestPerformanceAndVoiceDebugging(unittest.TestCase):
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
    # LATENCY & TIMING REPORT TESTS (Stages 1-9 & Largest Source)
    # -------------------------------------------------------------
    def test_latency_breakdown_all_9_stages(self):
        """Verify that all 9 pipeline stages are measured and accessible."""
        lat = LatencyBreakdown(
            mic_capture_ms=800.0,
            vad_ms=300.0,
            stt_ms=2100.0,
            fast_path_ms=2.5,
            planning_ms=1400.0,
            tools_ms=300.0,
            tool_breakdown={"get_cpu_usage": 150.0, "get_memory_usage": 150.0},
            evaluator_ms=1100.0,
            tts_ms=800.0,
            total_ms=5700.0,
        )

        self.assertEqual(lat.mic_capture_ms, 800.0)      # 1. microphone capture
        self.assertEqual(lat.vad_ms, 300.0)              # 2. speech activity detection
        self.assertEqual(lat.stt_ms, 2100.0)             # 3. STT
        self.assertEqual(lat.fast_path_ms, 2.5)          # 4. fast-path routing
        self.assertEqual(lat.planning_ms, 1400.0)        # 5. LLM planning
        self.assertEqual(lat.tools_ms, 300.0)            # 6. each tool execution sum
        self.assertIn("get_cpu_usage", lat.tool_breakdown)
        self.assertEqual(lat.evaluator_ms, 1100.0)       # 7. evaluator
        self.assertEqual(lat.tts_ms, 800.0)              # 8. TTS
        self.assertEqual(lat.total_ms, 5700.0)           # 9. total request latency

    def test_compact_timing_report_format(self):
        """Verify the exact required compact timing report format."""
        lat = LatencyBreakdown(
            stt_ms=2100.0,
            planning_ms=1400.0,
            tools_ms=300.0,
            evaluator_ms=1100.0,
            tts_ms=800.0,
            total_ms=5700.0,
        )
        report = lat.format_timing_report()
        self.assertIn("STT: 2.1s", report)
        self.assertIn("Planner: 1.4s", report)
        self.assertIn("Tools: 0.3s", report)
        self.assertIn("Evaluator: 1.1s", report)
        self.assertIn("TTS: 0.8s", report)
        self.assertIn("Total: 5.7s", report)

    def test_identify_single_largest_source_of_latency(self):
        """Verify identification of the single largest latency stage."""
        # STT largest
        lat_stt = LatencyBreakdown(stt_ms=2100.0, planning_ms=1400.0, tools_ms=300.0, evaluator_ms=1100.0, tts_ms=800.0)
        source, sec = lat_stt.get_largest_source()
        self.assertEqual(source, "STT")
        self.assertAlmostEqual(sec, 2.1, places=1)
        self.assertIn("Largest latency source: STT (2.1s)", lat_stt.format_timing_report())

        # Planner largest
        lat_plan = LatencyBreakdown(stt_ms=500.0, planning_ms=3200.0, tools_ms=100.0, evaluator_ms=800.0, tts_ms=400.0)
        source, sec = lat_plan.get_largest_source()
        self.assertEqual(source, "Planner")
        self.assertAlmostEqual(sec, 3.2, places=1)
        self.assertIn("Largest latency source: Planner (3.2s)", lat_plan.format_timing_report())

    # -------------------------------------------------------------
    # CORRECTNESS RULE 1: Greetings must not trigger long LLM response
    # -------------------------------------------------------------
    def test_rule1_greetings_use_fast_path_no_long_llm(self):
        fp = FastPathHandler()
        for greeting in ["hello", "hi", "hey", "good morning", "good evening", "namaste"]:
            res = fp.try_handle(greeting)
            self.assertIsNotNone(res, f"Greeting '{greeting}' was not handled by fast path")
            self.assertEqual(res.plan.intent, "greeting")
            # Must be concise: 1 short sentence
            self.assertLessEqual(len(res.answer), 80)
            self.assertLessEqual(len(res.spoken_summary), 80)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 2: Deterministic questions must use fast path
    # -------------------------------------------------------------
    def test_rule2_deterministic_questions_use_fast_path(self):
        fp = FastPathHandler()
        queries = [
            "what is my name",
            "what is my goal",
            "what is my project",
            "what time is it",
            "what date is it",
            "where am i",
            "list files",
            "check git status",
            "check my internet",
            "cpu usage",
            "memory usage",
            "disk usage",
            "system uptime",
            "who are you",
        ]
        for q in queries:
            res = fp.try_handle(q)
            self.assertIsNotNone(res, f"Deterministic query '{q}' bypassed fast path unexpectedly!")
            self.assertTrue(res.completed)

    def test_rule2_complex_diagnosis_bypasses_fast_path(self):
        fp = FastPathHandler()
        res = fp.try_handle("Diagnose my computer and tell me why it is slow")
        self.assertIsNone(res)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 3: Tool results are the source of truth
    # -------------------------------------------------------------
    def test_rule3_tool_results_are_source_of_truth(self):
        evaluator = Evaluator()
        diag_output = {
            "verdict": "HEALTHY",
            "summary": "Everything normal",
            "metrics": {"cpu_percent": 14.5, "ram_percent": 42.1, "swap_percent": 0.0, "disk_percent": 35.0, "disk_free_gb": 120.0},
            "issues": [],
            "recommendations": ["System running smoothly"],
            "top_memory_consumers": [{"pid": 100, "name": "python", "memory_percent": 5.2}],
            "top_cpu_consumers": [{"pid": 100, "name": "python", "cpu_percent": 2.1}],
        }
        res = evaluator.evaluate(
            user_input="Diagnose computer",
            plan=AgentPlan(intent="system_diagnosis", goal="Check health"),
            results=[ToolResult(tool_name="diagnose_system_bottleneck", success=True, output=diag_output)],
        )
        self.assertIn("14.5%", res.answer)
        self.assertIn("42.1%", res.answer)
        self.assertIn("120.0 GB free", res.answer)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 4: Never output Bash/Python code unless requested
    # -------------------------------------------------------------
    def test_rule4_never_output_code_unless_requested(self):
        self.assertFalse(is_code_requested("how is my memory usage"))
        self.assertTrue(is_code_requested("give me a bash script to clear cache"))
        self.assertTrue(is_code_requested("write python code to monitor cpu"))

        # Detect forbidden unrequested code patterns
        self.assertTrue(contains_hallucinated_code("#!/bin/bash\nsudo apt update"))
        self.assertTrue(contains_hallucinated_code("```python\nimport psutil\n```"))
        self.assertTrue(contains_hallucinated_code("```bash\ncat /proc/meminfo\n```"))

        # Fallback when LLM produces unsolicited code
        evaluator = Evaluator()
        with patch("ollama.chat", return_value={"message": {"content": "```bash\n#!/bin/bash\nfree -m\n```"}}):
            res = evaluator.evaluate(
                user_input="What is my CPU usage?",
                plan=AgentPlan(intent="check_cpu", goal="Check CPU"),
                results=[ToolResult(tool_name="get_cpu_usage", success=True, output=18.5)],
            )
            # Must reject the shell script and fall back to grounded truth
            self.assertNotIn("#!/bin/bash", res.answer)
            self.assertIn("18.5", res.answer)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 5: Never invent system metrics
    # -------------------------------------------------------------
    def test_rule5_never_invent_system_metrics(self):
        evaluator = Evaluator()
        # Top process output only specifies exact PID and RAM
        procs = [{"pid": 4321, "name": "code", "memory_percent": 18.2}]
        res = evaluator._synthesize_top_process_answer(
            processes=procs,
            user_lower="what is consuming memory",
            plan=AgentPlan(intent="investigate", goal="inspect memory"),
            results=[ToolResult(tool_name="get_top_processes", success=True, output=procs)],
        )
        self.assertIn("4321", res.answer)
        self.assertIn("18.2%", res.answer)
        self.assertIn("code", res.answer)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 6: Never produce unrelated DevOps explanations
    # -------------------------------------------------------------
    def test_rule6_never_produce_unrelated_devops_explanations(self):
        self.assertTrue(contains_unrelated_devops_lecture("In modern DevOps practices, Kubernetes and Prometheus should be used."))
        self.assertFalse(contains_unrelated_devops_lecture("CPU is running at 15%.", user_input="check cpu"))

        evaluator = Evaluator()
        with patch("ollama.chat", return_value={"message": {"content": "You should deploy a Kubernetes cluster with Prometheus and Grafana for monitoring."}}):
            res = evaluator.evaluate(
                user_input="Check system health",
                plan=AgentPlan(intent="check_cpu", goal="Check CPU"),
                results=[ToolResult(tool_name="get_cpu_usage", success=True, output=22.0)],
            )
            self.assertNotIn("Kubernetes cluster", res.answer)
            self.assertIn("22.0", res.answer)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 7: Voice responses must be concise
    # -------------------------------------------------------------
    def test_rule7_voice_responses_concise(self):
        long_text = "Your system is healthy. CPU is at 10%. Here is a third sentence that is too long. And a fourth sentence. And fifth."
        cleaned = clean_for_speech(long_text)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s]
        self.assertLessEqual(len(sentences), 2)
        self.assertLessEqual(len(cleaned), 200)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 8 & 9: STT retry <= 2 then clean fallback, no infinite loop
    # -------------------------------------------------------------
    def test_rule8_and_9_stt_retries_at_most_twice_and_clean_fallback(self):
        mock_capture = MagicMock(spec=AudioCapture)
        mock_capture.is_available.return_value = True
        # Return empty audio chunks (simulating failure / silence)
        mock_capture.record_phrase.return_value = (None, "No audio")

        stt = SpeechToText(audio_capture=mock_capture)
        # Verify it retries at most 2 times (total 3 attempts: initial + 2 retries)
        with patch("builtins.input", return_value="fallback command"):
            res = stt.listen_and_transcribe(max_retries=2)
            self.assertEqual(res, "fallback command")
            self.assertEqual(mock_capture.record_phrase.call_count, 3)

    def test_rule9_voice_loop_terminates_on_repeated_failure(self):
        mock_stt = MagicMock(spec=SpeechToText)
        mock_stt.get_microphone_info.return_value = "Mock Mic"
        # Always return None
        mock_stt.listen_and_transcribe.return_value = None

        mock_tts = MagicMock(spec=TextToSpeech)
        loop = VoiceAssistantLoop()
        loop.stt = mock_stt
        loop.tts = mock_tts

        # Should not loop indefinitely, must exit after 3 consecutive failures
        loop.run()
        self.assertEqual(mock_stt.listen_and_transcribe.call_count, 3)

    # -------------------------------------------------------------
    # CORRECTNESS RULE 10: Clear voice exit command: exit / quit / bye / goodbye
    # -------------------------------------------------------------
    def test_rule10_voice_exit_commands(self):
        for word in ["exit", "quit", "bye", "goodbye"]:
            self.assertIn(word, VOICE_EXIT_COMMANDS)
            fp = FastPathHandler()
            res = fp.try_handle(word)
            self.assertIsNotNone(res)
            self.assertEqual(res.plan.intent, "exit")
            self.assertIn("Goodbye", res.answer)


if __name__ == "__main__":
    unittest.main()
