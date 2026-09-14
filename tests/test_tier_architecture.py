"""
Comprehensive test suite for the Three-Tier Execution Architecture in CyberJarvis:
Tier 1: Fast Path (Deterministic queries)
Tier 2: Diagnostic Path (Parallel execution, deterministic analysis, zero LLM)
Tier 3: True Agent Path (LLM Planning & Evaluation)

Tests cover:
- Routing classifier
- Fast path bypass
- Diagnostic path bypass (Ollama planner and evaluator bypass verification)
- Deterministic diagnostic analyzer threshold logic & structured report schema
- Parallel tool execution verification
- Hallucination and code prevention
- Latency instrumentation and timing report formatting
- Voice routing and concise spoken summaries
"""
import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import ExecutionRouter, RouteType
from agent.fast_path import FastPathHandler
from agent.diagnostic_analyzer import DiagnosticAnalyzer, DiagnosticReport
from agent.diagnostic_path import DiagnosticPathHandler
from agent.gateway import AgentGateway, LatencyBreakdown, GatewayResult
from agent.schemas import AgentPlan, PlanStep
from tools.schemas import ToolResult
from memory import init_db, save_fact


class TestThreeTierArchitecture(unittest.TestCase):
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
    # 1. ROUTING CLASSIFIER TESTS
    # -------------------------------------------------------------
    def test_routing_fast_path_queries(self):
        router = ExecutionRouter()
        fast_queries = [
            "hello",
            "hi jarvis",
            "exit",
            "quit",
            "what is my name",
            "what is my goal",
            "what project am i working on",
            "what time is it",
            "what date is it",
            "current directory",
            "list files",
            "git status",
            "check my internet",
            "cpu usage",
            "check my ram",
            "memory usage",
            "disk usage",
            "uptime",
            "who are you",
        ]
        for q in fast_queries:
            decision = router.classify(q)
            self.assertEqual(decision, RouteType.FAST, f"Expected FAST for query: '{q}', got {decision}")

    def test_routing_diagnostic_path_queries(self):
        router = ExecutionRouter()
        diagnostic_queries = [
            "diagnose my computer",
            "why is my computer slow?",
            "check CPU, RAM, disk and network",
            "find the main system bottleneck",
            "check my system health",
            "what is consuming the most memory?",
            "what is consuming the most cpu?",
            "run system diagnosis",
            "system health",
            "troubleshoot my computer",
            "CyberJarvis, diagnose my computer and tell me why it is slow.",
            "CyberJarvis, what is consuming the most memory?",
        ]
        for q in diagnostic_queries:
            decision = router.classify(q)
            self.assertEqual(decision, RouteType.DIAGNOSTIC, f"Expected DIAGNOSTIC for query: '{q}', got {decision}")

    def test_routing_agent_path_queries(self):
        router = ExecutionRouter()
        agent_queries = [
            "Analyze my GitHub repository and suggest architectural improvements.",
            "Plan a multi-step deployment to AWS ECS.",
            "Compare monolithic and microservices architectures for our stack.",
            "Research this unknown database deadlock using available diagnostic logs.",
        ]
        for q in agent_queries:
            decision = router.classify(q)
            self.assertEqual(decision, RouteType.AGENT, f"Expected AGENT for query: '{q}', got {decision}")

    def test_wake_word_stripping(self):
        self.assertEqual(ExecutionRouter.strip_wake_words("CyberJarvis, what time is it?"), "what time is it?")
        self.assertEqual(ExecutionRouter.strip_wake_words("Hey Jarvis: diagnose computer"), "diagnose computer")
        self.assertEqual(ExecutionRouter.strip_wake_words("Jarvis check my ram"), "check my ram")

    # -------------------------------------------------------------
    # 2. FAST PATH BYPASS TESTS
    # -------------------------------------------------------------
    def test_fast_path_bypasses_llm_for_deterministic_requests(self):
        fp = FastPathHandler()
        # Hello
        res_hello = fp.try_handle("hello")
        self.assertIsNotNone(res_hello)
        self.assertEqual(res_hello.plan.intent, "greeting")

        # Name
        res_name = fp.try_handle("what is my name?")
        self.assertIsNotNone(res_name)
        self.assertIn("Shivaji", res_name.answer)

        # Time
        res_time = fp.try_handle("what time is it?")
        self.assertIsNotNone(res_time)
        self.assertIn("current system time", res_time.answer.lower())

        # RAM
        res_ram = fp.try_handle("check my RAM")
        self.assertIsNotNone(res_ram)
        self.assertIn("RAM usage", res_ram.answer)

    def test_fast_path_returns_none_on_diagnostic_queries(self):
        fp = FastPathHandler()
        # Must return None so router directs it to Tier 2 Diagnostic Path
        self.assertIsNone(fp.try_handle("diagnose my computer"))
        self.assertIsNone(fp.try_handle("why is my computer slow?"))
        self.assertIsNone(fp.try_handle("what is consuming the most memory?"))
        self.assertIsNone(fp.try_handle("find the main system bottleneck"))

    # -------------------------------------------------------------
    # 3. DIAGNOSTIC PATH BYPASS TESTS (OLLAMA NEVER CALLED)
    # -------------------------------------------------------------
    @patch("ollama.chat")
    def test_diagnostic_path_completely_bypasses_ollama_planner_and_evaluator(self, mock_ollama):
        """CRITICAL: Standard system diagnostics MUST NOT invoke Ollama at all."""
        mock_ollama.side_effect = AssertionError("Ollama LLM was invoked for a deterministic diagnostic task!")

        gateway = AgentGateway()

        queries = [
            "diagnose my computer",
            "why is my computer slow?",
            "check CPU, RAM, disk and network",
            "find the main system bottleneck",
            "check my system health",
            "what is consuming the most memory?",
        ]

        for q in queries:
            result = gateway.process_command(q, source="test")
            self.assertEqual(result.execution_path, "DIAGNOSTIC")
            self.assertEqual(result.route, "diagnostic_path")
            self.assertEqual(result.latency.planning_ms, 0.0)
            self.assertEqual(result.latency.evaluator_ms, 0.0)
            self.assertTrue(result.response.completed)
            self.assertIsNotNone(result.response.diagnostic_report)
            self.assertGreater(len(result.response.answer), 20)
            self.assertGreater(len(result.response.spoken_summary), 10)

        # Confirm Ollama was never called
        mock_ollama.assert_not_called()

    # -------------------------------------------------------------
    # 4. DETERMINISTIC DIAGNOSTIC ANALYZER TESTS
    # -------------------------------------------------------------
    def test_diagnostic_analyzer_healthy_system(self):
        analyzer = DiagnosticAnalyzer()
        results = [
            ToolResult(tool_name="get_cpu_usage", success=True, output=17.0),
            ToolResult(tool_name="get_memory_usage", success=True, output={"percent": 41.0, "used_gb": 6.5, "total_gb": 16.0}),
            ToolResult(tool_name="get_swap_usage", success=True, output={"percent": 0.0, "used_gb": 0.0}),
            ToolResult(tool_name="get_disk_usage", success=True, output={"percent": 27.0, "free_gb": 120.0}),
            ToolResult(tool_name="get_load_average", success=True, output={"load_1m": 0.85, "cpu_cores": 4}),
            ToolResult(tool_name="check_network_connectivity", success=True, output={"connected": True, "target": "1.1.1.1:53"}),
            ToolResult(tool_name="get_top_processes", success=True, output=[
                {"pid": 101, "name": "code", "memory_percent": 4.5, "cpu_percent": 2.0},
            ]),
        ]
        report, display = analyzer.analyze(results, "diagnose my computer")
        self.assertEqual(report.status, "healthy")
        self.assertEqual(report.primary_issue, "none")
        self.assertIn("healthy", report.spoken_summary.lower())
        self.assertIn("normal", report.spoken_summary.lower())
        self.assertIn("17.0%", display)
        self.assertIn("41.0%", display)

    def test_diagnostic_analyzer_critical_ram(self):
        analyzer = DiagnosticAnalyzer()
        results = [
            ToolResult(tool_name="get_cpu_usage", success=True, output=22.0),
            ToolResult(tool_name="get_memory_usage", success=True, output={"percent": 91.5, "used_gb": 14.6, "total_gb": 16.0}),
            ToolResult(tool_name="get_swap_usage", success=True, output={"percent": 65.0, "used_gb": 1.3}),
            ToolResult(tool_name="get_disk_usage", success=True, output={"percent": 30.0, "free_gb": 100.0}),
            ToolResult(tool_name="get_top_processes", success=True, output=[
                {"pid": 2048, "name": "chrome", "memory_percent": 28.5, "cpu_percent": 3.0},
            ]),
        ]
        report, display = analyzer.analyze(results, "why is my computer slow?")
        self.assertEqual(report.status, "critical")
        self.assertIn("RAM", report.primary_issue)
        self.assertIn("chrome", report.primary_issue)
        self.assertIn("chrome", report.spoken_summary)
        self.assertTrue(any("closing" in r.lower() or "chrome" in r.lower() for r in report.recommendations))

    def test_diagnostic_analyzer_critical_cpu(self):
        analyzer = DiagnosticAnalyzer()
        results = [
            ToolResult(tool_name="get_cpu_usage", success=True, output=94.0),
            ToolResult(tool_name="get_memory_usage", success=True, output={"percent": 45.0, "used_gb": 7.2, "total_gb": 16.0}),
            ToolResult(tool_name="get_top_processes", success=True, output=[
                {"pid": 3099, "name": "ffmpeg", "memory_percent": 2.0, "cpu_percent": 75.0},
            ]),
        ]
        report, display = analyzer.analyze(results, "diagnose my computer")
        self.assertEqual(report.status, "critical")
        self.assertIn("CPU", report.primary_issue)
        self.assertIn("ffmpeg", report.primary_issue)

    def test_diagnostic_analyzer_low_disk(self):
        analyzer = DiagnosticAnalyzer()
        results = [
            ToolResult(tool_name="get_cpu_usage", success=True, output=15.0),
            ToolResult(tool_name="get_memory_usage", success=True, output={"percent": 40.0, "used_gb": 6.0, "total_gb": 16.0}),
            ToolResult(tool_name="get_disk_usage", success=True, output={"percent": 94.0, "free_gb": 3.2}),
        ]
        report, display = analyzer.analyze(results, "diagnose my computer")
        self.assertEqual(report.status, "critical")
        self.assertIn("disk", report.primary_issue.lower())

    def test_diagnostic_analyzer_process_specific_query(self):
        analyzer = DiagnosticAnalyzer()
        results = [
            ToolResult(tool_name="get_top_processes", success=True, output=[
                {"pid": 4120, "name": "code", "memory_percent": 18.2, "cpu_percent": 2.1},
                {"pid": 4125, "name": "chrome", "memory_percent": 12.0, "cpu_percent": 1.5},
            ]),
            ToolResult(tool_name="get_memory_usage", success=True, output={"percent": 65.0, "used_gb": 10.4, "total_gb": 16.0}),
        ]
        report, display = analyzer.analyze(results, "what is consuming the most memory?")
        self.assertIn("code", display)
        self.assertIn("18.2%", display)
        self.assertIn("4120", display)
        self.assertEqual(report.spoken_summary, "Among active processes, code is consuming the most memory at 18.2 percent.")
        # Must NOT contain code blocks
        self.assertNotIn("```", display)
        self.assertNotIn("#!/bin/bash", display)

    def test_diagnostic_report_schema_compliance(self):
        """Verify report conforms to required DiagnosticReport structure."""
        report = DiagnosticReport(
            status="healthy",
            observations=["CPU: 15%"],
            findings=["Normal"],
            primary_issue="none",
            recommendations=["No action needed"],
            spoken_summary="System is healthy.",
        )
        data = report.model_dump()
        for required_key in ["status", "observations", "findings", "primary_issue", "recommendations", "spoken_summary"]:
            self.assertIn(required_key, data)

    # -------------------------------------------------------------
    # 5. PARALLEL EXECUTION CONCURRENCY TESTS
    # -------------------------------------------------------------
    def test_parallel_execution_measures_time_saving(self):
        """Verify independent diagnostic tools run concurrently and save time."""
        handler = DiagnosticPathHandler()

        # Mock tool execution to simulate I/O delays
        with patch("tools.registry.registry.execute") as mock_exec:
            def fake_execute(tool_name, parameters=None, policy=None, confirmation_handler=None):
                time.sleep(0.04)  # 40ms per tool
                return ToolResult(tool_name=tool_name, success=True, output=10.0, duration_ms=40.0)

            mock_exec.side_effect = fake_execute

            response, parallel_ms, seq_ms, analysis_ms = handler.handle("diagnose my computer")

            # With 9 tools, sequential time is 9 * 40ms = 360ms
            # With parallel execution across up to 8 threads, wall clock time is ~80-120ms
            self.assertGreater(seq_ms, parallel_ms, "Parallel execution was not faster than sequential sum!")
            self.assertGreaterEqual(len(response.tool_results), 8)

    # -------------------------------------------------------------
    # 6. HALLUCINATION & CODE FILTERING
    # -------------------------------------------------------------
    def test_diagnostic_output_never_outputs_code_or_bash(self):
        gateway = AgentGateway()
        res = gateway.process_command("why is my computer slow?", source="test")
        forbidden = ["#!/bin/bash", "cat /proc", "ps aux |", "free -m &&", "sudo apt", "import psutil"]
        for f in forbidden:
            self.assertNotIn(f, res.response.answer)
            if res.response.spoken_summary:
                self.assertNotIn(f, res.response.spoken_summary)

    # -------------------------------------------------------------
    # 7. LATENCY INSTRUMENTATION TESTS
    # -------------------------------------------------------------
    def test_latency_instrumentation_all_stages(self):
        lat = LatencyBreakdown(
            stt_ms=100.0,
            router_ms=2.0,
            fast_path_ms=0.0,
            planning_ms=0.0,
            tools_ms=500.0,
            parallel_tools_ms=500.0,
            sequential_tools_ms=650.0,
            analysis_ms=1.5,
            evaluator_ms=0.0,
            tts_ms=80.0,
            total_ms=683.5,
            execution_path="DIAGNOSTIC",
        )
        report = lat.format_timing_report()
        self.assertIn("STT: 0.1s", report)
        self.assertIn("Router:", report)
        self.assertIn("Planner: SKIPPED", report)
        self.assertIn("Tools: 0.5s", report)
        self.assertIn("Analysis:", report)
        self.assertIn("Evaluator: SKIPPED", report)
        self.assertIn("TTS: 0.1s", report)
        self.assertIn("Total: 0.7s", report)

    # -------------------------------------------------------------
    # 8. VOICE ROUTING & CONCISE SPOKEN SUMMARY
    # -------------------------------------------------------------
    def test_voice_spoken_summary_conciseness(self):
        gateway = AgentGateway()
        res = gateway.process_command("diagnose my computer", source="voice")
        spoken = res.response.spoken_summary
        self.assertIsNotNone(spoken)
        self.assertLessEqual(len(spoken), 200)
        # Verify no markdown or json in spoken output
        self.assertNotIn("###", spoken)
        self.assertNotIn("**", spoken)
        self.assertNotIn("```", spoken)
        self.assertNotIn("{", spoken)
        self.assertNotIn("}", spoken)


if __name__ == "__main__":
    unittest.main()
