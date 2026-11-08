"""
Unit tests for AGENT project-architecture execution and deterministic task profiling.
Verifies:
1. Task profile classification:
   - "analyze my CyberJarvis architecture" -> PROJECT_ARCHITECTURE
   - "analyse my CyberJarvis architecture" -> PROJECT_ARCHITECTURE
   - "analyze my CyberJarvis project" -> PROJECT_ARCHITECTURE
   - "inspect my CyberJarvis architecture" -> PROJECT_ARCHITECTURE
   - "analyze this project architecture" -> PROJECT_ARCHITECTURE
   - "explain my CyberJarvis architecture" -> PROJECT_ARCHITECTURE
   - "review my CyberJarvis design" -> PROJECT_ARCHITECTURE
2. Deterministic execution pipeline for PROJECT_ARCHITECTURE:
   - Bypasses generic planner (planner.plan is NOT called)
   - Executes bounded repository inspection tools (list_project_files, read_project_file)
   - Bounded synthesis with grounded repository evidence
   - Evaluator skipped
3. Open-ended tasks preserve general planner:
   - "refactor the memory subsystem"
   - "plan a deployment architecture"
   - "diagnose a repository problem"
4. Grounded content verification:
   - Response contains required architecture sections (Overview, Request Flow, Major Components,
     Agent Execution Flow, Voice Flow, Memory Flow, Security Boundary, Strengths, Limitations, Next Steps)
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.task_profile import AgentTaskProfile, classify_agent_task_profile
from agent.project_context import ProjectContextCollector, ArchitectureMapper
from agent.agent_core import AgentCore
from agent.gateway import AgentGateway
from agent.router import ExecutionRouter, RouteType
from memory import init_db


class TestAgentProjectArchitecture(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        os.environ["CYBERJARVIS_DB_PATH"] = self.tmp_db.name
        init_db()

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
        os.environ.pop("CYBERJARVIS_DB_PATH", None)

    # ------------------------------------------------------------------
    # 1. Profile Classification Tests
    # ------------------------------------------------------------------
    def test_classify_project_architecture_profiles(self):
        queries = [
            "analyze my CyberJarvis architecture",
            "analyse my CyberJarvis architecture",
            "analyze my CyberJarvis project",
            "inspect my CyberJarvis architecture",
            "analyze this project architecture",
            "explain my CyberJarvis architecture",
            "review my CyberJarvis design",
        ]
        for q in queries:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(
                    profile,
                    AgentTaskProfile.PROJECT_ARCHITECTURE,
                    f"Query '{q}' classified as {profile}, expected {AgentTaskProfile.PROJECT_ARCHITECTURE}",
                )

    def test_classify_open_ended_tasks(self):
        open_ended = [
            ("refactor the memory subsystem", AgentTaskProfile.GENERAL_PLANNING),
            ("plan a deployment architecture", AgentTaskProfile.DEPLOYMENT_PLANNING),
            ("diagnose a repository problem", AgentTaskProfile.PROJECT_DEBUGGING),
        ]
        for q, expected in open_ended:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(
                    profile,
                    expected,
                    f"Query '{q}' classified as {profile}, expected {expected}",
                )

    # ------------------------------------------------------------------
    # 2. Deterministic Pipeline: No Generic Planner Call, Executes Tools
    # ------------------------------------------------------------------
    @patch("agent.project_context.ollama.chat")
    def test_project_architecture_bypasses_generic_planner_and_runs_tools(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Architecture Overview\nCyberJarvis is a local AI agent.\n\n"
                    "### Request Flow\nRequests pass through the gateway and router.\n\n"
                    "SUMMARY: CyberJarvis four-tier local architecture."
                )
            }
        }
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = AssertionError("Generic planner.plan was unexpectedly called!")

        agent = AgentCore(planner=mock_planner)
        response = agent.process("analyze my CyberJarvis architecture")

        # 1. Generic planner must NOT have been called
        self.assertFalse(mock_planner.plan.called)

        # 2. Profile must be PROJECT_ARCHITECTURE
        self.assertEqual(response.project_profile, "PROJECT_ARCHITECTURE")

        # 3. Planning ms must be 0.0 (skipped general planner)
        self.assertEqual(response.planning_ms, 0.0)

        # 4. Evaluator ms must be 0.0 (skipped evaluator)
        self.assertEqual(response.evaluator_ms, 0.0)

        # 5. At least one project inspection tool must have executed
        self.assertGreaterEqual(len(response.tool_results), 1)
        tool_names = [r.tool_name for r in response.tool_results]
        self.assertIn("list_project_files", tool_names)
        self.assertIn("read_project_file", tool_names)

        # 6. Response must be completed and contain architecture explanation
        self.assertTrue(response.completed)
        self.assertIn("Architecture", response.answer)
        self.assertTrue(bool(response.spoken_summary))

    @patch("agent.project_context.ollama.chat")
    def test_all_architecture_variants_use_deterministic_profile(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Architecture Overview\nOverview.\n\n"
                    "### Request Flow\nFlow.\n\n"
                    "SUMMARY: Architecture summary."
                )
            }
        }
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = AssertionError("Generic planner was called!")

        agent = AgentCore(planner=mock_planner)
        test_queries = [
            "analyse my CyberJarvis architecture",
            "inspect my CyberJarvis architecture",
            "analyze my CyberJarvis project",
        ]
        for q in test_queries:
            with self.subTest(query=q):
                resp = agent.process(q)
                self.assertEqual(resp.project_profile, "PROJECT_ARCHITECTURE")
                self.assertGreater(len(resp.tool_results), 0)
                self.assertEqual(resp.planning_ms, 0.0)

    # ------------------------------------------------------------------
    # 3. Open-Ended Tasks Still Use General Planner
    # ------------------------------------------------------------------
    @patch("agent.planner.ollama.chat")
    @patch("agent.evaluator.ollama.chat")
    def test_open_ended_tasks_use_general_planner(self, mock_eval_chat, mock_plan_chat):
        mock_plan_chat.return_value = {
            "message": {
                "content": '{"intent": "general_refactor", "goal": "Refactor memory subsystem", "steps": []}'
            }
        }
        mock_eval_chat.return_value = {
            "message": {
                "content": "Refactoring plan analysis."
            }
        }
        agent = AgentCore()
        resp = agent.process("refactor the memory subsystem")

        # General planner was invoked
        self.assertTrue(mock_plan_chat.called)
        self.assertNotEqual(resp.project_profile, "PROJECT_ARCHITECTURE")

    # ------------------------------------------------------------------
    # 4. Gateway End-to-End Execution for Architecture Query
    # ------------------------------------------------------------------
    @patch("agent.project_context.ollama.chat")
    def test_gateway_agent_architecture_flow(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Architecture Overview\nCyberJarvis local diagnostic architecture.\n\n"
                    "### Request Flow\n1. STT -> 2. Router -> 3. Execution Path.\n\n"
                    "### Major Components\nGateway, Router, AgentCore, Tools, Voice.\n\n"
                    "SUMMARY: Four-tier local AI architecture."
                )
            }
        }
        gw = AgentGateway()
        res = gw.process_user_request("analyze my CyberJarvis architecture", source="test")

        self.assertEqual(res.execution_path, "AGENT")
        self.assertEqual(res.project_profile, "PROJECT_ARCHITECTURE")
        self.assertEqual(res.latency.planning_ms, 0.0)
        self.assertGreater(len(res.response.tool_results), 0)
        self.assertIn("Architecture", res.response.answer)

    # ------------------------------------------------------------------
    # 5. Offline Fallback Grounded Architecture Synthesis
    # ------------------------------------------------------------------
    def test_offline_fallback_architecture_report_sections(self):
        collector = ProjectContextCollector()
        plan = collector.build_architecture_task_plan("analyze my CyberJarvis architecture")
        from agent.executor import PlanExecutor
        executor = PlanExecutor()
        results = executor.execute_plan(plan)

        # Test fallback directly (Ollama offline)
        with patch("agent.project_context.ollama.chat") as mock_chat:
            mock_chat.side_effect = Exception("Ollama offline in unit test")
            answer, spoken, duration = collector.synthesize_architecture(
                "analyze my CyberJarvis architecture", results
            )

        # Verify all required sections are present
        required_sections = [
            "### Architecture Overview",
            "### Request Flow",
            "### Major Components",
            "### Agent Execution Flow",
            "### Voice Flow",
            "### Memory Flow",
            "### Security Boundary",
            "### Current Strengths",
            "### Current Limitations",
            "### Recommended Next Engineering Step",
        ]
        for sec in required_sections:
            self.assertIn(sec, answer, f"Missing required section: {sec}")

        # Check required component mentions
        components = [
            "main.py", "agent/gateway.py", "agent/router.py",
            "FAST", "DIAGNOSTIC", "INFORMATION", "AGENT",
            "tools/registry.py", "voice", "memory", "security",
        ]
        for comp in components:
            self.assertIn(comp, answer, f"Missing component in report: {comp}")

        self.assertTrue(bool(spoken))

    # ------------------------------------------------------------------
    # 6. ArchitectureMapper: Component Detection & Boundaries
    # ------------------------------------------------------------------
    def test_architecture_mapper_component_detection(self):
        mapper = ArchitectureMapper()
        components = mapper.scan_components()

        # Verify all 15 required subsystems are scanned
        expected_keys = [
            "entry_point", "gateway", "router", "agent_core",
            "planner", "executor", "evaluator", "tools",
            "voice", "memory", "security", "integrations",
            "observability", "ui", "tests",
        ]
        for key in expected_keys:
            self.assertIn(key, components, f"Missing component key: {key}")

        # Verify existence of key core files
        self.assertTrue(components["entry_point"]["exists"], "main.py must exist")
        self.assertTrue(components["gateway"]["exists"], "agent/gateway.py must exist")
        self.assertTrue(components["router"]["exists"], "agent/router.py must exist")
        self.assertEqual(
            components["router"]["tiers"],
            ["FAST", "DIAGNOSTIC", "INFORMATION", "AGENT"],
        )
        self.assertTrue(components["agent_core"]["exists"], "agent/agent_core.py must exist")
        self.assertTrue(components["tools"]["exists"], "tools/registry.py must exist")
        self.assertGreaterEqual(components["tests"]["count"], 10, "Should detect automated tests")

    def test_architecture_context_length_stays_bounded(self):
        mapper = ArchitectureMapper()
        components = mapper.scan_components()
        context = mapper.build_structured_context(components)

        # Target: <= 2500 characters
        self.assertLessEqual(
            len(context),
            2500,
            f"Context length {len(context)} exceeds max allowed 2500 characters",
        )
        self.assertIn("CyberJarvis Structured Architecture Facts", context)
        self.assertIn("main.py", context)
        self.assertIn("agent/gateway.py", context)

    def test_architecture_mapper_excludes_unrelated_files(self):
        mapper = ArchitectureMapper()

        # Excluded directories
        self.assertIsNone(mapper.read_bounded_file(".git/config"))
        self.assertIsNone(mapper.read_bounded_file("__pycache__/test.cpython-310.pyc"))

        # Excluded file extensions
        self.assertIsNone(mapper.read_bounded_file("cyberjarvis.db"))
        self.assertIsNone(mapper.read_bounded_file("recording.wav"))
        self.assertIsNone(mapper.read_bounded_file("app.log"))

        # Path traversal guard
        self.assertIsNone(mapper.read_bounded_file("../../etc/passwd"))

        # Valid relevant file
        readme_content = mapper.read_bounded_file("README.md", max_chars=100)
        self.assertIsNotNone(readme_content)
        self.assertLessEqual(len(readme_content), 125)  # 100 + ellipsis

    # ------------------------------------------------------------------
    # 7. Synthesis Timeout Failsafe & Timing Breakdown
    # ------------------------------------------------------------------
    def test_synthesis_timeout_failsafe_triggers_deterministic_fallback(self):
        import time

        def slow_ollama(*args, **kwargs):
            time.sleep(1.0)
            return {"message": {"content": "Slow content"}}

        collector = ProjectContextCollector(timeout_seconds=0.05)
        start_t = time.perf_counter()

        with patch("agent.project_context.ollama.chat", side_effect=slow_ollama):
            synth_res = collector.synthesize_architecture(
                "analyze my CyberJarvis architecture"
            )

        elapsed = time.perf_counter() - start_t

        self.assertLess(elapsed, 0.8)
        self.assertEqual(synth_res.status, "TIMEOUT")
        self.assertTrue(synth_res.fallback_used)

        answer = synth_res.answer
        spoken = synth_res.spoken_summary

        self.assertIn("### Architecture Overview", answer)
        self.assertIn("### Request Flow", answer)
        self.assertIn("### Major Components", answer)
        self.assertIn("### Recommended Next Engineering Step", answer)
        self.assertTrue(bool(spoken))



    @patch("agent.project_context.ollama.chat")
    def test_gateway_timing_breakdown_includes_repo_scan_and_evidence(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Architecture Overview\nCyberJarvis architecture.\n\n"
                    "### Request Flow\nGateway routes queries.\n\n"
                    "### Major Components\nEntry, Gateway, Router, Tools.\n\n"
                    "### Recommended Next Engineering Step\nExtend task profiles.\n\n"
                    "SUMMARY: Grounded architecture summary."
                )
            }
        }
        gw = AgentGateway()
        res = gw.process_user_request("analyze my CyberJarvis architecture", source="test")

        # Check granular timing metrics in breakdown
        self.assertGreaterEqual(res.latency.repo_scan_ms, 0.0)
        self.assertGreaterEqual(res.latency.evidence_extraction_ms, 0.0)
        self.assertGreaterEqual(res.latency.synthesis_ms, 0.0)
        self.assertGreaterEqual(res.latency.total_ms, 0.0)
        self.assertEqual(res.latency.synthesis_status, "SUCCESS")
        self.assertFalse(res.latency.fallback_used)

        # Check formatted timing report
        report = res.latency.format_timing_report()
        self.assertIn("Repository Scan:", report)
        self.assertIn("Evidence Extraction:", report)
        self.assertIn("LLM Synthesis:", report)
        self.assertIn("Status: SUCCESS", report)
        self.assertIn("Fallback: NO", report)
        self.assertIn("Total:", report)
        self.assertIn("Planner: SKIPPED (Deterministic Profile)", report)


if __name__ == "__main__":
    unittest.main()

