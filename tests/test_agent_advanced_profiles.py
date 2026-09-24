"""
Comprehensive unit tests for CyberJarvis Phase 4 Advanced Agent Task Profiles:
1. PROJECT_INSPECTION
2. PROJECT_DEBUGGING
3. DEPLOYMENT_PLANNING

Verifies:
- Profile classification for all target phrases and variations
- Preservation of existing profiles (PROJECT_ARCHITECTURE, FAST, DIAGNOSTIC, INFORMATION, GENERAL_PLANNING)
- Irrelevant tools are not selected in deterministic plans
- General planner is skipped (planning_ms == 0.0)
- Evaluator is skipped (evaluator_ms == 0.0)
- Bounded context enforcement (<= 2500 characters)
- Timeout fallback mechanism with grounded sections
- Gateway end-to-end execution for voice and terminal
- Project tools execution (test summary, log reading, git status/log)
"""
import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.task_profile import AgentTaskProfile, classify_agent_task_profile
from agent.project_context import ProjectContextCollector, ArchitectureMapper, SynthesisResult
from agent.agent_core import AgentCore
from agent.gateway import AgentGateway
from agent.router import ExecutionRouter, RouteType
from memory import init_db
from tools.registry import registry


class TestAdvancedAgentTaskProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        os.environ["CYBERJARVIS_DB_PATH"] = self.tmp_db.name
        init_db()
        self.router = ExecutionRouter()

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
        os.environ.pop("CYBERJARVIS_DB_PATH", None)

    # ------------------------------------------------------------------
    # 1. Profile Classification Tests
    # ------------------------------------------------------------------
    def test_classify_project_inspection_queries(self):
        queries = [
            "inspect my CyberJarvis project",
            "inspect my project",
            "review my CyberJarvis codebase",
            "show me the current project structure",
            "review my repository",
            "inspect project structure",
            "review our codebase",
            "show project structure",
            "inspect files in this project",
            "browse project files",
        ]
        for q in queries:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(
                    profile,
                    AgentTaskProfile.PROJECT_INSPECTION,
                    f"Query '{q}' classified as {profile}, expected {AgentTaskProfile.PROJECT_INSPECTION}",
                )

    def test_classify_project_debugging_queries(self):
        queries = [
            "debug my CyberJarvis project",
            "why are my tests failing",
            "find the problem in my project",
            "analyze this error",
            "debug the current issue",
            "debug this repository problem",
            "diagnose a repository problem",
            "why are tests failing",
            "find bug in project",
            "troubleshoot project",
        ]
        for q in queries:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(
                    profile,
                    AgentTaskProfile.PROJECT_DEBUGGING,
                    f"Query '{q}' classified as {profile}, expected {AgentTaskProfile.PROJECT_DEBUGGING}",
                )

    def test_classify_deployment_planning_queries(self):
        queries = [
            "plan deployment for CyberJarvis",
            "how should I deploy CyberJarvis",
            "prepare CyberJarvis for Linux deployment",
            "create a deployment architecture",
            "plan a deployment architecture",
            "plan deployment to AWS ECS",
            "how to deploy CyberJarvis",
            "deploy CyberJarvis",
        ]
        for q in queries:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(
                    profile,
                    AgentTaskProfile.DEPLOYMENT_PLANNING,
                    f"Query '{q}' classified as {profile}, expected {AgentTaskProfile.DEPLOYMENT_PLANNING}",
                )

    def test_preservation_of_existing_profiles(self):
        # 1. PROJECT_ARCHITECTURE
        arch_queries = [
            "analyze my CyberJarvis architecture",
            "analyse my CyberJarvis architecture",
            "analyze my CyberJarvis project",
            "inspect my CyberJarvis architecture",
            "analyze this project architecture",
            "explain my CyberJarvis architecture",
            "review my CyberJarvis design",
        ]
        for q in arch_queries:
            with self.subTest(query=q):
                profile = classify_agent_task_profile(q)
                self.assertEqual(profile, AgentTaskProfile.PROJECT_ARCHITECTURE)

        # 2. GENERAL_PLANNING
        self.assertEqual(
            classify_agent_task_profile("refactor the memory subsystem"),
            AgentTaskProfile.GENERAL_PLANNING,
        )

        # 3. Router classifications for other tiers
        self.assertEqual(self.router.classify("what time is it"), RouteType.FAST)
        self.assertEqual(self.router.classify("check my ram"), RouteType.FAST)
        self.assertEqual(self.router.classify("diagnose my computer"), RouteType.DIAGNOSTIC)
        self.assertEqual(self.router.classify("why is my computer slow"), RouteType.DIAGNOSTIC)
        self.assertEqual(self.router.classify("what is python"), RouteType.INFORMATION)
        self.assertEqual(self.router.classify("explain devops"), RouteType.INFORMATION)

    # ------------------------------------------------------------------
    # 2. Router Routing to AGENT Path
    # ------------------------------------------------------------------
    def test_router_routes_advanced_profiles_to_agent(self):
        target_queries = [
            "inspect my CyberJarvis project",
            "inspect my project",
            "review my CyberJarvis codebase",
            "show me the current project structure",
            "review my repository",
            "debug my CyberJarvis project",
            "why are my tests failing",
            "find the problem in my project",
            "analyze this error",
            "debug the current issue",
            "plan deployment for CyberJarvis",
            "how should I deploy CyberJarvis",
            "prepare CyberJarvis for Linux deployment",
            "create a deployment architecture",
        ]
        for q in target_queries:
            with self.subTest(query=q):
                route = self.router.classify(q)
                self.assertEqual(
                    route,
                    RouteType.AGENT,
                    f"Query '{q}' classified as {route}, expected {RouteType.AGENT}",
                )

    # ------------------------------------------------------------------
    # 3. Deterministic Task Plans & Tool Relevance
    # ------------------------------------------------------------------
    def test_project_inspection_plan_and_tool_relevance(self):
        collector = ProjectContextCollector()
        plan = collector.build_inspection_task_plan("inspect my CyberJarvis project")

        self.assertEqual(plan.intent, "project_inspection")
        tool_names = [s.tool for s in plan.steps]

        # Must contain relevant inspection tools
        self.assertIn("list_project_files", tool_names)
        self.assertIn("get_git_status", tool_names)
        self.assertIn("get_git_recent_commits", tool_names)
        self.assertIn("get_test_summary", tool_names)
        self.assertIn("read_project_file", tool_names)

        # Must NOT contain unrelated system/hardware diagnostic tools
        unrelated = ["get_cpu_usage", "get_memory_usage", "get_top_processes", "diagnose_system_bottleneck", "check_network_connectivity"]
        for un in unrelated:
            self.assertNotIn(un, tool_names, f"Unrelated tool '{un}' found in PROJECT_INSPECTION plan")

    def test_project_debugging_plan_and_tool_relevance(self):
        collector = ProjectContextCollector()
        plan = collector.build_debugging_task_plan("debug my CyberJarvis project")

        self.assertEqual(plan.intent, "project_debugging")
        tool_names = [s.tool for s in plan.steps]

        self.assertIn("get_test_summary", tool_names)
        self.assertIn("read_project_log", tool_names)
        self.assertIn("get_git_status", tool_names)

        # Destructive or mutating tools must NEVER be present
        self.assertNotIn("delete_project_file", tool_names)
        self.assertNotIn("terminate_process", tool_names)

    def test_deployment_planning_plan_and_tool_relevance(self):
        collector = ProjectContextCollector()
        plan = collector.build_deployment_task_plan("plan deployment for CyberJarvis")

        self.assertEqual(plan.intent, "deployment_planning")
        tool_names = [s.tool for s in plan.steps]

        self.assertIn("list_project_files", tool_names)
        self.assertIn("read_project_file", tool_names)

        # Verify files targeted for inspection
        filenames = [s.parameters.get("filename") for s in plan.steps if "filename" in s.parameters]
        self.assertIn("requirements.txt", filenames)
        self.assertIn("config.py", filenames)
        self.assertIn("main.py", filenames)

    # ------------------------------------------------------------------
    # 4. Context Bounds Enforcement (<= 2500 chars)
    # ------------------------------------------------------------------
    def test_context_bounds_for_all_profiles(self):
        collector = ProjectContextCollector()

        # Inspection context
        insp_ev = collector.collect_inspection_evidence()
        self.assertLessEqual(
            len(insp_ev.get("structured_context", "")),
            2500,
            "Inspection structured context exceeds 2500 characters",
        )

        # Debugging context
        debug_ev = collector.collect_debugging_evidence()
        self.assertLessEqual(
            len(debug_ev.get("structured_context", "")),
            2500,
            "Debugging structured context exceeds 2500 characters",
        )

        # Deployment context
        dep_ev = collector.collect_deployment_evidence()
        self.assertLessEqual(
            len(dep_ev.get("structured_context", "")),
            2500,
            "Deployment structured context exceeds 2500 characters",
        )

    # ------------------------------------------------------------------
    # 5. Deterministic Execution: Planner & Evaluator Bypassed
    # ------------------------------------------------------------------
    @patch("agent.project_context.ollama.chat")
    def test_all_three_profiles_skip_general_planner_and_evaluator(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Project Structure Overview\nStructure.\n\n"
                    "### Repository & Git State\nGit state.\n\n"
                    "### Commit History\nHistory.\n\n"
                    "### Test Suite Status\nTests.\n\n"
                    "### Key Codebase Findings\nFindings.\n\n"
                    "SUMMARY: Grounded summary."
                )
            }
        }
        mock_planner = MagicMock()
        mock_planner.plan.side_effect = AssertionError("Generic planner.plan was unexpectedly invoked!")

        agent = AgentCore(planner=mock_planner)

        test_profiles = [
            ("inspect my CyberJarvis project", "PROJECT_INSPECTION"),
            ("debug my CyberJarvis project", "PROJECT_DEBUGGING"),
            ("plan deployment for CyberJarvis", "DEPLOYMENT_PLANNING"),
        ]

        for q, expected_profile in test_profiles:
            with self.subTest(query=q):
                resp = agent.process(q)
                self.assertEqual(resp.project_profile, expected_profile)
                self.assertEqual(resp.planning_ms, 0.0, "Planner must be skipped (0.0ms)")
                self.assertEqual(resp.evaluator_ms, 0.0, "Evaluator must be skipped (0.0ms)")
                self.assertTrue(resp.completed)
                self.assertGreater(len(resp.tool_results), 0)
                self.assertTrue(bool(resp.spoken_summary))

        self.assertFalse(mock_planner.plan.called)

    # ------------------------------------------------------------------
    # 6. Timeout Fallback Behavior
    # ------------------------------------------------------------------
    def test_inspection_timeout_fallback(self):
        collector = ProjectContextCollector(timeout_seconds=0.01)

        def slow_call(*args, **kwargs):
            time.sleep(0.5)
            return {"message": {"content": "Slow"}}

        with patch("agent.project_context.ollama.chat", side_effect=slow_call):
            synth_res = collector.synthesize_inspection("inspect my CyberJarvis project")

        self.assertEqual(synth_res.status, "TIMEOUT")
        self.assertTrue(synth_res.fallback_used)
        answer = synth_res.answer

        # Required sections in inspection fallback
        self.assertIn("### Project Structure Overview", answer)
        self.assertIn("### Repository & Git State", answer)
        self.assertIn("### Commit History", answer)
        self.assertIn("### Test Suite Status", answer)
        self.assertIn("### Key Codebase Findings", answer)
        self.assertTrue(bool(synth_res.spoken_summary))

    def test_debugging_timeout_fallback(self):
        collector = ProjectContextCollector(timeout_seconds=0.01)

        def slow_call(*args, **kwargs):
            time.sleep(0.5)
            return {"message": {"content": "Slow"}}

        with patch("agent.project_context.ollama.chat", side_effect=slow_call):
            synth_res = collector.synthesize_debugging("debug my CyberJarvis project")

        self.assertEqual(synth_res.status, "TIMEOUT")
        self.assertTrue(synth_res.fallback_used)
        answer = synth_res.answer

        # Required sections in debugging fallback
        self.assertIn("### Diagnostic Finding", answer)
        self.assertIn("### Test Execution & Results", answer)
        self.assertIn("### Log Analysis & Errors", answer)
        self.assertIn("### Git State & Recent Changes", answer)
        self.assertIn("### Root Cause Hypothesis", answer)
        self.assertIn("### Recommended Next Engineering Step", answer)
        self.assertTrue(bool(synth_res.spoken_summary))

    def test_deployment_timeout_fallback(self):
        collector = ProjectContextCollector(timeout_seconds=0.01)

        def slow_call(*args, **kwargs):
            time.sleep(0.5)
            return {"message": {"content": "Slow"}}

        with patch("agent.project_context.ollama.chat", side_effect=slow_call):
            synth_res = collector.synthesize_deployment("plan deployment for CyberJarvis")

        self.assertEqual(synth_res.status, "TIMEOUT")
        self.assertTrue(synth_res.fallback_used)
        answer = synth_res.answer

        # Required sections in deployment fallback
        self.assertIn("### Deployment Architecture Overview", answer)
        self.assertIn("### Prerequisites & System Dependencies", answer)
        self.assertIn("### Python & Voice Stack Configuration", answer)
        self.assertIn("### Ollama & LLM Model Setup", answer)
        self.assertIn("### Database & Persistence Setup", answer)
        self.assertIn("### Recommended Service Architecture", answer)
        self.assertTrue(bool(synth_res.spoken_summary))

    # ------------------------------------------------------------------
    # 7. Gateway End-to-End Execution: Terminal & Voice
    # ------------------------------------------------------------------
    @patch("agent.project_context.ollama.chat")
    def test_gateway_voice_and_terminal_routing(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": (
                    "### Project Structure Overview\nStructure.\n\n"
                    "### Repository & Git State\nState.\n\n"
                    "### Commit History\nHistory.\n\n"
                    "### Test Suite Status\nTests.\n\n"
                    "### Key Codebase Findings\nFindings.\n\n"
                    "### Diagnostic Finding\nDiagnosis.\n\n"
                    "### Test Execution & Results\nResults.\n\n"
                    "### Log Analysis & Errors\nLogs.\n\n"
                    "### Git State & Recent Changes\nChanges.\n\n"
                    "### Root Cause Hypothesis\nHypothesis.\n\n"
                    "### Recommended Next Engineering Step\nStep.\n\n"
                    "### Deployment Architecture Overview\nDeployment.\n\n"
                    "### Prerequisites & System Dependencies\nPrereqs.\n\n"
                    "### Python & Voice Stack Configuration\nPython.\n\n"
                    "### Ollama & LLM Model Setup\nOllama.\n\n"
                    "### Database & Persistence Setup\nDB.\n\n"
                    "### Recommended Service Architecture\nService.\n\n"
                    "SUMMARY: Fast gateway response."
                )
            }
        }
        gw = AgentGateway()

        cases = [
            ("inspect my CyberJarvis project", "PROJECT_INSPECTION"),
            ("debug my CyberJarvis project", "PROJECT_DEBUGGING"),
            ("plan deployment for CyberJarvis", "DEPLOYMENT_PLANNING"),
        ]

        for cmd, expected_profile in cases:
            # Voice request
            with self.subTest(source="voice", cmd=cmd):
                res = gw.process_user_request(cmd, source="voice")
                self.assertEqual(res.execution_path, "AGENT")
                self.assertEqual(res.project_profile, expected_profile)
                self.assertEqual(res.latency.planning_ms, 0.0)
                self.assertEqual(res.latency.evaluator_ms, 0.0)
                self.assertGreater(len(res.response.tool_results), 0)
                self.assertTrue(bool(res.response.spoken_summary))

            # Terminal request
            with self.subTest(source="terminal", cmd=cmd):
                res = gw.process_user_request(cmd, source="terminal")
                self.assertEqual(res.execution_path, "AGENT")
                self.assertEqual(res.project_profile, expected_profile)
                self.assertEqual(res.latency.planning_ms, 0.0)
                self.assertEqual(res.latency.evaluator_ms, 0.0)
                self.assertGreater(len(res.response.tool_results), 0)

    # ------------------------------------------------------------------
    # 8. Project Tools Functionality
    # ------------------------------------------------------------------
    def test_project_tools_execution(self):
        # 1. get_test_summary
        test_sum_res = registry.execute("get_test_summary")
        self.assertTrue(test_sum_res.success)
        self.assertGreaterEqual(test_sum_res.output.get("test_file_count", 0), 10)

        # 2. read_project_log
        log_res = registry.execute("read_project_log", {"lines": 20})
        self.assertTrue(log_res.success)
        self.assertIn("line_count", log_res.output)

        # 3. git_status & git_log aliases
        git_st = registry.execute("git_status")
        self.assertTrue(git_st.success)
        self.assertIn("branch", git_st.output)

        git_lg = registry.execute("git_log", {"limit": 2})
        self.assertTrue(git_lg.success)
        self.assertGreaterEqual(len(git_lg.output), 1)


if __name__ == "__main__":
    unittest.main()
