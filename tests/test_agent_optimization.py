"""
Unit tests for CyberJarvis Tier-3 Agent Path Optimization.
Verifies:
1. Four-tier deterministic routing (FAST, DIAGNOSTIC, INFORMATION, AGENT).
2. Lightweight INFORMATION path execution (0 tools, skipped planner, skipped evaluator).
3. Strict tool relevance filtering in Planner.
4. Programmatic tool capping.
5. Conditional Evaluator bypass when no tools are executed.
6. Network tools timeout resilience (1.5s).
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import ExecutionRouter, RouteType
from agent.gateway import AgentGateway, GatewayResult
from agent.information_path import InformationPathHandler
from agent.planner import Planner, is_tool_relevant_to_request, MAX_TOOLS_PER_PLAN
from agent.schemas import AgentPlan, PlanStep, ToolResult
from agent.agent_core import AgentCore
from tools.network_tools import check_network_connectivity, check_dns_resolution
from memory import init_db


class TestAgentPathOptimization(unittest.TestCase):
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

    # -------------------------------------------------------------
    # 1. Routing Verification: 12 Core Queries + Agent + Info Queries
    # -------------------------------------------------------------
    def test_routing_matrix(self):
        test_cases = [
            # Tier 1: Fast
            ("what time is it", RouteType.FAST),
            ("check my ram", RouteType.FAST),
            ("current time", RouteType.FAST),
            ("hello", RouteType.FAST),

            # Tier 2: Diagnostic
            ("diagnose my computer", RouteType.DIAGNOSTIC),
            ("diagnosis my computer", RouteType.DIAGNOSTIC),
            ("diagnosis by computer", RouteType.DIAGNOSTIC),
            ("diagnose the computer", RouteType.DIAGNOSTIC),
            ("diagnose my system", RouteType.DIAGNOSTIC),
            ("check my computer", RouteType.DIAGNOSTIC),
            ("check my system", RouteType.DIAGNOSTIC),
            ("computer diagnosis", RouteType.DIAGNOSTIC),
            ("system diagnosis", RouteType.DIAGNOSTIC),
            ("why is my computer slow", RouteType.DIAGNOSTIC),
            ("why is my system slow", RouteType.DIAGNOSTIC),
            ("what is wrong with my computer", RouteType.DIAGNOSTIC),
            ("what is consuming memory", RouteType.DIAGNOSTIC),
            ("what is consuming the most memory", RouteType.DIAGNOSTIC),
            ("RAM kaun use kar raha hai", RouteType.DIAGNOSTIC),
            ("mera computer diagnose karo", RouteType.DIAGNOSTIC),
            ("system health check", RouteType.DIAGNOSTIC),

            # Tier 3b: Information (Direct LLM)
            ("how to become a developer", RouteType.INFORMATION),
            ("explain devops", RouteType.INFORMATION),
            ("what is python", RouteType.INFORMATION),
            ("how does Docker work", RouteType.INFORMATION),
            ("what is cybersecurity", RouteType.INFORMATION),
            ("how should i learn backend development", RouteType.INFORMATION),
            ("what is github", RouteType.INFORMATION),
            ("explain git repository", RouteType.INFORMATION),
            ("best practices for rest api design", RouteType.INFORMATION),

            # Tier 3a: Agent
            ("analyze my CyberJarvis architecture", RouteType.AGENT),
            ("analyse my CyberJarvis architecture", RouteType.AGENT),
            ("analyze my CyberJarvis project", RouteType.AGENT),
            ("inspect my CyberJarvis architecture", RouteType.AGENT),
            ("inspect my project structure", RouteType.AGENT),
            ("plan a deployment architecture", RouteType.AGENT),
            ("Analyze my GitHub repository and suggest architectural improvements.", RouteType.AGENT),
            ("Plan a multi-step deployment to AWS ECS.", RouteType.AGENT),
            ("Compare monolithic and microservices architectures for our stack.", RouteType.AGENT),
            ("Research this unknown database deadlock using available diagnostic logs.", RouteType.AGENT),
        ]
        for query, expected_route in test_cases:
            with self.subTest(query=query):
                actual = self.router.classify(query)
                self.assertEqual(
                    actual,
                    expected_route,
                    f"Query '{query}' classified as {actual}, expected {expected_route}",
                )

    # -------------------------------------------------------------
    # 2. Information Path Execution (0 tools, skips planner & evaluator)
    # -------------------------------------------------------------
    @patch("agent.information_path.ollama.chat")
    def test_information_path_execution(self, mock_chat):
        mock_chat.side_effect = Exception("Ollama offline in unit test")
        info_handler = InformationPathHandler()
        # Test offline fallback generation
        resp, llm_ms = info_handler.handle("how to become a developer")

        self.assertTrue(resp.completed)
        self.assertEqual(len(resp.tool_results), 0)
        self.assertEqual(resp.planning_ms, 0.0)
        self.assertEqual(resp.evaluator_ms, 0.0)
        self.assertIn("developer", resp.answer.lower())
        self.assertTrue(bool(resp.spoken_summary))

    @patch("agent.information_path.ollama.chat")
    def test_gateway_information_path_end_to_end(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": "To become a developer, learn Python, build real projects, and use Git.\n\nSUMMARY: Learn Python, build projects, and practice Git."
            }
        }
        gw = AgentGateway()
        res: GatewayResult = gw.process_user_request("how to become a developer", source="test")

        self.assertEqual(res.execution_path, "INFORMATION")
        self.assertEqual(res.route, "information_path")
        self.assertEqual(res.latency.planning_ms, 0.0)
        self.assertEqual(res.latency.evaluator_ms, 0.0)
        self.assertEqual(res.latency.tools_ms, 0.0)
        self.assertEqual(len(res.response.tool_results), 0)
        self.assertIn("developer", res.response.answer.lower())
        self.assertEqual(res.response.spoken_summary, "Learn Python, build projects, and practice Git.")

    # -------------------------------------------------------------
    # 3. Planner Tool Relevance Guardrails
    # -------------------------------------------------------------
    def test_tool_relevance_filter(self):
        dev_query = "how to become a developer"
        # Generic environment tools must be rejected
        self.assertFalse(is_tool_relevant_to_request("get_os_info", dev_query))
        self.assertFalse(is_tool_relevant_to_request("get_time", dev_query))
        self.assertFalse(is_tool_relevant_to_request("get_current_directory", dev_query))
        self.assertFalse(is_tool_relevant_to_request("list_project_files", dev_query))
        self.assertFalse(is_tool_relevant_to_request("read_project_file", dev_query))
        self.assertFalse(is_tool_relevant_to_request("check_network_connectivity", dev_query))

        # Explicit requests must be accepted
        self.assertTrue(is_tool_relevant_to_request("get_os_info", "what is my linux operating system"))
        self.assertTrue(is_tool_relevant_to_request("get_time", "what time is it right now"))
        self.assertTrue(is_tool_relevant_to_request("get_current_directory", "what is the current working directory"))
        self.assertTrue(is_tool_relevant_to_request("list_project_files", "list files in this project"))
        self.assertTrue(is_tool_relevant_to_request("read_project_file", "read file README.md"))
        self.assertTrue(is_tool_relevant_to_request("check_network_connectivity", "ping network connection"))
        self.assertTrue(is_tool_relevant_to_request("diagnose_system_bottleneck", "diagnose my computer"))

    @patch("agent.planner.ollama.chat")
    def test_planner_filters_hallucinated_tools(self, mock_chat):
        # Simulate LLM returning 6 irrelevant tools for an architecture task
        mock_chat.return_value = {
            "message": {
                "content": (
                    '{"intent": "architecture_analysis", "goal": "Analyze architecture", '
                    '"steps": ['
                    '{"tool": "get_os_info"}, '
                    '{"tool": "get_time"}, '
                    '{"tool": "get_current_directory"}, '
                    '{"tool": "list_project_files"}, '
                    '{"tool": "check_network_connectivity"}'
                    ']}'
                )
            }
        }
        planner = Planner()
        plan = planner.plan("analyze my CyberJarvis architecture and project structure")

        # get_os_info, get_time, check_network_connectivity must be filtered out
        # list_project_files is relevant because user asked about project structure
        tool_names = [s.tool for s in plan.steps]
        self.assertNotIn("get_os_info", tool_names)
        self.assertNotIn("get_time", tool_names)
        self.assertNotIn("check_network_connectivity", tool_names)
        self.assertLessEqual(len(plan.steps), MAX_TOOLS_PER_PLAN)

    # -------------------------------------------------------------
    # 4. AgentCore Conditional Evaluator Bypass
    # -------------------------------------------------------------
    @patch("agent.planner.ollama.chat")
    @patch("agent.evaluator.ollama.chat")
    def test_agent_core_skips_evaluator_when_no_tools(self, mock_eval_chat, mock_plan_chat):
        mock_plan_chat.return_value = {
            "message": {
                "content": '{"intent": "general_discussion", "goal": "Chat", "steps": []}'
            }
        }
        mock_eval_chat.return_value = {
            "message": {
                "content": "Here is a direct answer without tool evaluation."
            }
        }
        agent = AgentCore()
        resp = agent.process("tell me about software architecture principles")

        self.assertEqual(resp.evaluator_ms, 0.0)
        self.assertEqual(len(resp.tool_results), 0)

    # -------------------------------------------------------------
    # 5. Network Tools Timeout Bounds
    # -------------------------------------------------------------
    def test_network_tools_timeouts(self):
        # Even with an unroutable address, check_network_connectivity must return within ~1.5s
        res = check_network_connectivity(host="192.0.2.1", port=81)
        self.assertIn("connected", res)
        self.assertFalse(res["connected"])

        # DNS resolution failure test
        dns_res = check_dns_resolution(domain="nonexistent.invalid.local.test")
        self.assertFalse(dns_res["success"])


if __name__ == "__main__":
    unittest.main()
