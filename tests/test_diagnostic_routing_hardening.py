"""
Unit tests for CyberJarvis diagnostic routing hardening and speech-artifact normalization.
Verifies:
1. Deterministic classification of natural diagnostic phrasings and ASR variants to DIAGNOSTIC.
2. Conservative speech-artifact normalization for CyberJarvis variants.
3. Preservation of unrelated "cyber..." phrases (e.g., "what is cybersecurity" -> INFORMATION).
4. AGENT path classification requiring clear evidence of CyberJarvis/project inspection or multi-step task.
5. INFORMATION system prompt hygiene (no unsolicited personal background facts or CyberJarvis references).
6. Non-regression of existing FAST, DIAGNOSTIC, AGENT, and CLARIFICATION routing.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.router import ExecutionRouter, RouteType, normalize_transcript, is_incomplete_transcript
from agent.gateway import AgentGateway, GatewayResult
from agent.information_path import InformationPathHandler
from memory import init_db, save_fact


class TestDiagnosticRoutingHardening(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        os.environ["CYBERJARVIS_DB_PATH"] = self.tmp_db.name
        init_db()
        save_fact("My name is Shivaji")
        save_fact("My goal is to become a DevOps engineer")
        save_fact("My project is CyberJarvis")
        self.router = ExecutionRouter()

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except Exception:
            pass
        os.environ.pop("CYBERJARVIS_DB_PATH", None)

    # ------------------------------------------------------------------
    # 1. Required DIAGNOSTIC Queries (Issue 1)
    # ------------------------------------------------------------------
    def test_required_diagnostic_routing(self):
        """All required diagnostic queries and common ASR variants must route to DIAGNOSTIC."""
        diagnostic_queries = [
            "diagnose my computer",
            "diagnosis my computer",
            "diagnosis by computer",
            "diagnose the computer",
            "diagnose my system",
            "check my computer",
            "check my system",
            "computer diagnosis",
            "system diagnosis",
            "why is my computer slow",
            "what is wrong with my computer",
            "what is consuming the most memory",
            "RAM kaun use kar raha hai",
            "mera computer diagnose karo",
            # Additional common ASR variants
            "diagnostic my computer",
            "check computer",
            "diagnose system",
            "check computer health",
            "what is consuming memory",
            "who is using my RAM",
        ]
        for query in diagnostic_queries:
            with self.subTest(query=query):
                route = self.router.classify(query)
                self.assertEqual(
                    route,
                    RouteType.DIAGNOSTIC,
                    f"Query '{query}' classified as {route}, expected {RouteType.DIAGNOSTIC}",
                )

    # ------------------------------------------------------------------
    # 2. Required AGENT Queries (Issue 2)
    # ------------------------------------------------------------------
    def test_required_agent_routing(self):
        """All queries requesting CyberJarvis or project inspection must route to AGENT."""
        agent_queries = [
            "analyze my CyberJarvis architecture",
            "analyse my CyberJarvis architecture",
            "analyze my CyberJarvis project",
            "inspect my CyberJarvis architecture",
        ]
        for query in agent_queries:
            with self.subTest(query=query):
                route = self.router.classify(query)
                self.assertEqual(
                    route,
                    RouteType.AGENT,
                    f"Query '{query}' classified as {route}, expected {RouteType.AGENT}",
                )

    # ------------------------------------------------------------------
    # 3. Required INFORMATION Queries
    # ------------------------------------------------------------------
    def test_required_information_routing(self):
        """Generic informational queries must route to INFORMATION (Direct LLM)."""
        information_queries = [
            "how to become a developer",
            "what is Python",
            "explain DevOps",
            "how does Docker work",
            "what is cybersecurity",
        ]
        for query in information_queries:
            with self.subTest(query=query):
                route = self.router.classify(query)
                self.assertEqual(
                    route,
                    RouteType.INFORMATION,
                    f"Query '{query}' classified as {route}, expected {RouteType.INFORMATION}",
                )

    # ------------------------------------------------------------------
    # 4. Speech-Artifact Normalization Layer
    # ------------------------------------------------------------------
    def test_speech_artifact_normalization(self):
        """Verify conservative speech-artifact normalization for ASR corruptions."""
        cases = [
            # Issue 1: Diagnostic ASR variants
            ("diagnosis by computer", "diagnose my computer"),
            ("diagnosis my computer", "diagnose my computer"),
            ("diagnostic my computer", "diagnose my computer"),
            ("diagnose the computer", "diagnose the computer"),
            ("diagnose by computer", "diagnose my computer"),
            # Issue 2: CyberJarvis variants
            ("cyber jarvis architecture", "cyberjarvis architecture"),
            ("analyse my cyber jarvis", "analyze my cyberjarvis"),
            ("analyze my cyber jarvis", "analyze my cyberjarvis"),
            ("analyse my cyber service", "analyze my cyberjarvis architecture"),
            ("analyze my cyber service", "analyze my cyberjarvis architecture"),
            ("cyber service architecture", "cyberjarvis architecture"),
            ("analyze cyber jarvis architecture", "analyze cyberjarvis architecture"),
            # Unrelated phrases with "cyber" must NOT be corrupted
            ("what is cybersecurity", "what is cybersecurity"),
            ("explain cybersecurity", "explain cybersecurity"),
            ("what is a cyber attack", "what is a cyber attack"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                norm = normalize_transcript(raw)
                self.assertEqual(norm, expected, f"Failed for raw='{raw}': got '{norm}', expected '{expected}'")

    # ------------------------------------------------------------------
    # 5. ASR Corruption End-to-End Routing
    # ------------------------------------------------------------------
    def test_asr_corruptions_route_correctly(self):
        """Verify corrupted voice inputs route to their intended tiers."""
        corrupted_cases = [
            ("diagnosis by computer", RouteType.DIAGNOSTIC),
            ("analyse my cyber service", RouteType.AGENT),
            ("analyze my cyber service", RouteType.AGENT),
            ("analyse my cyber jarvis", RouteType.AGENT),
            ("analyze cyber jarvis architecture", RouteType.AGENT),
            ("what is cybersecurity", RouteType.INFORMATION),
        ]
        for query, expected_route in corrupted_cases:
            with self.subTest(query=query):
                route = self.router.classify(query)
                self.assertEqual(
                    route,
                    expected_route,
                    f"Corrupted query '{query}' classified as {route}, expected {expected_route}",
                )

    # ------------------------------------------------------------------
    # 6. Non-Regression of Existing FAST Path Queries
    # ------------------------------------------------------------------
    def test_fast_path_non_regression(self):
        """Deterministic FAST queries must not regress to DIAGNOSTIC or AGENT."""
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
            "what is my project",
            "what is cyberjarvis",
            "git status",
            "uptime",
        ]
        for query in fast_queries:
            with self.subTest(query=query):
                route = self.router.classify(query)
                self.assertEqual(
                    route,
                    RouteType.FAST,
                    f"Fast query '{query}' regressed to {route}, expected {RouteType.FAST}",
                )

    # ------------------------------------------------------------------
    # 7. Gateway Zero-Ollama Assertion for DIAGNOSTIC and FAST
    # ------------------------------------------------------------------
    def test_gateway_zero_ollama_for_diagnosis_by_computer(self):
        """'diagnosis by computer' must execute through DIAGNOSTIC with zero Ollama calls."""
        gateway = AgentGateway()
        with patch("ollama.chat") as mock_ollama:
            mock_ollama.side_effect = AssertionError("Ollama chat was invoked on DIAGNOSTIC path!")
            result = gateway.process_user_request("diagnosis by computer", source="voice")
            self.assertEqual(result.execution_path, "DIAGNOSTIC")
            self.assertEqual(result.route, "diagnostic_path")
            self.assertEqual(result.latency.planning_ms, 0.0)
            self.assertEqual(result.latency.evaluator_ms, 0.0)
            self.assertTrue(result.response.completed)
            self.assertIn("diagnostic_report", result.response.__dict__)

    # ------------------------------------------------------------------
    # 8. Information System Prompt Hygiene
    # ------------------------------------------------------------------
    @patch("agent.information_path.ollama.chat")
    def test_information_prompt_does_not_inject_unsolicited_facts(self, mock_chat):
        """Generic questions like 'what is Python' must NOT inject user name or CyberJarvis into system prompt."""
        mock_chat.return_value = {
            "message": {
                "content": "Python is a high-level programming language.\n\nSUMMARY: Python is a versatile language."
            }
        }
        handler = InformationPathHandler()
        resp, _ = handler.handle("what is Python")

        self.assertTrue(mock_chat.called)
        call_kwargs = mock_chat.call_args[1]
        messages = call_kwargs.get("messages", [])
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

        self.assertNotIn("User background context", system_msg)
        self.assertNotIn("Shivaji", system_msg)
        self.assertNotIn("DevOps", system_msg)
        self.assertNotIn("My project is CyberJarvis", system_msg)

    @patch("agent.information_path.ollama.chat")
    def test_information_prompt_injects_facts_when_personalized(self, mock_chat):
        """Personalized advice queries DO inject user background facts when explicitly requested."""
        mock_chat.return_value = {
            "message": {
                "content": "Based on your background, here is career advice.\n\nSUMMARY: Advice for your career."
            }
        }
        handler = InformationPathHandler()
        resp, _ = handler.handle("give me advice for my career based on my goals")

        self.assertTrue(mock_chat.called)
        call_kwargs = mock_chat.call_args[1]
        messages = call_kwargs.get("messages", [])
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

        self.assertIn("Shivaji", system_msg)
        self.assertIn("DevOps", system_msg)


if __name__ == "__main__":
    unittest.main()
