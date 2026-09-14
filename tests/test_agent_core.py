import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent_core import AgentCore
from memory import init_db, get_facts


class TestAgentCore(unittest.TestCase):
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

    @patch("ollama.chat")
    def test_agent_core_diagnose_flow(self, mock_ollama):
        # Mock LLM planner output
        mock_ollama.return_value = {
            "message": {
                "content": '{"intent": "system_diagnosis", "goal": "Find cause of slowdown", "steps": [{"tool": "diagnose_system_bottleneck"}]}'
            }
        }
        statuses = []
        agent = AgentCore(on_status_change=lambda s: statuses.append(s))
        resp = agent.process("CyberJarvis, diagnose my computer and tell me why it is slow.")

        self.assertTrue(resp.completed)
        self.assertIsNotNone(resp.plan)
        self.assertEqual(resp.plan.intent, "system_diagnosis")
        self.assertGreater(len(resp.tool_results), 0)
        self.assertIn("CPU Usage", resp.answer)
        self.assertIn("RAM Usage", resp.answer)
        self.assertIsNotNone(resp.spoken_summary)
        self.assertGreater(len(resp.spoken_summary), 10)

        # Verify statuses
        self.assertTrue(any("Planning" in s for s in statuses))
        self.assertTrue(any("diagnose_system_bottleneck" in s for s in statuses))
        self.assertTrue(any("Analyzing" in s for s in statuses))

    @patch("ollama.chat")
    def test_agent_core_fact_saving(self, mock_ollama):
        mock_ollama.return_value = {
            "message": {
                "content": '{"intent": "general_chat", "goal": "Save fact", "steps": []}'
            }
        }
        agent = AgentCore()
        resp = agent.process("my name is TestEngineer")
        facts = get_facts(limit=5)
        self.assertIn("my name is TestEngineer", facts)


if __name__ == "__main__":
    unittest.main()
