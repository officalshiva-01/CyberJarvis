import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.gateway import AgentGateway
from agent.fast_path import FastPathHandler
from agent.evaluator import contains_hallucinated_code, Evaluator
from agent.executor import PlanExecutor
from agent.schemas import AgentPlan, PlanStep
from memory import init_db, save_fact


class TestGatewayAndComponents(unittest.TestCase):
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

    def test_fast_path_name_resolution(self):
        fp = FastPathHandler()
        res = fp.try_handle("What is my name?")
        self.assertIsNotNone(res)
        self.assertIn("Shivaji", res.answer)
        self.assertEqual(res.spoken_summary, "Your name is Shivaji.")

    def test_fast_path_time_resolution(self):
        fp = FastPathHandler()
        res = fp.try_handle("What time is it?")
        self.assertIsNotNone(res)
        self.assertIn("current system time", res.answer.lower())
        self.assertGreater(len(res.tool_results), 0)

    def test_fast_path_connectivity(self):
        fp = FastPathHandler()
        res = fp.try_handle("Check my internet")
        self.assertIsNotNone(res)
        self.assertTrue("Internet Status" in res.answer)

    def test_fast_path_bypass_on_complex_query(self):
        fp = FastPathHandler()
        res = fp.try_handle("Diagnose my computer and tell me why it is slow")
        self.assertIsNone(res)  # Must return None to trigger LLM Agent Path

    def test_contains_hallucinated_code(self):
        self.assertTrue(contains_hallucinated_code("#!/bin/bash\ncat /proc/meminfo"))
        self.assertTrue(contains_hallucinated_code("```bash\n#!/bin/bash\nfree -m && ps aux\n```"))
        self.assertFalse(contains_hallucinated_code("Your system is running smoothly with 16.9% CPU."))

    def test_parallel_tool_execution(self):
        executor = PlanExecutor()
        plan = AgentPlan(
            intent="system_diagnosis",
            goal="Test parallel execution",
            steps=[
                PlanStep(tool="get_cpu_usage"),
                PlanStep(tool="get_memory_usage"),
                PlanStep(tool="get_disk_usage"),
            ],
        )
        results = executor.execute_plan(plan)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))

    def test_gateway_routing_and_latency(self):
        gateway = AgentGateway()
        res = gateway.process_command("What is my name?", source="test")
        self.assertEqual(res.route, "fast_path")
        self.assertIn("Shivaji", res.response.answer)
        self.assertGreaterEqual(res.latency.total_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
