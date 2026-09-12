import unittest
from agent.planner import Planner, extract_json_from_text
from agent.schemas import AgentPlan, PlanStep
from agent.state import ConversationState


class TestPlanner(unittest.TestCase):
    def test_extract_json_direct(self):
        text = '{"intent": "system_diagnosis", "goal": "Check system", "steps": []}'
        data = extract_json_from_text(text)
        self.assertIsNotNone(data)
        self.assertEqual(data["intent"], "system_diagnosis")

    def test_extract_json_markdown(self):
        text = 'Here is the plan:\n```json\n{"intent": "system_diagnosis", "goal": "Check system", "steps": []}\n```'
        data = extract_json_from_text(text)
        self.assertIsNotNone(data)
        self.assertEqual(data["intent"], "system_diagnosis")

    def test_deterministic_fallback_planner_diagnose(self):
        planner = Planner()
        plan = planner._deterministic_fallback_plan("CyberJarvis, diagnose my computer and tell me why it is slow.")
        self.assertEqual(plan.intent, "system_diagnosis")
        self.assertGreater(len(plan.steps), 0)
        self.assertEqual(plan.steps[0].tool, "diagnose_system_bottleneck")

    def test_deterministic_fallback_planner_memory(self):
        planner = Planner()
        plan = planner._deterministic_fallback_plan("Check my ram memory usage")
        self.assertEqual(plan.intent, "check_memory")
        step_tools = [s.tool for s in plan.steps]
        self.assertIn("get_memory_usage", step_tools)

    def test_follow_up_reasoning_in_planner(self):
        planner = Planner()
        state = ConversationState()
        state.latest_diagnosis = {"verdict": "DEGRADED", "metrics": {"ram_percent": 91.0}}

        plan = planner.plan("What is consuming the most memory?", state=state)
        self.assertEqual(plan.intent, "investigate_processes")
        self.assertEqual(plan.steps[0].tool, "get_top_processes")
        self.assertEqual(plan.steps[0].parameters.get("sort_by"), "memory")


if __name__ == "__main__":
    unittest.main()
