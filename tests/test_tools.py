import unittest
from tools.registry import registry
import tools


class TestTools(unittest.TestCase):
    def test_registry_has_expected_tools(self):
        tool_names = registry.list_tools()
        self.assertIn("get_cpu_usage", tool_names)
        self.assertIn("get_memory_usage", tool_names)
        self.assertIn("get_disk_usage", tool_names)
        self.assertIn("diagnose_system_bottleneck", tool_names)
        self.assertIn("get_top_processes", tool_names)
        self.assertIn("check_network_connectivity", tool_names)
        self.assertIn("read_project_file", tool_names)

    def test_get_cpu_usage_execution(self):
        res = registry.execute("get_cpu_usage")
        self.assertTrue(res.success)
        self.assertIsInstance(res.output, (int, float))
        self.assertTrue(0.0 <= res.output <= 100.0)

    def test_get_memory_usage_execution(self):
        res = registry.execute("get_memory_usage")
        self.assertTrue(res.success)
        self.assertIn("percent", res.output)
        self.assertIn("used_gb", res.output)
        self.assertIn("total_gb", res.output)

    def test_diagnose_system_bottleneck_execution(self):
        res = registry.execute("diagnose_system_bottleneck")
        self.assertTrue(res.success)
        self.assertIn("verdict", res.output)
        self.assertIn("metrics", res.output)
        self.assertIn("top_memory_consumers", res.output)
        self.assertIn("recommendations", res.output)

    def test_read_project_file_safety(self):
        res = registry.execute("read_project_file", {"filename": "config.py"})
        self.assertTrue(res.success)
        self.assertIn("content", res.output)

        # Path traversal attempt
        res_unsafe = registry.execute("read_project_file", {"filename": "../../../etc/passwd"})
        self.assertTrue("error" in res_unsafe.output or not res_unsafe.success)


if __name__ == "__main__":
    unittest.main()
