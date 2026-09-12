import unittest
from security.policy import SecurityPolicy, RiskLevel, PermissionLevel
from security.permissions import is_path_safe_for_project, is_process_kill_safe


class TestSecurity(unittest.TestCase):
    def test_security_policy_read_only_allowed(self):
        policy = SecurityPolicy()
        decision = policy.evaluate("get_cpu_usage", PermissionLevel.READ_ONLY, RiskLevel.LOW, is_read_only=True)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_confirmation)

    def test_security_policy_mutating_requires_confirmation(self):
        policy = SecurityPolicy(allow_auto_mutating=False)
        decision = policy.evaluate("delete_file", PermissionLevel.MUTATING, RiskLevel.HIGH, is_read_only=False)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_security_policy_critical_blocked(self):
        policy = SecurityPolicy()
        decision = policy.evaluate("format_drive", PermissionLevel.PRIVILEGED, RiskLevel.CRITICAL, is_read_only=False)
        self.assertFalse(decision.allowed)
        self.assertIn("CRITICAL", decision.reason)

    def test_is_path_safe(self):
        safe, _ = is_path_safe_for_project("main.py")
        self.assertTrue(safe)

        unsafe, _ = is_path_safe_for_project("../../etc/shadow")
        self.assertFalse(unsafe)

        unsafe_ssh, _ = is_path_safe_for_project(".ssh/id_rsa")
        self.assertFalse(unsafe_ssh)

    def test_process_kill_protection(self):
        safe, _ = is_process_kill_safe(1)
        self.assertFalse(safe)

        safe, _ = is_process_kill_safe(12345)
        self.assertTrue(safe)


if __name__ == "__main__":
    unittest.main()
