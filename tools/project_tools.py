import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel
from config import PROJECT_ROOT


@registry.register(
    name="get_test_summary",
    description="Get summary of automated test suites and files in the CyberJarvis project.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_test_summary() -> Dict[str, Any]:
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.is_dir():
        return {"error": "Tests directory not found.", "test_file_count": 0, "test_files": []}

    test_files = sorted([f.name for f in tests_dir.glob("test_*.py")])
    return {
        "test_file_count": len(test_files),
        "test_files": test_files,
        "framework": "pytest",
        "status": "ready",
    }


@registry.register(
    name="test_summary",
    description="Summary of automated test suites (alias for get_test_summary).",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def test_summary() -> Dict[str, Any]:
    return get_test_summary()


@registry.register(
    name="run_project_tests",
    description="Run project automated tests with pytest in a bounded execution sandbox.",
    parameters={
        "test_target": {"type": "string", "description": "Specific test file or directory to run", "default": ""},
        "max_fail": {"type": "integer", "description": "Maximum failures before stopping", "default": 1},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=60.0,
    requires_confirmation=False,
    reversible=True,
)
def run_project_tests(test_target: str = "", max_fail: int = 1) -> Dict[str, Any]:
    python_bin = sys.executable or "python"
    target = test_target.strip() if test_target else "tests/test_agent_core.py"

    cmd = [python_bin, "-m", "pytest", target, "-q", f"--maxfail={max_fail}", "--tb=short"]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=50.0,
        )
        return {
            "target": target,
            "exit_code": res.returncode,
            "success": res.returncode == 0,
            "stdout": res.stdout[-1000:].strip() if res.stdout else "",
            "stderr": res.stderr[-500:].strip() if res.stderr else "",
            "verified": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "target": target,
            "exit_code": -1,
            "success": False,
            "error": "Test run timed out after 50.0s",
            "verified": False,
        }
    except Exception as ex:
        return {
            "target": target,
            "exit_code": -1,
            "success": False,
            "error": str(ex),
            "verified": False,
        }


@registry.register(
    name="run_tests",
    description="Run automated pytest test suite in the CyberJarvis project.",
    parameters={
        "test_target": {"type": "string", "description": "Specific test file to run (default: test_agent_core.py)", "default": ""},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=60.0,
    requires_confirmation=False,
    reversible=True,
)
def run_tests(test_target: str = "") -> Dict[str, Any]:
    return run_project_tests(test_target=test_target)


@registry.register(
    name="test_execution",
    description="Execute test runner (alias for run_project_tests).",
    parameters={
        "test_target": {"type": "string", "description": "Specific test file to run", "default": ""},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=60.0,
    requires_confirmation=False,
    reversible=True,
)
def test_execution(test_target: str = "") -> Dict[str, Any]:
    return run_project_tests(test_target=test_target)


@registry.register(
    name="read_project_log",
    description="Read recent log entries from the CyberJarvis log file with sensitive data masked.",
    parameters={
        "lines": {"type": "integer", "description": "Number of recent lines to read (default 50)", "default": 50},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def read_project_log(lines: int = 50) -> Dict[str, Any]:
    from observability.logging import sanitize_data
    log_file = PROJECT_ROOT / "logs" / "cyberjarvis.log"
    if not log_file.is_file():
        return {"error": "Log file not found.", "lines": [], "errors": []}

    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
        recent = [sanitize_data(l) for l in recent]

        errors = [l for l in recent if any(err in l for err in ("ERROR", "CRITICAL", "Traceback", "Exception"))]
        warnings = [l for l in recent if "WARNING" in l]

        return {
            "line_count": len(recent),
            "errors": errors[-10:],
            "warnings": warnings[-10:],
            "recent_lines": recent[-20:],
        }
    except Exception as ex:
        return {"error": f"Failed to read log: {ex}", "lines": [], "errors": []}
