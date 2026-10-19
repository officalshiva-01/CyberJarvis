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
    # 1. Validate max_fail: integer only, positive, bounded 1..10, reject booleans
    if isinstance(max_fail, bool) or not isinstance(max_fail, int):
        return {
            "target": str(test_target),
            "exit_code": -1,
            "success": False,
            "error": f"Invalid max_fail '{max_fail}': must be an integer.",
            "verified": False,
        }
    if max_fail < 1 or max_fail > 10:
        return {
            "target": str(test_target),
            "exit_code": -1,
            "success": False,
            "error": f"Invalid max_fail '{max_fail}': must be between 1 and 10.",
            "verified": False,
        }

    # 2. Validate test_target type
    if not isinstance(test_target, str):
        return {
            "target": str(test_target),
            "exit_code": -1,
            "success": False,
            "error": f"Invalid test_target: expected string, got {type(test_target).__name__}.",
            "verified": False,
        }

    tests_root = (PROJECT_ROOT / "tests").resolve()
    raw_target = test_target.strip()

    if not raw_target:
        # Preserve default target behavior when test_target is empty
        target_str = "tests/test_agent_core.py"
    else:
        # Reject command-line options
        if raw_target.startswith("-"):
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": f"Invalid test_target '{raw_target}': command-line options starting with '-' are not allowed.",
                "verified": False,
            }

        # Reject control characters or null bytes
        if any(ord(c) < 32 or ord(c) == 127 for c in raw_target) or "\x00" in raw_target:
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": "Invalid test_target: control characters are not allowed.",
                "verified": False,
            }

        # Reject path traversal (e.g. ../)
        path_obj = Path(raw_target)
        if ".." in path_obj.parts or ".." in raw_target:
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": f"Path traversal rejected in test_target: '{raw_target}'.",
                "verified": False,
            }

        # Resolve path safely and check boundaries
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            if path_obj.parts and path_obj.parts[0] == "tests":
                resolved = (PROJECT_ROOT / path_obj).resolve()
            elif (PROJECT_ROOT / path_obj).exists():
                resolved = (PROJECT_ROOT / path_obj).resolve()
            elif (tests_root / path_obj).exists():
                resolved = (tests_root / path_obj).resolve()
            else:
                resolved = (tests_root / path_obj).resolve()

        if resolved != tests_root and tests_root not in resolved.parents:
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": f"Invalid test_target '{raw_target}': target must reference a path inside {tests_root}.",
                "verified": False,
            }

        if not resolved.exists():
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": f"Test target '{raw_target}' not found: path does not exist.",
                "verified": False,
            }

        if not (resolved.is_file() or resolved.is_dir()):
            return {
                "target": raw_target,
                "exit_code": -1,
                "success": False,
                "error": f"Test target '{raw_target}' is neither a regular file nor a directory.",
                "verified": False,
            }

        try:
            target_str = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            target_str = str(resolved)

    python_bin = sys.executable or "python"
    cmd = [python_bin, "-m", "pytest", target_str, "-q", f"--maxfail={max_fail}", "--tb=short"]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=50.0,
            shell=False,
        )
        return {
            "target": target_str,
            "exit_code": res.returncode,
            "success": res.returncode == 0,
            "stdout": res.stdout[-1000:].strip() if res.stdout else "",
            "stderr": res.stderr[-500:].strip() if res.stderr else "",
            "verified": True,
        }
    except subprocess.TimeoutExpired:
        return {
            "target": target_str,
            "exit_code": -1,
            "success": False,
            "error": "Test run timed out after 50.0s",
            "verified": False,
        }
    except Exception as ex:
        return {
            "target": target_str,
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
        "max_fail": {"type": "integer", "description": "Maximum failures before stopping", "default": 1},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=60.0,
    requires_confirmation=False,
    reversible=True,
)
def run_tests(test_target: str = "", max_fail: int = 1) -> Dict[str, Any]:
    return run_project_tests(test_target=test_target, max_fail=max_fail)


@registry.register(
    name="test_execution",
    description="Execute test runner (alias for run_project_tests).",
    parameters={
        "test_target": {"type": "string", "description": "Specific test file to run", "default": ""},
        "max_fail": {"type": "integer", "description": "Maximum failures before stopping", "default": 1},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=60.0,
    requires_confirmation=False,
    reversible=True,
)
def test_execution(test_target: str = "", max_fail: int = 1) -> Dict[str, Any]:
    return run_project_tests(test_target=test_target, max_fail=max_fail)


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
