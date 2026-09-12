from pathlib import Path
from typing import Tuple
from config import PROJECT_ROOT

# Strictly prohibited paths
BLOCKED_PATH_PATTERNS = [
    ".ssh",
    "id_rsa",
    "id_ed25519",
    ".gnupg",
    "/etc/shadow",
    "/etc/sudoers",
]

# Protected system PIDs
PROTECTED_PIDS = {0, 1, 2}


def is_path_safe_for_project(path_str: str) -> Tuple[bool, str]:
    """Check if path is securely bounded inside CyberJarvis project."""
    try:
        resolved = (PROJECT_ROOT / path_str).resolve()
        # Ensure it doesn't escape PROJECT_ROOT
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (ValueError, RuntimeError):
        return False, f"Access denied: '{path_str}' is outside the project root."

    resolved_str = str(resolved).lower()
    for blocked in BLOCKED_PATH_PATTERNS:
        if blocked.lower() in resolved_str:
            return False, f"Access denied: '{path_str}' contains sensitive path pattern."

    return True, "Path is safe within project."


def is_process_kill_safe(pid: int) -> Tuple[bool, str]:
    """Ensure sensitive or system-critical PIDs cannot be terminated."""
    if pid in PROTECTED_PIDS:
        return False, f"Cannot kill protected system PID {pid}."
    if pid < 100:
        return False, f"Cannot terminate low-level system daemon PID {pid}."
    return True, f"Process {pid} can be evaluated for termination."
