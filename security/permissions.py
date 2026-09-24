from pathlib import Path
from typing import Tuple
from config import PROJECT_ROOT

# Strictly prohibited paths
BLOCKED_PATH_PATTERNS = [
    ".ssh",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    ".gnupg",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/passwd",
    ".env",
    ".bashrc",
    ".zshrc",
]

# Protected system PIDs
PROTECTED_PIDS = {0, 1, 2}


def is_path_safe_for_project(path_str: str) -> Tuple[bool, str]:
    """Check if path is securely bounded inside CyberJarvis project."""
    if not path_str or not isinstance(path_str, str):
        return False, "Access denied: Invalid path specified."

    # Immediate rejection of traversal fragments
    if ".." in path_str.replace("\\", "/").split("/"):
        # Let resolve check if it escapes, but also check relative_to
        pass

    try:
        p = Path(path_str)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (PROJECT_ROOT / p).resolve()

        # Ensure it doesn't escape PROJECT_ROOT
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (ValueError, RuntimeError):
        return False, f"Access denied: '{path_str}' is outside the project root."

    # Check symlinks: if target itself is a symlink, ensure its target also stays in project
    try:
        check_p = Path(path_str) if Path(path_str).is_absolute() else (PROJECT_ROOT / path_str)
        if check_p.is_symlink():
            link_target = check_p.resolve()
            link_target.relative_to(PROJECT_ROOT.resolve())
    except (ValueError, RuntimeError):
        return False, f"Access denied: symlink '{path_str}' escapes project root."

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
