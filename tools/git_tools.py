import subprocess
from typing import Dict, Any, List

from tools.registry import registry
from tools.schemas import IsolationMode
from security.policy import RiskLevel, PermissionLevel
from config import PROJECT_ROOT


def _run_git(args: List[str]) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "Git command failed")
    return res.stdout.strip()


@registry.register(
    name="get_git_status",
    description="Get current Git branch, modified files, and untracked files in the project.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
)
def get_git_status() -> Dict[str, Any]:
    try:
        branch = _run_git(["branch", "--show-current"])
        status_raw = _run_git(["status", "--porcelain"])
        changed_files = [line.strip() for line in status_raw.splitlines() if line.strip()]
        return {
            "branch": branch,
            "is_clean": len(changed_files) == 0,
            "modified_count": len(changed_files),
            "changes": changed_files[:15],
        }
    except Exception as ex:
        return {"error": f"Failed to get Git status: {ex}"}


@registry.register(
    name="get_git_recent_commits",
    description="Get recent Git commit history with hashes, authors, dates, and messages.",
    parameters={
        "limit": {"type": "integer", "description": "Number of commits to retrieve (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_git_recent_commits(limit: int = 5) -> List[Dict[str, Any]]:
    try:
        format_str = "%h|%an|%ad|%s"
        output = _run_git(["log", f"-n{limit}", f"--pretty=format:{format_str}", "--date=short"])
        commits = []
        for line in output.splitlines():
            if line.strip():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "date": parts[2],
                        "message": parts[3],
                    })
        return commits
    except Exception as ex:
        return [{"error": f"Failed to get Git commits: {ex}"}]


@registry.register(
    name="git_status",
    description="Get current Git branch and status (alias for get_git_status).",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def git_status() -> Dict[str, Any]:
    return get_git_status()


@registry.register(
    name="git_recent_commits",
    description="Get recent Git commit history (alias for get_git_recent_commits).",
    parameters={
        "limit": {"type": "integer", "description": "Number of commits to retrieve (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def git_recent_commits(limit: int = 5) -> List[Dict[str, Any]]:
    return get_git_recent_commits(limit=limit)


@registry.register(
    name="git_log",
    description="Get recent Git commit history (alias for get_git_recent_commits).",
    parameters={
        "limit": {"type": "integer", "description": "Number of commits to retrieve (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def git_log(limit: int = 5) -> List[Dict[str, Any]]:
    return get_git_recent_commits(limit=limit)

