from pathlib import Path
from typing import Dict, Any, List

from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel
from security.permissions import is_path_safe_for_project
from config import PROJECT_ROOT, MAX_FILE_READ_CHARS


@registry.register(
    name="list_project_files",
    description="List files and directories inside the CyberJarvis project.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def list_project_files() -> List[Dict[str, Any]]:
    if not PROJECT_ROOT.exists():
        return [{"error": "CyberJarvis project directory does not exist."}]

    items = []
    for path in sorted(PROJECT_ROOT.iterdir()):
        # Skip hidden files except git/venv
        items.append({
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        })
    return items


@registry.register(
    name="read_project_file",
    description="Read contents of a text file inside the CyberJarvis project safely.",
    parameters={
        "filename": {"type": "string", "description": "Relative filename to read (e.g. main.py, config.py)", "required": True},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def read_project_file(filename: str) -> Dict[str, Any]:
    safe, reason = is_path_safe_for_project(filename)
    if not safe:
        return {"error": reason}

    target = (PROJECT_ROOT / filename).resolve()
    if not target.exists():
        return {"error": f"File '{filename}' not found."}
    if not target.is_file():
        return {"error": f"'{filename}' is not a regular file."}

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        truncated = False
        if len(content) > MAX_FILE_READ_CHARS:
            content = content[:MAX_FILE_READ_CHARS] + "\n\n[Output truncated for safety.]"
            truncated = True

        return {
            "filename": filename,
            "content": content,
            "truncated": truncated,
            "length": len(content),
        }
    except Exception as ex:
        return {"error": f"Failed to read '{filename}': {ex}"}


@registry.register(
    name="delete_project_file",
    description="Delete a temporary or unused file in the project. (MUTATING, REQUIRES CONFIRMATION)",
    parameters={
        "filename": {"type": "string", "description": "Relative filename to delete", "required": True},
    },
    permission_level=PermissionLevel.MUTATING,
    is_read_only=False,
    risk_level=RiskLevel.HIGH,
)
def delete_project_file(filename: str) -> Dict[str, Any]:
    safe, reason = is_path_safe_for_project(filename)
    if not safe:
        return {"error": reason}

    target = (PROJECT_ROOT / filename).resolve()
    if not target.exists():
        return {"error": f"File '{filename}' does not exist."}
    if not target.is_file():
        return {"error": f"Cannot delete '{filename}': not a regular file."}

    try:
        target.unlink()
        return {"success": True, "message": f"Successfully deleted '{filename}'."}
    except Exception as ex:
        return {"error": f"Failed to delete '{filename}': {ex}"}
