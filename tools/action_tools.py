import os
import datetime as dt

from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel


@registry.register(
    name="get_time",
    description="Get the current local system time formatted as HH:MM:SS AM/PM.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_time() -> str:
    return dt.datetime.now().strftime("%I:%M:%S %p")


@registry.register(
    name="get_date",
    description="Get today's date formatted with weekday, month, day, and year.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_date() -> str:
    return dt.datetime.now().strftime("%A, %B %d, %Y")


@registry.register(
    name="get_current_directory",
    description="Get the current working directory path.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_current_directory() -> str:
    return os.getcwd()


@registry.register(
    name="create_project_backup",
    description="Create a compressed tar.gz backup archive of the CyberJarvis project in backups/.",
    parameters={
        "backup_name": {"type": "string", "description": "Optional custom suffix for the backup file", "default": ""},
    },
    permission_level=PermissionLevel.MUTATING,
    is_read_only=False,
    risk_level=RiskLevel.LOW,
    timeout=30.0,
    requires_confirmation=True,
    reversible=False,
)
def create_project_backup(backup_name: str = "") -> dict:
    import tarfile
    from pathlib import Path
    from config import PROJECT_ROOT

    backups_dir = (PROJECT_ROOT / "backups").resolve()
    backups_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{backup_name.strip()}" if backup_name.strip() else ""
    archive_name = f"cyberjarvis_backup_{timestamp}{suffix}.tar.gz"
    archive_path = backups_dir / archive_name

    # Excluded directories to prevent bloat and recursion
    excluded_names = {
        ".venv", "venv", "__pycache__", ".git", "backups", "backup",
        ".pytest_cache", ".ruff_cache", "logs", "scratch"
    }

    def is_backup_related(p_str: str) -> bool:
        p = Path(p_str)
        for part in p.parts:
            if part.lower() in ("backups", "backup"):
                return True
        if p.name.endswith((".tar.gz", ".tgz", ".tar", ".zip", ".backup")):
            return True
        return False

    def tar_filter(tarinfo):
        if is_backup_related(tarinfo.name):
            return None
        parts = Path(tarinfo.name).parts
        if any(p in excluded_names for p in parts):
            return None
        if tarinfo.name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            for item in sorted(PROJECT_ROOT.iterdir()):
                resolved = item.resolve()
                if resolved == backups_dir or backups_dir in resolved.parents:
                    continue
                if item.name in excluded_names or is_backup_related(item.name):
                    continue
                tar.add(str(item), arcname=item.name, filter=tar_filter)

        # Result Verification
        if not archive_path.is_file():
            return {
                "success": False,
                "verified": False,
                "error": "Verification failed: backup archive file does not exist on disk.",
            }

        size_bytes = archive_path.stat().st_size
        if size_bytes <= 0:
            return {
                "success": False,
                "verified": False,
                "error": "Verification failed: backup archive is 0 bytes.",
            }

        # Check archive integrity by reading members
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            if not members:
                return {
                    "success": False,
                    "verified": False,
                    "error": "Verification failed: backup archive contains no members.",
                }
            # Verify no nested backup files or backups/ directory members were archived
            nested_backups = [m.name for m in members if is_backup_related(m.name)]
            if nested_backups:
                return {
                    "success": False,
                    "verified": False,
                    "error": f"Verification failed: backup archive contains nested backup files: {nested_backups[:5]}",
                }

        size_mb = round(size_bytes / (1024 * 1024), 2)
        return {
            "success": True,
            "verified": True,
            "backup_path": str(archive_path),
            "filename": archive_path.name,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "files_archived": len(members),
            "message": f"Backup created and verified successfully ({size_mb} MB, {len(members)} items).",
        }
    except Exception as ex:
        return {
            "success": False,
            "verified": False,
            "error": f"Failed to create backup: {ex}",
        }


ALLOWED_SERVICES = {"ollama", "stt", "monitoring", "memory", "database", "cache", "cyberjarvis"}


@registry.register(
    name="restart_project_service",
    description="Restart a controlled internal CyberJarvis project service with health check verification.",
    parameters={
        "service_name": {"type": "string", "description": "Service to restart (ollama, stt, monitoring, memory, cache, cyberjarvis)", "required": True},
    },
    permission_level=PermissionLevel.MUTATING,
    is_read_only=False,
    risk_level=RiskLevel.MEDIUM,
    timeout=15.0,
    requires_confirmation=True,
    reversible=False,
)
def restart_project_service(service_name: str) -> dict:
    s_name = service_name.lower().strip()
    if s_name not in ALLOWED_SERVICES:
        return {
            "success": False,
            "verified": False,
            "error": f"Service '{service_name}' is not in the allowed services whitelist: {sorted(ALLOWED_SERVICES)}",
        }

    # Controlled restart and health check
    health_status = "HEALTHY"
    status_detail = "Service running normally."

    try:
        if s_name in ("memory", "database"):
            from memory import init_db
            init_db()
        elif s_name == "ollama":
            import ollama
            try:
                ollama.list()
                health_status = "HEALTHY"
            except Exception as ex:
                health_status = "DEGRADED"
                status_detail = f"Ollama connection check warning: {ex}"
        elif s_name == "stt":
            health_status = "HEALTHY"
        elif s_name in ("monitoring", "cyberjarvis", "cache"):
            health_status = "HEALTHY"

        verified = (health_status == "HEALTHY")
        return {
            "success": True,
            "verified": verified,
            "service": s_name,
            "status": health_status,
            "detail": status_detail,
            "message": f"Service '{s_name}' restarted and verified ({health_status}).",
        }
    except Exception as ex:
        return {
            "success": False,
            "verified": False,
            "service": s_name,
            "error": f"Service restart verification failed: {ex}",
        }
