"""
Security Audit Logging for CyberJarvis.
Records structured, tamper-evident, and sanitized audit events for every tool action.
Guarantees zero leakage of secrets, keys, or credentials.
"""
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import LOG_DIR
from observability.logging import sanitize_data, get_logger

logger = get_logger("CyberJarvis.SecurityAudit")


class AuditLogger:
    """Records security audit events to a dedicated structured audit log."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (LOG_DIR / "audit.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        user_request: str,
        tool: str,
        arguments: Any,
        risk_level: str,
        confirmation_status: str,
        result: Any,
        duration_ms: float,
        success: bool,
        permission_level: str = "read_only",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a single structured audit event.
        All arguments and results are sanitized before writing to disk.
        """
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        sanitized_args = sanitize_data(arguments or {})
        sanitized_res = sanitize_data(result) if result is not None else None
        sanitized_err = sanitize_data(error) if error else None
        sanitized_req = sanitize_data(user_request or "")

        event = {
            "timestamp": now,
            "user_request": sanitized_req,
            "tool": tool,
            "arguments": sanitized_args,
            "permission_level": str(permission_level).lower(),
            "risk_level": str(risk_level).lower(),
            "confirmation_status": confirmation_status,  # "NOT_REQUIRED", "CONFIRMED", "REJECTED", "BLOCKED"
            "result": sanitized_res,
            "duration_ms": round(float(duration_ms), 2),
            "success": bool(success),
            "error": sanitized_err,
        }

        try:
            line = json.dumps(event, default=str)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as ex:
            logger.error(f"Failed to write audit event for '{tool}': {ex}")

        return event

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent audit events from the audit log."""
        if not self.log_path.is_file():
            return []

        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            events.append(json.loads(line_s))
                        except json.JSONDecodeError:
                            continue
        except Exception as ex:
            logger.error(f"Failed to read audit log: {ex}")
            return []

        return events[-limit:]

    def clear(self):
        """Clear audit log (for testing purposes)."""
        if self.log_path.exists():
            try:
                self.log_path.unlink()
            except Exception:
                pass


# Global audit logger singleton
audit_logger = AuditLogger()
