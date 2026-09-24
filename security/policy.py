from enum import Enum
from typing import NamedTuple, Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionLevel(str, Enum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    PRIVILEGED = "privileged"
    ADMIN = "admin"


DANGEROUS_SHELL_PATTERNS = [
    r"\brm\s+-(?:rf|fr|r|f)\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bchmod\s+-[rR]\s+777\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # fork bomb
    r"curl\s+[^|]+\|\s*(?:ba)?sh",
    r"wget\s+[^|]+\|\s*(?:ba)?sh",
    r"\b(?:bash|sh|zsh)\s+-c\b",
    r"\bexecute\s+(?:arbitrary\s+)?(?:shell|bash|command)\b",
    r"\brun\s+(?:this\s+)?arbitrary\s+(?:shell|command)\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
]


def is_arbitrary_shell_attempt(query: str) -> tuple[bool, str]:
    """Detect if a user query is attempting to execute arbitrary shell commands."""
    import re
    q = query.lower()
    for pat in DANGEROUS_SHELL_PATTERNS:
        if re.search(pat, q):
            return True, f"Security Violation: Dangerous pattern or arbitrary shell execution detected matching '{pat}'."
    return False, ""


class PolicyDecision(NamedTuple):
    allowed: bool
    requires_confirmation: bool
    reason: str


class SecurityPolicy:
    """
    Evaluates actions against security guardrails.
    Rules are guardrails, not the brain.
    """

    def __init__(self, allow_auto_read: bool = True, allow_auto_mutating: bool = False):
        self.allow_auto_read = allow_auto_read
        self.allow_auto_mutating = allow_auto_mutating

    def evaluate(
        self,
        tool_name: str,
        permission_level: PermissionLevel,
        risk_level: RiskLevel,
        is_read_only: bool,
    ) -> PolicyDecision:
        # Critical risk operations are blocked by default unless explicitly audited
        if risk_level == RiskLevel.CRITICAL:
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Operation '{tool_name}' is classified as CRITICAL risk and blocked.",
            )

        # High risk requires explicit user confirmation
        if risk_level == RiskLevel.HIGH:
            return PolicyDecision(
                allowed=True,
                requires_confirmation=True,
                reason=f"High risk operation '{tool_name}' requires explicit user confirmation.",
            )

        # Mutating operations
        if not is_read_only or permission_level == PermissionLevel.MUTATING:
            if self.allow_auto_mutating:
                return PolicyDecision(allowed=True, requires_confirmation=False, reason="Auto mutating permitted.")
            return PolicyDecision(
                allowed=True,
                requires_confirmation=True,
                reason=f"Mutating operation '{tool_name}' requires confirmation.",
            )

        # Read-only operations
        if is_read_only and risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
            return PolicyDecision(
                allowed=True,
                requires_confirmation=False,
                reason="Safe read-only operation.",
            )

        return PolicyDecision(
            allowed=False,
            requires_confirmation=False,
            reason=f"Access denied by security policy for '{tool_name}'.",
        )
