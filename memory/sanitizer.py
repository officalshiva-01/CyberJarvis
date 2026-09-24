"""
Secret Sanitizer for CyberJarvis Memory Subsystem.
Strictly ensures passwords, tokens, API keys, bearer credentials, and private keys
are never persisted to the SQLite database.
"""
import re
from typing import Any, Dict
from observability.logging import sanitize_data

SECRET_PATTERNS = [
    # Passwords in natural text or assignment: "my password is secret123", "password = 12345"
    r"(?i)\b(?:password|passwd|pwd)\s*(?:is|[:=])\s*['\"]?([^\s'\"]+)",
    # API keys and tokens (handling spaces e.g. "api key: ...")
    r"(?i)\b(?:api[\s_-]?key|access[\s_-]?token|secret[\s_-]?key|auth[\s_-]?token)\s*(?:is|[:=])\s*['\"]?([^\s'\"]+)",
    # OpenAI and similar sk- prefixed keys
    r"\b(sk-[a-zA-Z0-9_\-]{20,})\b",
    # Bearer tokens and Authorization headers
    r"(?i)(?:authorization\s*:\s*)?bearer\s+([a-zA-Z0-9_\-\.]{16,})",
    # GitHub Tokens
    r"(ghp_[a-zA-Z0-9]{36})",
    r"(github_pat_[a-zA-Z0-9_]{22,})",
    # OpenWeatherMap / Generic 32-char hex API keys
    r"\b([a-f0-9]{32})\b",
    # AWS / Cloud Keys
    r"\b(AKIA[0-9A-Z]{16})\b",
    # Private Key Blocks
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----",
]


def sanitize_memory_content(text: str) -> str:
    """Mask secrets and credentials before database insertion or updates."""
    if not text:
        return ""

    sanitized = text
    # 1. Private keys
    sanitized = re.sub(
        r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        sanitized,
    )

    # 2. Bearer tokens
    sanitized = re.sub(
        r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{16,}",
        r"\1[REDACTED_TOKEN]",
        sanitized,
    )

    # 3. Explicit password & token assignments
    for pattern in SECRET_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED_SECRET]", sanitized)

    # 4. Run through baseline observability sanitizer
    sanitized = sanitize_data(sanitized)
    return sanitized if isinstance(sanitized, str) else str(sanitized)


def sanitize_memory_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively scrub sensitive keys from memory metadata dicts."""
    if not isinstance(metadata, dict):
        return {}
    return sanitize_data(metadata)
