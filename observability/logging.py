import os
import re
import sys
import logging
from pathlib import Path
from typing import Any

from config import LOG_DIR

SENSITIVE_PATTERNS = [
    r"(?i)(api[_-]?key|token|password|secret|authorization|bearer)\s*[:=]\s*['\"]?([^\s'\"]+)",
    r"(ghp_[a-zA-Z0-9]{36})",
    r"(github_pat_[a-zA-Z0-9_]{22,})",
]


def sanitize_data(data: Any) -> Any:
    """Recursively mask sensitive values in strings, dicts, and lists."""
    if isinstance(data, str):
        masked = data
        for pattern in SENSITIVE_PATTERNS:
            masked = re.sub(pattern, r"\1=[REDACTED]", masked)
        return masked
    elif isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if any(s in k.lower() for s in ["key", "token", "password", "secret", "auth"]):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_data(v)
        return cleaned
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data


class SanitizingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        orig = super().format(record)
        return sanitize_data(orig)


def get_logger(name: str = "CyberJarvis") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "cyberjarvis.log"

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = SanitizingFormatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
