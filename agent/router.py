"""
Execution Router for CyberJarvis.
Classifies instructions into FAST, DIAGNOSTIC, or AGENT execution tiers
BEFORE invoking any LLM call, ensuring sub-second execution for all deterministic tasks.
"""
import re
from enum import Enum
from typing import Set
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Router")


class RouteType(str, Enum):
    FAST = "FAST"
    DIAGNOSTIC = "DIAGNOSTIC"
    AGENT = "AGENT"


# Precompiled word sets and phrases
EXIT_PHRASES: Set[str] = {"exit", "quit", "bye", "goodbye", "stop", "terminate", "band karo"}
GREETING_WORDS: Set[str] = {"hello", "hi", "hey", "namaste", "greetings", "yo"}

WAKE_WORDS = [
    r"^cyberjarvis[\s,:\-]*",
    r"^hey jarvis[\s,:\-]*",
    r"^hello jarvis[\s,:\-]*",
    r"^jarvis[\s,:\-]*",
    r"^ok jarvis[\s,:\-]*",
]


class ExecutionRouter:
    """Classifies user queries into execution tiers with zero LLM overhead."""

    @staticmethod
    def strip_wake_words(text: str) -> str:
        """Strip leading wake words such as 'CyberJarvis, ' or 'Hey Jarvis'."""
        cleaned = text.strip()
        for pattern in WAKE_WORDS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def classify(self, user_input: str) -> RouteType:
        """
        Classify input into RouteType.FAST, RouteType.DIAGNOSTIC, or RouteType.AGENT.
        Executed before any Ollama planner call.
        """
        raw_text = user_input.strip()
        if not raw_text:
            return RouteType.FAST

        stripped = self.strip_wake_words(raw_text).lower()
        clean_tokens = set(re.sub(r"[^\w\s]", "", stripped).split())

        # -------------------------------------------------------------
        # Tier 2: DIAGNOSTIC PATH (Checked before Fast Path when diagnostic keywords exist)
        # -------------------------------------------------------------
        if self._is_diagnostic_query(stripped, clean_tokens):
            logger.info(f"Router classified as DIAGNOSTIC: '{raw_text}'")
            return RouteType.DIAGNOSTIC

        # -------------------------------------------------------------
        # Tier 1: FAST PATH (Deterministic single-fact/tool queries)
        # -------------------------------------------------------------
        if self._is_fast_path_query(stripped, clean_tokens):
            logger.info(f"Router classified as FAST: '{raw_text}'")
            return RouteType.FAST

        # -------------------------------------------------------------
        # Tier 3: TRUE AGENT PATH (Complex reasoning, planning, multi-step tasks)
        # -------------------------------------------------------------
        logger.info(f"Router classified as AGENT: '{raw_text}'")
        return RouteType.AGENT

    def _is_diagnostic_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for known diagnostic or troubleshooting queries."""
        # 1. Direct diagnostic keywords
        if any(w in text for w in ["diagnose", "slow", "sluggish", "bottleneck", "troubleshoot"]):
            return True

        # 2. Multi-component diagnostic phrases
        diagnostic_phrases = [
            "check cpu ram disk and network",
            "check cpu, ram, disk and network",
            "check cpu ram disk",
            "check cpu, ram, disk",
            "find the main system bottleneck",
            "find system bottleneck",
            "what is the main system bottleneck",
            "what is the bottleneck",
            "system bottleneck",
            "check my system health",
            "check system health",
            "system health",
            "check system status",
            "check my system",
            "system status",
            "run system diagnosis",
            "system diagnosis",
            "why is my computer slow",
            "why is my pc slow",
            "why is the system slow",
            "why is it slow",
            "computer is slow",
            "system is slow",
            "my computer is slow",
        ]
        if any(p in text for p in diagnostic_phrases):
            return True

        # 3. Resource consumer inquiries ("what is consuming the most memory", etc.)
        if any(p in text for p in [
            "what is consuming", "what is using", "what is taking",
            "consuming the most", "using the most", "taking the most",
            "top processes", "top process", "most memory", "most cpu",
            "which process is consuming", "which process is using",
            "who is consuming", "heavy process", "offender"
        ]):
            return True

        return False

    def _is_fast_path_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for deterministic single-tool or memory fact queries."""
        # Exit commands
        if text in EXIT_PHRASES or tokens.issubset(EXIT_PHRASES) or (len(tokens) == 1 and tokens.intersection(EXIT_PHRASES)):
            return True

        # Greetings
        if text in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "namaste"] or (
            len(tokens) <= 2 and bool(tokens.intersection(GREETING_WORDS))
        ):
            return True

        # Assistant identity
        if any(p in text for p in ["who are you", "what are you", "what is cyberjarvis", "introduce yourself", "tum kaun ho"]):
            return True

        # Facts: Name
        if any(p in text for p in ["what is my name", "what's my name", "who am i", "mera naam kya hai", "tell me my name", "do you know my name"]):
            return True

        # Facts: Goal
        if any(p in text for p in ["what is my goal", "what's my goal", "mera goal kya hai", "what do i want to become", "mera lakshya kya hai"]):
            return True

        # Facts: Project
        if any(p in text for p in ["what project am i working on", "what is my project", "what's my project", "what project am i building", "mera project kya hai", "project batao"]):
            return True

        # Time
        if any(p in text for p in ["what time is it", "what's the time", "current time", "kya time hua hai", "time batao", "samay kya hai"]) or text == "time":
            return True

        # Date
        if any(p in text for p in ["what date is it", "what's the date", "today's date", "current date", "aaj ki date", "aaj kya tarikh hai"]) or text == "date":
            return True

        # Working directory
        if any(p in text for p in ["current directory", "working directory", "where am i", "current folder", "pwd"]):
            return True

        # Project files
        if any(p in text for p in ["show my project files", "list project files", "list files", "show files", "project files dikhao", "files list karo"]):
            return True

        # Git status
        if text in ["git status", "check git status", "what is git status", "check my project git status"] or any(
            p in text for p in ["mere project ka git status", "project ka git status"]
        ):
            return True

        # Internet connectivity
        if any(p in text for p in ["check my internet", "check internet", "internet check karo", "check connectivity", "am i online", "is internet working"]):
            return True

        # Single Hardware Metrics
        # CPU
        if any(p in text for p in ["check cpu", "check my cpu", "cpu usage", "what is cpu usage", "what is my cpu usage", "current cpu usage", "check cpu usage"]):
            return True

        # RAM / Memory
        if any(p in text for p in [
            "check ram", "check my ram", "ram usage", "memory usage", "check memory usage",
            "what is memory usage", "what is my memory usage", "current ram usage", "check memory",
            "check my memory", "what is my ram usage"
        ]):
            return True

        # Disk
        if any(p in text for p in ["check disk", "check my disk", "disk usage", "what is disk usage", "check disk space", "disk space", "storage space", "check my disk space"]):
            return True

        # Uptime
        if text in ["uptime", "system uptime", "check uptime", "check system uptime"]:
            return True

        return False
