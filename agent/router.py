"""
Execution Router for CyberJarvis.
Classifies instructions into FAST, DIAGNOSTIC, or AGENT execution tiers
BEFORE invoking any LLM call, ensuring sub-second execution for all deterministic tasks.
Includes transcript normalization and wake-word stripping.
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

WAKE_PREFIXES = [
    r"^hey\s+cyberjarvis[\s,:\-]*",
    r"^hey\s+cyber\s+jarvis[\s,:\-]*",
    r"^okay\s+cyberjarvis[\s,:\-]*",
    r"^okay\s+cyber\s+jarvis[\s,:\-]*",
    r"^ok\s+cyberjarvis[\s,:\-]*",
    r"^ok\s+cyber\s+jarvis[\s,:\-]*",
    r"^hello\s+cyberjarvis[\s,:\-]*",
    r"^hello\s+cyber\s+jarvis[\s,:\-]*",
    r"^cyberjarvis[\s,:\-]*",
    r"^cyber\s+jarvis[\s,:\-]*",
    r"^hey\s+jarvis[\s,:\-]*",
    r"^okay\s+jarvis[\s,:\-]*",
    r"^ok\s+jarvis[\s,:\-]*",
    r"^hello\s+jarvis[\s,:\-]*",
    r"^jarvis[\s,:\-]*",
]


def normalize_transcript(text: str) -> str:
    """
    Normalize speech or text input:
    - lowercase
    - trim whitespace & collapse repeated whitespace
    - remove harmless wake words ("cyberjarvis", "hey cyberjarvis", "okay cyberjarvis", etc.)
    - strip leading/trailing punctuation (. , ? ! : ; - ' ")
    """
    if not text:
        return ""

    # 1. Lowercase and collapse spaces
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()

    # 2. Strip wake words in priority order
    for pat in WAKE_PREFIXES:
        new_cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        if new_cleaned != cleaned:
            cleaned = new_cleaned
            break

    # 3. Strip leading and trailing punctuation
    cleaned = re.sub(r"^[\s,:\-.?!'\"]+", "", cleaned)
    cleaned = re.sub(r"[\s,:\-.?!'\"]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


class ExecutionRouter:
    """Classifies user queries into execution tiers with zero LLM overhead."""

    @staticmethod
    def strip_wake_words(text: str) -> str:
        """Strip leading wake words such as 'CyberJarvis, ' or 'Hey Jarvis'."""
        cleaned = text.strip()
        for pat in WAKE_PREFIXES:
            new_cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                break
        return cleaned

    def classify(self, user_input: str) -> RouteType:
        """
        Classify input into RouteType.FAST, RouteType.DIAGNOSTIC, or RouteType.AGENT.
        Executed before any Ollama planner call.
        """
        # Normalize transcript before routing
        normalized = normalize_transcript(user_input)
        if not normalized:
            # Fall back to raw text if normalization cleared it entirely
            normalized = user_input.lower().strip()
            if not normalized:
                return RouteType.FAST

        clean_tokens = set(re.sub(r"[^\w\s]", "", normalized).split())

        # -------------------------------------------------------------
        # Tier 2: DIAGNOSTIC PATH (Checked before Fast Path when diagnostic keywords exist)
        # -------------------------------------------------------------
        if self._is_diagnostic_query(normalized, clean_tokens):
            logger.info(f"Router classified as DIAGNOSTIC: '{user_input}' -> '{normalized}'")
            return RouteType.DIAGNOSTIC

        # -------------------------------------------------------------
        # Tier 1: FAST PATH (Deterministic single-fact/tool queries)
        # -------------------------------------------------------------
        if self._is_fast_path_query(normalized, clean_tokens):
            logger.info(f"Router classified as FAST: '{user_input}' -> '{normalized}'")
            return RouteType.FAST

        # -------------------------------------------------------------
        # Tier 3: TRUE AGENT PATH (Complex reasoning, planning, multi-step tasks)
        # -------------------------------------------------------------
        logger.info(f"Router classified as AGENT: '{user_input}' -> '{normalized}'")
        return RouteType.AGENT

    def _is_diagnostic_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for known diagnostic or troubleshooting queries."""
        # Guard: If query is specifically about architecture, deployment, codebase, or refactoring,
        # it requires True Agent Path reasoning, UNLESS it is specifically about computer/system health.
        if any(w in text for w in ["architecture", "deployment", "repository", "codebase", "refactor"]):
            if not any(w in text for w in ["diagnose my computer", "why is my computer slow"]):
                return False

        # 1. Direct diagnostic keywords for system/hardware
        if any(w in text for w in ["diagnose", "slow", "sluggish", "bottleneck", "troubleshoot"]):
            return True

        # 2. Multi-component diagnostic phrases (English)
        diagnostic_phrases = [
            "diagnose my computer",
            "diagnose my system",
            "diagnose my pc",
            "diagnose computer",
            "analyze my computer",
            "analyse my computer",
            "check my computer",
            "check my system",
            "why is my computer slow",
            "why is my system slow",
            "why is my pc slow",
            "why is the system slow",
            "why is it slow",
            "computer is slow",
            "system is slow",
            "my computer is slow",
            "check system health",
            "check my system health",
            "system health",
            "check cpu ram disk and network",
            "check cpu, ram, disk and network",
            "check cpu ram disk",
            "check cpu, ram, disk",
            "find the main system bottleneck",
            "find system bottleneck",
            "what is the main system bottleneck",
            "what is the bottleneck",
            "system bottleneck",
            "run system diagnosis",
            "system diagnosis",
            "check system status",
            "system status",
            "check my network",
            "check network",
        ]
        if any(p in text for p in diagnostic_phrases):
            return True

        # 3. Hinglish Diagnostic Phrases
        hinglish_diagnostic = [
            "mera computer diagnose karo",
            "mera system check karo",
            "mera computer slow kyu hai",
            "mere system ko analyse karo",
            "mere system ko analyze karo",
            "ram kaun use kar raha hai",
            "mera internet check karo",
            "computer slow kyu hai",
            "system diagnose karo",
            "computer diagnose karo",
            "system check karo",
            "system ko check karo",
            "kaun memory use kar raha hai",
            "kaun ram use kar raha hai",
        ]
        if any(p in text for p in hinglish_diagnostic):
            return True

        # 4. Resource consumer inquiries ("what is consuming the most memory", etc.)
        if any(p in text for p in [
            "what is consuming", "what is using", "what is taking",
            "consuming the most", "using the most", "taking the most",
            "what is taking memory", "taking memory", "using memory", "consuming memory",
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

        # Facts: Profile / Identity (Phase 5)
        profile_phrases = [
            "tell me about me",
            "tell me about myself",
            "who am i",
            "describe me",
            "what do you know about me",
            "my profile",
            "mere bare mein batao",
            "mere baare mein batao",
            "main kaun hoon",
            "tell me my profile",
            "about me",
            "about myself",
        ]
        if any(p in text for p in profile_phrases):
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

        # Time (including short deterministic requests and Hinglish)
        time_matches = [
            "what time is it", "what's the time", "current time", "tell me the time",
            "time which is my computer", "what is my computer time", "computer ka time kya hai",
            "computer time", "system time", "kya time hua hai", "time batao", "samay kya hai",
            "kya samay hai", "samay batao", "tell me time", "what is time"
        ]
        if text in time_matches or any(p in text for p in time_matches) or text == "time" or (
            "time" in text and any(w in text for w in ["computer", "system", "tell", "current", "what", "kya", "which"])
        ):
            return True

        # Date
        if any(p in text for p in ["what date is it", "what's the date", "today's date", "current date", "aaj ki date", "aaj kya tarikh hai"]) or text == "date":
            return True

        # Working directory
        if any(p in text for p in ["what is my current directory", "current directory", "working directory", "where am i", "current folder", "pwd"]):
            return True

        # Project files
        if any(p in text for p in ["show my project files", "list project files", "list files", "show files", "project files dikhao", "files list karo"]):
            return True

        # Git status
        if text in ["git status", "check git status", "what is git status", "check my project git status"] or any(
            p in text for p in ["mere project ka git status", "project ka git status"]
        ):
            return True

        # Internet connectivity (Basic single probe)
        if any(p in text for p in ["check my internet", "check internet", "internet check karo", "check connectivity", "am i online", "is internet working"]):
            return True

        # Single Hardware Metrics
        # CPU
        if any(p in text for p in [
            "check cpu", "check my cpu", "cpu usage", "what is cpu usage",
            "what is my cpu usage", "current cpu usage", "check cpu usage"
        ]):
            return True

        # RAM / Memory
        if any(p in text for p in [
            "check ram", "check my ram", "ram usage", "memory usage", "check memory usage",
            "what is memory usage", "what is my memory usage", "current ram usage", "check memory",
            "check my memory", "what is my ram usage"
        ]):
            return True

        # Disk
        if any(p in text for p in [
            "check disk", "check my disk", "disk usage", "what is disk usage",
            "check disk space", "disk space", "storage space", "check my disk space"
        ]):
            return True

        # Uptime
        if text in ["uptime", "system uptime", "check uptime", "check system uptime"]:
            return True

        return False
