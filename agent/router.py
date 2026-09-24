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
    INFORMATION = "INFORMATION"
    AGENT = "AGENT"
    CLARIFICATION = "CLARIFICATION"


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
    r"^cyberjarvis\b(?!\s+(?:architecture|codebase|project|subsystem)\b)[\s,:\-]*",
    r"^cyber\s+jarvis\b(?!\s+(?:architecture|codebase|project|subsystem)\b)[\s,:\-]*",
    r"^hey\s+jarvis[\s,:\-]*",
    r"^okay\s+jarvis[\s,:\-]*",
    r"^ok\s+jarvis[\s,:\-]*",
    r"^hello\s+jarvis[\s,:\-]*",
    r"^jarvis[\s,:\-]*",
]

INCOMPLETE_EXACT_FRAGMENTS: Set[str] = {
    "",
    "what",
    "what is",
    "whats",
    "what's",
    "how",
    "how is",
    "why",
    "why is",
    "who",
    "who is",
    "tell me",
    "tell me about",
    "check",
    "check my",
    "diagnose",
    "diagnose my",
    "and diagnose",
    "and check",
    "analyze",
    "analyze my",
    "analyse",
    "analyse my",
    "and",
    "also",
    "please",
    "so",
    "then",
    "uh",
    "um",
    "can you",
    "could you",
    "is",
    "are",
    "which",
    "where",
    "where is",
    "when",
    "when is",
    "look at",
    "show",
    "show me",
    "list",
}

ALLOWED_SINGLE_WORDS: Set[str] = {
    "exit", "quit", "bye", "goodbye", "stop", "terminate",
    "hello", "hi", "hey", "namaste", "greetings", "yo",
    "time", "date", "uptime", "pwd",
}


def normalize_transcript(text: str) -> str:
    """
    Normalize speech or text input:
    - lowercase
    - trim whitespace & collapse repeated whitespace
    - remove harmless wake words ("cyberjarvis", "hey cyberjarvis", "okay cyberjarvis", etc.)
    - remove harmless leading conjunctions/fillers ("and", "also", "okay", "ok", "so", "then", "please", "uh", "um")
    - normalize speech artifacts like 'diagnosis my computer' -> 'diagnose my computer'
    - strip leading/trailing punctuation (. , ? ! : ; - ' ")
    """
    if not text:
        return ""

    # 1. Lowercase and collapse spaces
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()

    # 2. Strip wake words and leading conjunctions/fillers in loop
    changed = True
    while changed:
        changed = False
        # Strip wake words
        for pat in WAKE_PREFIXES:
            new_cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True
                break
        # Strip leading conjunctions/fillers (only at start of command)
        new_cleaned = re.sub(r"^(?:and|also|okay|ok|so|then|please|uh|um)\b[\s,:\-]*", "", cleaned, flags=re.IGNORECASE).strip()
        if new_cleaned != cleaned:
            cleaned = new_cleaned
            changed = True

    # 3. Speech artifact recovery:
    # ASR variants for diagnosis commands:
    # "diagnosis by computer", "diagnosis my computer", "diagnostic my computer", "diagnosis computer" -> "diagnose my computer"
    cleaned = re.sub(
        r"\b(?:diagnosis|diagnostic|diagnostics)\s+(?:by|my|the|of|for|on\s+my|in\s+my)?\s*(computer|system|pc|laptop|machine)\b",
        r"diagnose my \1",
        cleaned,
    )
    # ASR preposition corruptions: "diagnose by computer" -> "diagnose my computer"
    cleaned = re.sub(r"\bdiagnose\s+by\s+(computer|system|pc|laptop|machine)\b", r"diagnose my \1", cleaned)

    # Normalize British/American spelling: "analyse" -> "analyze"
    cleaned = re.sub(r"\banalyse\b", "analyze", cleaned)

    # Conservative speech-artifact normalization for highly probable CyberJarvis variants:
    # Normalize "cyber jarvis" -> "cyberjarvis"
    cleaned = re.sub(r"\bcyber\s+jarvis\b", "cyberjarvis", cleaned)

    # "analyse/analyze my cyber service" -> "analyze my cyberjarvis architecture"
    # Only normalize when the surrounding phrase strongly indicates the CyberJarvis project:
    cleaned = re.sub(
        r"\b(analyze|inspect|audit|review|explore|examine)\s+(?:my\s+|our\s+|the\s+)?cyber\s+service(\s+architecture)?\b",
        r"\1 my cyberjarvis architecture",
        cleaned,
    )
    cleaned = re.sub(r"\bcyber\s+service\s+(architecture|codebase|project|subsystem)\b", r"cyberjarvis \1", cleaned)

    # 4. Strip leading and trailing punctuation
    cleaned = re.sub(r"^[\s,:\-.?!'\"]+", "", cleaned)
    cleaned = re.sub(r"[\s,:\-.?!'\"]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def is_incomplete_transcript(text: str) -> bool:
    """
    Validates whether an STT transcript is obviously incomplete or a trailing fragment.
    Incomplete commands must return a clarification state and NEVER trigger expensive reasoning (Ollama).
    """
    if not text:
        return True

    raw_clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not raw_clean or raw_clean in INCOMPLETE_EXACT_FRAGMENTS:
        return True

    norm = normalize_transcript(text)
    if not norm or norm in INCOMPLETE_EXACT_FRAGMENTS:
        return True

    norm_clean = re.sub(r"[^\w\s]", "", norm).strip()
    if not norm_clean or norm_clean in INCOMPLETE_EXACT_FRAGMENTS:
        return True

    tokens = norm_clean.split()
    if not tokens:
        return True

    # Single-word guard: only specific whitelist words are valid standalone commands
    if len(tokens) == 1 and tokens[0] not in ALLOWED_SINGLE_WORDS:
        return True

    # Incomplete prefix patterns:
    # 1. Dangling question starters: "what is", "why is", "how is", "where is", etc.
    if re.match(r"^(?:what|why|who|how|where|when)\s+(?:is|are|was|were|do|does|did|can|could|am)$", norm_clean):
        return True

    # 2. Dangling command starters: "tell me about", "check my", "diagnose my", etc.
    if re.match(r"^(?:tell\s+me\s+about|check\s+my|diagnose\s+my|analyze\s+my|analyse\s+my|look\s+at)$", norm_clean):
        return True

    return False


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
        Classify input into RouteType.FAST, RouteType.DIAGNOSTIC, RouteType.AGENT, or RouteType.CLARIFICATION.
        Executed before any Ollama planner call.
        """
        # Guard: Incomplete transcripts immediately route to CLARIFICATION
        if is_incomplete_transcript(user_input):
            logger.info(f"Router classified as CLARIFICATION: '{user_input}'")
            return RouteType.CLARIFICATION

        # Normalize transcript before routing
        normalized = normalize_transcript(user_input)
        if not normalized:
            normalized = user_input.lower().strip()
            if not normalized or is_incomplete_transcript(normalized):
                return RouteType.CLARIFICATION

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
        # Tier 3a: TRUE AGENT PATH (Local system, architecture, codebase, file inspection/mutation)
        # -------------------------------------------------------------
        if self._is_agent_query(normalized, clean_tokens):
            logger.info(f"Router classified as AGENT: '{user_input}' -> '{normalized}'")
            return RouteType.AGENT

        # -------------------------------------------------------------
        # Tier 3b: INFORMATION / DIRECT-LLM PATH (Conceptual, educational, general knowledge)
        # -------------------------------------------------------------
        if self._is_information_query(normalized, clean_tokens):
            logger.info(f"Router classified as INFORMATION: '{user_input}' -> '{normalized}'")
            return RouteType.INFORMATION

        # Default for unclassified open-ended queries is INFORMATION (lightweight, zero tools)
        logger.info(f"Router classified as default INFORMATION: '{user_input}' -> '{normalized}'")
        return RouteType.INFORMATION

    def _is_agent_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for genuine agentic tasks requiring local tools, planning, and evaluation."""
        # External Live API actions (Weather & GitHub tools)
        if any(w in text for w in ["weather", "forecast"]) and not any(text == w or text.startswith(w + " ") for w in ["what is weather forecasting", "define weather", "explain weather"]):
            return True

        if "github" in text and not any(text == w or text.startswith(w + " ") for w in ["what is github", "explain github", "define github"]):
            return True

        # Queries explicitly targeting local project, codebase, or multi-step execution
        agent_phrases = [
            "architectur",
            "subsystem",
            "project structure",
            "codebase",
            "refactor",
            "deployment",
            "deploy",
            "repository",
            "repo",
            "inspect files",
            "diagnose repository",
            "multi-step",
            "deadlock",
            "diagnostic log",
            "debug",
            "tests failing",
            "test failing",
            "failing tests",
            "why are my tests failing",
            "tests error",
            "find the problem",
            "problem in my project",
            "analyze this error",
            "analyze error",
            "current issue",
            "backup",
            "create a backup",
            "create backup",
            "run my tests",
            "run tests",
            "run the tests",
            "run project tests",
            "execute tests",
            "restart service",
        ]

        # Specific CyberJarvis / Project inspection requests
        # A phrase should only become AGENT when there is evidence that the user wants
        # CyberJarvis/project inspection, debugging, deployment, or a multi-step task.
        inspection_verbs = ("analyze", "inspect", "review", "audit", "explore", "examine", "investigate", "debug", "troubleshoot", "find")
        if any(v in text for v in inspection_verbs):
            if "cyberjarvis" in text:
                return True
            if any(t in text for t in ["project", "codebase", "subsystem", "architecture", "repository", "repo", "error", "issue", "bug", "test", "tests"]):
                if any(ctx in text for ctx in ["my ", "our ", "this project", "the project", "local", "current", "this "]):
                    return True

        # If it's a general informational question ("what is github", "explain repository", "what is cybersecurity"), don't treat as agent
        info_starters = ("what is ", "what are ", "explain ", "how to ", "how do i ", "how can i ", "how should i ", "tell me about ", "define ", "how does ")
        has_local_context = any(w in text for w in ["my ", "our ", "this project", "cyberjarvis", "local", "here", "work tree"])

        if any(text.startswith(s) for s in info_starters) and not has_local_context:
            return False

        if any(p in text for p in agent_phrases):
            return True

        # 2. File / Code mutations or multi-step execution tasks
        if any(w in text for w in ["git branch", "git commit", "git diff", "git log", "recent commits"]):
            return True

        if any(w in text for w in ["search files", "find files", "run tests", "run my tests", "run the tests", "execute tests", "backup", "execute script"]):
            return True

        return False

    def _is_information_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for informational, conceptual, or educational questions that do not require tools."""
        # Common informational query starters
        if any(text.startswith(p) for p in [
            "how to", "how do i", "how can i", "how should i", "how would i",
            "how does", "how do", "how can", "how would",
            "explain", "what is", "what are", "what does", "define",
            "why is", "why does", "tell me about", "describe",
            "difference between", "comparison between", "compare",
            "can you explain", "give me advice", "tips for", "guide to",
            "roadmap", "tutorial", "best practices",
        ]):
            return True

        # Learning & Career queries
        if any(w in text for w in [
            "become a developer", "become developer", "learn programming",
            "learn coding", "learn backend", "learn frontend", "learn devops",
            "learn python", "learn go", "learn rust", "career",
        ]):
            return True

        return False

    def _is_diagnostic_query(self, text: str, tokens: Set[str]) -> bool:
        """Check for known diagnostic or troubleshooting queries."""
        # Guard: If query is specifically about architecture, deployment, codebase, or refactoring,
        # it requires True Agent Path reasoning, UNLESS it is specifically about computer/system health.
        if any(w in text for w in ["architecture", "deployment", "repository", "codebase", "refactor", "cyberjarvis"]):
            if not any(w in text for w in ["diagnose my computer", "why is my computer slow"]):
                return False

        # 1. Direct diagnostic keywords for system/hardware
        system_targets = ["computer", "system", "pc", "laptop", "machine", "hardware"]
        if any(w in text for w in ["slow", "sluggish", "bottleneck", "troubleshoot"]):
            return True

        if any(w in text for w in ["diagnose", "diagnosis", "diagnostic"]):
            if any(t in text for t in system_targets) or text.startswith(("diagnose", "run ", "system diagnosis", "computer diagnosis")):
                return True

        # 2. Multi-component diagnostic phrases (English)
        diagnostic_phrases = [
            "diagnose my computer",
            "diagnose the computer",
            "diagnose computer",
            "diagnose my system",
            "diagnose the system",
            "diagnose system",
            "diagnosis by computer",
            "diagnosis my computer",
            "diagnosis the computer",
            "diagnosis computer",
            "diagnostic my computer",
            "diagnostic computer",
            "run computer diagnosis",
            "run system diagnosis",
            "computer diagnosis",
            "system diagnosis",
            "analyze my computer",
            "analyse my computer",
            "check my computer",
            "check the computer",
            "check computer",
            "check my system",
            "check the system",
            "check system",
            "check computer health",
            "check my computer health",
            "check system health",
            "check my system health",
            "system health check",
            "computer health check",
            "system health",
            "computer health",
            "why is my computer slow",
            "why is my system slow",
            "why is my pc slow",
            "why is the computer slow",
            "why is the system slow",
            "why is it slow",
            "computer is slow",
            "system is slow",
            "my computer is slow",
            "what is wrong with my computer",
            "what is wrong with the computer",
            "what is wrong with my system",
            "what is wrong with the system",
            "what is wrong with my pc",
            "what is wrong with computer",
            "whats wrong with my computer",
            "what's wrong with my computer",
            "what is wrong with it",
            "check cpu ram disk and network",
            "check cpu, ram, disk and network",
            "check cpu ram disk",
            "check cpu, ram, disk",
            "find the main system bottleneck",
            "find system bottleneck",
            "what is the main system bottleneck",
            "what is the bottleneck",
            "system bottleneck",
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

        # 4. Resource consumer inquiries ("what is consuming memory", "who is using my RAM", etc.)
        if any(p in text for p in [
            "what is consuming", "what is using", "what is taking",
            "who is using", "who is consuming",
            "consuming the most", "using the most", "taking the most",
            "what is taking memory", "taking memory", "using memory", "consuming memory",
            "top processes", "top process", "most memory", "most cpu",
            "which process is consuming", "which process is using",
            "heavy process", "offender"
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
        if any(p in text for p in ["who are you", "what are you", "what is cyberjarvis", "introduce yourself", "tum kaun ho", "what is your name", "what's your name", "tumhara naam kya hai"]):
            return True

        # Facts: Profile / Identity (Phase 5 & 7)
        profile_phrases = [
            "tell me about me",
            "tell me about myself",
            "who am i",
            "describe me",
            "what do you know about me",
            "what do you remember about me",
            "what do you remember",
            "what all do you remember",
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

        # Project Status & Continuity (Phase 7)
        status_phrases = [
            "what is the current cyberjarvis project status",
            "what is the current cyberjarvis status",
            "what is current cyberjarvis project status",
            "what is cyberjarvis project status",
            "what is current project status",
            "what is project status",
            "current project status",
            "cyberjarvis project status",
            "cyberjarvis status",
            "project status",
        ]
        if any(p in text for p in status_phrases):
            return True

        # Task Continuity & Decisions (Phase 7)
        continuity_phrases = [
            "what were we working on",
            "what was i working on",
            "what are we working on",
            "continue my previous task",
            "continue previous task",
            "continue task",
            "what did we decide about stt",
            "what did we decide",
            "what was decided",
            "what decisions did we make",
        ]
        if any(p in text for p in continuity_phrases):
            return True

        # Memory management commands (Phase 7)
        if text.startswith(("remember this", "remember that", "remember :", "remember:", "forget this", "forget that", "forget :", "forget:")):
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
