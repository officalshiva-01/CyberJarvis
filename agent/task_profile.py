"""
Agent Task Profiler for CyberJarvis.
Classifies AGENT queries into specialized deterministic task profiles
(such as PROJECT_ARCHITECTURE) before invoking the general LLM planner.
"""
from enum import Enum
import re
from observability.logging import get_logger

logger = get_logger("CyberJarvis.TaskProfile")


class AgentTaskProfile(str, Enum):
    PROJECT_ARCHITECTURE = "PROJECT_ARCHITECTURE"
    PROJECT_INSPECTION = "PROJECT_INSPECTION"
    PROJECT_DEBUGGING = "PROJECT_DEBUGGING"
    DEPLOYMENT_PLANNING = "DEPLOYMENT_PLANNING"
    GENERAL_PLANNING = "GENERAL_PLANNING"


def classify_agent_task_profile(query: str) -> AgentTaskProfile:
    """
    Classify an AGENT query into an explicit task profile.
    Recognizes known project analysis intents deterministically to avoid
    costly and ungrounded general-purpose planner calls.
    """
    if not query:
        return AgentTaskProfile.GENERAL_PLANNING

    q = query.lower().strip()

    # 1. DEPLOYMENT_PLANNING
    # Examples:
    # - "plan deployment for CyberJarvis"
    # - "how should I deploy CyberJarvis"
    # - "prepare CyberJarvis for Linux deployment"
    # - "create a deployment architecture"
    # - "plan a deployment architecture"
    # - "deploy to aws", "deploy to kubernetes", "ecs deployment"
    if any(p in q for p in [
        "plan deployment", "plan a deployment", "how should i deploy", "how to deploy",
        "prepare cyberjarvis for linux deployment", "prepare for deployment", "prepare deployment",
        "deployment architecture", "deployment plan", "deployment planning",
        "deploy to aws", "deploy to kubernetes", "deploy to docker", "ecs deployment",
        "deployment to", "deploy cyberjarvis"
    ]) or ("deploy" in q and any(w in q for w in ["cyberjarvis", "project", "architecture", "linux", "cloud", "plan"])):
        logger.info(f"Agent Task Profile: DEPLOYMENT_PLANNING for query: '{query}'")
        return AgentTaskProfile.DEPLOYMENT_PLANNING

    # 2. PROJECT_DEBUGGING
    # Examples:
    # - "debug my CyberJarvis project"
    # - "why are my tests failing"
    # - "find the problem in my project"
    # - "analyze this error"
    # - "debug the current issue"
    # - "diagnose a repository problem", "debug deadlock", "investigate error log"
    if any(p in q for p in [
        "debug my cyberjarvis", "debug cyberjarvis", "debug this project", "debug my project",
        "debug the project", "debug project", "debug current issue", "debug the current issue",
        "debug this issue", "debug issue", "why are my tests failing", "why are tests failing",
        "tests failing", "test failing", "failing tests", "tests are failing", "tests broken",
        "find the problem", "problem in my project", "problem in the project", "problem in this project",
        "analyze this error", "analyze error", "debug this error", "debug error",
        "diagnose a repository", "diagnose repository", "repository problem",
        "debug deadlock", "diagnostic log", "deadlock in", "fix deadlock", "investigate error log",
        "troubleshoot project", "troubleshoot my project", "find bug in project"
    ]) or (re.search(r"\bdebug\b", q) and any(w in q for w in ["cyberjarvis", "project", "codebase", "issue", "error", "test", "repo", "repository"])):
        logger.info(f"Agent Task Profile: PROJECT_DEBUGGING for query: '{query}'")
        return AgentTaskProfile.PROJECT_DEBUGGING

    # 3. PROJECT_INSPECTION
    # Examples:
    # - "inspect my CyberJarvis project"
    # - "inspect my project"
    # - "review my CyberJarvis codebase"
    # - "show me the current project structure"
    # - "review my repository"
    # - "inspect files in this project", "show project structure", "explore codebase"
    # Note: If it has "architecture" or "design", it belongs to PROJECT_ARCHITECTURE below.
    has_arch_keyword = any(w in q for w in ["architecture", "design", "internals", "subsystem", "system design", "technical design"])

    if not has_arch_keyword:
        if any(p in q for p in [
            "inspect my cyberjarvis project", "inspect cyberjarvis project", "inspect my project",
            "inspect this project", "inspect the project", "inspect project", "inspect our project",
            "review my cyberjarvis codebase", "review cyberjarvis codebase", "review my codebase",
            "review the codebase", "review this codebase", "review our codebase",
            "show me the current project structure", "show current project structure", "show project structure",
            "current project structure", "project structure",
            "review my repository", "review this repository", "review the repository", "review our repository",
            "review repository", "inspect repository", "inspect my repository",
            "inspect files", "list files in project", "show project files", "explore codebase", "browse project files"
        ]):
            logger.info(f"Agent Task Profile: PROJECT_INSPECTION for query: '{query}'")
            return AgentTaskProfile.PROJECT_INSPECTION

    # 4. PROJECT_ARCHITECTURE (Preserved)
    # At minimum recognizes:
    # - analyze my CyberJarvis architecture
    # - analyse my CyberJarvis architecture
    # - analyze my CyberJarvis project
    # - inspect my CyberJarvis architecture
    # - analyze this project architecture
    # - explain my CyberJarvis architecture
    # - review my CyberJarvis design
    # and related variations
    architecture_actions = [
        "analyze", "analyse", "inspect", "explain", "review",
        "describe", "detail", "outline", "understand", "study"
    ]
    architecture_terms = [
        "architecture", "design", "internals", "subsystem",
        "system design", "technical design"
    ]
    project_indicators = [
        "cyberjarvis", "cyber jarvis", "this project", "my project",
        "the project", "our project", "our codebase", "this codebase",
        "my codebase", "the codebase", "this repository"
    ]

    has_action = any(re.search(rf"\b{re.escape(act)}\b", q) for act in architecture_actions)
    has_arch_term = any(term in q for term in architecture_terms)
    has_project_ref = any(proj in q for proj in project_indicators)

    # Patterns like "analyze my CyberJarvis architecture", "review my CyberJarvis design", "analyze my CyberJarvis project"
    if has_action and has_project_ref and (has_arch_term or "project" in q):
        logger.info(f"Agent Task Profile: PROJECT_ARCHITECTURE for query: '{query}'")
        return AgentTaskProfile.PROJECT_ARCHITECTURE

    # Direct "cyberjarvis architecture" or "project architecture" without explicit action verb
    if has_arch_term and has_project_ref:
        logger.info(f"Agent Task Profile: PROJECT_ARCHITECTURE for query: '{query}'")
        return AgentTaskProfile.PROJECT_ARCHITECTURE

    # Default to open-ended general planning
    logger.info(f"Agent Task Profile: GENERAL_PLANNING for query: '{query}'")
    return AgentTaskProfile.GENERAL_PLANNING
