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

    # 1. PROJECT_ARCHITECTURE
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

    # Patterns like "analyze my CyberJarvis architecture", "review my CyberJarvis design"
    if has_action and has_project_ref and (has_arch_term or "project" in q):
        logger.info(f"Agent Task Profile: PROJECT_ARCHITECTURE for query: '{query}'")
        return AgentTaskProfile.PROJECT_ARCHITECTURE

    # Direct "cyberjarvis architecture" or "project architecture" without explicit action verb
    if has_arch_term and has_project_ref:
        logger.info(f"Agent Task Profile: PROJECT_ARCHITECTURE for query: '{query}'")
        return AgentTaskProfile.PROJECT_ARCHITECTURE

    # 2. PROJECT_INSPECTION (e.g. "inspect files in this project", "show project structure")
    if any(p in q for p in [
        "inspect files", "list files in project", "show project files",
        "explore codebase", "inspect repository", "browse project files"
    ]):
        logger.info(f"Agent Task Profile: PROJECT_INSPECTION for query: '{query}'")
        return AgentTaskProfile.PROJECT_INSPECTION

    # 3. PROJECT_DEBUGGING (e.g. "debug this repository problem", "diagnose a repository problem", "deadlock in diagnostic logs")
    if any(p in q for p in [
        "debug deadlock", "diagnostic log", "deadlock in", "fix deadlock",
        "debug repository problem", "diagnose a repository", "diagnose repository",
        "repository problem", "investigate error log"
    ]):
        logger.info(f"Agent Task Profile: PROJECT_DEBUGGING for query: '{query}'")
        return AgentTaskProfile.PROJECT_DEBUGGING

    # 4. DEPLOYMENT_PLANNING (e.g. "plan a deployment architecture", "plan deployment to AWS ECS")
    if any(p in q for p in [
        "plan a deployment", "plan deployment", "deploy to aws", "deploy to kubernetes",
        "ecs deployment", "deployment to"
    ]):
        logger.info(f"Agent Task Profile: DEPLOYMENT_PLANNING for query: '{query}'")
        return AgentTaskProfile.DEPLOYMENT_PLANNING

    # Default to open-ended general planning
    logger.info(f"Agent Task Profile: GENERAL_PLANNING for query: '{query}'")
    return AgentTaskProfile.GENERAL_PLANNING
