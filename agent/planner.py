import re
import json
from typing import Dict, Any, Optional, List
import ollama

from config import DEFAULT_MODEL
from agent.schemas import AgentPlan, PlanStep
from agent.state import ConversationState
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Planner")


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract and parse JSON object from LLM response, handling markdown fences and Python dict literals."""
    import ast
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block_match:
        cand = block_match.group(1).strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(cand)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                try:
                    val = ast.literal_eval(candidate)
                    if isinstance(val, dict):
                        return val
                except Exception:
                    pass
    return None


def is_tool_relevant_to_request(tool_name: str, query: str) -> bool:
    """
    Strict guardrail: programmatically prevents the LLM from selecting generic
    environment, network, or file tools for requests that do not explicitly require them.
    """
    q = query.lower()

    # 1. System environment & time tools
    if tool_name == "get_os_info":
        return any(w in q for w in ["os", "operating system", "distro", "distribution", "kernel", "system info", "linux version"])

    if tool_name == "get_time":
        return any(w in q for w in ["time", "clock", "what time", "current time"])

    if tool_name == "get_date":
        return any(w in q for w in ["date", "today", "calendar", "day is it"])

    if tool_name == "get_battery_status":
        return any(w in q for w in ["battery", "power", "charge", "plugged in"])

    # 2. File & Directory tools
    if tool_name == "get_current_directory":
        return any(w in q for w in ["current directory", "cwd", "working dir", "where am i", "current folder", "pwd"])

    if tool_name == "list_project_files":
        return any(w in q for w in ["list files", "show files", "project files", "directory contents", "tree", "structure", "inspect files", "workspace files"])

    if tool_name == "read_project_file":
        return any(w in q for w in ["read file", "view file", "inspect file", "show content", "cat ", "readme", ".py", ".md", ".json", ".txt", ".toml"])

    # 3. Network tools
    if tool_name in ("check_network_connectivity", "check_dns_resolution", "get_network_interfaces"):
        return any(w in q for w in ["network", "internet", "ping", "wifi", "connect", "dns", "ip address", "online", "offline", "adapter"])

    # 4. Diagnostics & Resource tools
    if tool_name in ("get_cpu_usage", "get_memory_usage", "get_disk_usage", "get_top_processes", "diagnose_system_bottleneck", "explain_bottleneck_causes", "get_system_health"):
        return any(w in q for w in [
            "cpu", "processor", "ram", "memory", "disk", "storage", "space", "process", "processes",
            "slow", "sluggish", "freeze", "lag", "hang", "bottleneck", "performance", "diagnos",
            "health", "load", "utilization", "usage", "hardware", "system status", "computer", "machine"
        ])

    # 5. Git tools
    if tool_name in ("get_git_status", "get_git_recent_commits", "git_status", "git_log", "git_recent_commits"):
        return any(w in q for w in ["git", "branch", "commit", "working tree", "repo", "repository", "diff", "staged", "untracked"])

    # 6. Test & Project inspection tools
    if tool_name in ("get_test_summary", "test_summary", "run_project_tests", "run_tests", "test_execution"):
        return any(w in q for w in ["test", "tests", "pytest", "unit test", "suite", "fail", "failing", "assert", "debug", "error"])

    # 7. Project log tools
    if tool_name == "read_project_log":
        return any(w in q for w in ["log", "logs", "error", "errors", "exception", "traceback", "debug", "issue", "problem"])

    # 8. Action & Mutation tools
    if tool_name == "create_project_backup":
        return any(w in q for w in ["backup", "archive", "tar", "snapshot"])

    if tool_name == "write_project_file":
        return any(w in q for w in ["write", "create file", "save file", "update file", "modify file"])

    if tool_name == "restart_project_service":
        return any(w in q for w in ["restart", "reload", "service"])

    if tool_name in ("open_terminal", "open_browser", "launch_application"):
        return any(w in q for w in ["open", "launch", "start", "run application", "browser", "terminal"])

    return True


MAX_TOOLS_PER_PLAN = 3


class Planner:
    """Uses Ollama local LLM to generate structured action plans."""

    def __init__(self, model: str = DEFAULT_MODEL, registry=None):
        self.model = model
        self._registry = registry

    @property
    def registry(self):
        if self._registry is None:
            from tools.registry import registry
            self._registry = registry
        return self._registry

    def build_prompt(self, user_input: str, state: Optional[ConversationState] = None) -> List[Dict[str, str]]:
        tool_declarations = self.registry.get_tool_declarations_for_llm()
        concise_tools = [f"- {d['name']}: {d['description']}" for d in tool_declarations]
        tools_str = "\n".join(concise_tools)

        system_instruction = f"""You are CyberJarvis, an intelligent local system assistant and diagnostic engineer.
Given a user's request, formulate a structured JSON plan to fulfill the goal using available tools.

AVAILABLE TOOLS:
{tools_str}

STRICT RULES:
1. Select ONLY tools strictly necessary to satisfy the user's explicit request.
2. NEVER select generic environment tools (get_os_info, get_time, get_current_directory, list_project_files, check_network_connectivity) unless the user explicitly requested that specific information.
3. If no tools are required, return "steps": [].
4. Maximum {MAX_TOOLS_PER_PLAN} tools per plan.
5. You MUST respond with ONLY a valid JSON object matching:
{{
  "intent": "<string intent>",
  "goal": "<string describing what to achieve>",
  "steps": [
    {{"tool": "<registered_tool_name>", "parameters": {{}}, "explanation": "<why this tool is chosen>"}}
  ]
}}
"""
        messages = [{"role": "system", "content": system_instruction}]

        if state:
            messages.extend(state.get_recent_dialogue())

        messages.append({"role": "user", "content": user_input})
        return messages

    def plan(self, user_input: str, state: Optional[ConversationState] = None) -> AgentPlan:
        user_lower = user_input.lower()
        if state and state.latest_diagnosis:
            if any(q in user_lower for q in ["what is consuming", "what is using", "who is taking", "which process", "offender"]):
                if "memory" in user_lower or "ram" in user_lower:
                    return AgentPlan(
                        intent="investigate_processes",
                        goal="Inspect memory-consuming processes following up on diagnosis",
                        steps=[PlanStep(tool="get_top_processes", parameters={"limit": 5, "sort_by": "memory"}, explanation="Follow-up on memory bottleneck")],
                    )
                elif "cpu" in user_lower or "processor" in user_lower:
                    return AgentPlan(
                        intent="investigate_processes",
                        goal="Inspect CPU-consuming processes following up on diagnosis",
                        steps=[PlanStep(tool="get_top_processes", parameters={"limit": 5, "sort_by": "cpu"}, explanation="Follow-up on CPU bottleneck")],
                    )

        messages = self.build_prompt(user_input, state)
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,
                    "num_predict": 250,
                },
            )
            raw_content = response.get("message", {}).get("content", "")
            logger.debug(f"Planner LLM raw response: {raw_content}")

            parsed = extract_json_from_text(raw_content)
            if parsed and "steps" in parsed:
                valid_steps = []
                for s in parsed.get("steps", []):
                    tool_name = s.get("tool")
                    if tool_name and self.registry.get(tool_name):
                        if is_tool_relevant_to_request(tool_name, user_input):
                            valid_steps.append(
                                PlanStep(
                                    tool=tool_name,
                                    parameters=s.get("parameters", {}),
                                    explanation=s.get("explanation"),
                                )
                            )
                        else:
                            logger.info(f"Planner filtered out irrelevant tool '{tool_name}' for query: '{user_input}'")

                # Cap tools to MAX_TOOLS_PER_PLAN
                valid_steps = valid_steps[:MAX_TOOLS_PER_PLAN]

                return AgentPlan(
                    intent=parsed.get("intent", "general_task"),
                    goal=parsed.get("goal", user_input),
                    steps=valid_steps,
                )
        except Exception as ex:
            logger.warning(f"Ollama planner invocation failed ({ex}). Engaging deterministic fallback planner.")

        return self._deterministic_fallback_plan(user_input)

    def _deterministic_fallback_plan(self, user_input: str) -> AgentPlan:
        lower = user_input.lower().strip()

        if any(w in lower for w in ["diagnose", "slow", "sluggish", "health", "bottleneck", "check system", "system status"]):
            return AgentPlan(
                intent="system_diagnosis",
                goal="Diagnose computer performance and potential slowdown causes",
                steps=[PlanStep(tool="diagnose_system_bottleneck", explanation="Comprehensive hardware and process bottleneck analysis")],
            )

        if any(w in lower for w in ["cpu", "processor"]):
            return AgentPlan(
                intent="check_cpu",
                goal="Inspect CPU utilization and top CPU processes",
                steps=[
                    PlanStep(tool="get_cpu_usage", explanation="Check current CPU utilization percentage"),
                    PlanStep(tool="get_top_processes", parameters={"sort_by": "cpu", "limit": 3}, explanation="Identify top CPU consumers"),
                ],
            )

        if any(w in lower for w in ["ram", "memory"]):
            return AgentPlan(
                intent="check_memory",
                goal="Inspect RAM utilization and top memory consumers",
                steps=[
                    PlanStep(tool="get_memory_usage", explanation="Check virtual memory statistics"),
                    PlanStep(tool="get_top_processes", parameters={"sort_by": "memory", "limit": 3}, explanation="Identify top RAM consumers"),
                ],
            )

        if any(w in lower for w in ["disk", "storage", "free space"]):
            return AgentPlan(
                intent="check_disk",
                goal="Inspect disk space and usage",
                steps=[PlanStep(tool="get_disk_usage", explanation="Check root filesystem storage capacity")],
            )

        if any(w in lower for w in ["internet", "network", "ping", "connection", "wifi"]):
            return AgentPlan(
                intent="check_network",
                goal="Check network connectivity and DNS resolution",
                steps=[
                    PlanStep(tool="check_network_connectivity", explanation="Verify outbound internet connection"),
                    PlanStep(tool="check_dns_resolution", parameters={"domain": "github.com"}, explanation="Verify DNS resolution"),
                ],
            )

        if "backup" in lower:
            return AgentPlan(
                intent="project_backup",
                goal="Create and verify project backup",
                steps=[PlanStep(tool="create_project_backup", parameters={}, explanation="Create project backup archive")],
            )

        if "test" in lower and not any(w in lower for w in ["why", "failing", "fail", "broken"]):
            return AgentPlan(
                intent="project_testing",
                goal="Run project automated tests",
                steps=[PlanStep(tool="run_tests", parameters={}, explanation="Run project pytest suite")],
            )

        if any(w in lower for w in ["list files", "show files", "project files"]):
            return AgentPlan(
                intent="project_files",
                goal="List files in the CyberJarvis project",
                steps=[PlanStep(tool="list_project_files", explanation="List project workspace items")],
            )

        if any(w in lower for w in ["git status", "git branch", "commits", "what changed"]):
            return AgentPlan(
                intent="git_inspection",
                goal="Inspect Git version control status and history",
                steps=[
                    PlanStep(tool="get_git_status", explanation="Check working tree changes"),
                    PlanStep(tool="get_git_recent_commits", parameters={"limit": 3}, explanation="Check recent commits"),
                ],
            )

        if any(w in lower for w in ["time", "what time"]):
            return AgentPlan(
                intent="get_time",
                goal="Get current time",
                steps=[PlanStep(tool="get_time", explanation="Check system clock")],
            )

        if any(w in lower for w in ["date", "today's date"]):
            return AgentPlan(
                intent="get_date",
                goal="Get current date",
                steps=[PlanStep(tool="get_date", explanation="Check system calendar")],
            )

        return AgentPlan(
            intent="general_chat",
            goal=user_input,
            steps=[],
        )
