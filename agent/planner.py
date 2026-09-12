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
    """Extract and parse JSON object from LLM response, handling markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
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
                pass
    return None


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

        system_instruction = f"""You are CyberJarvis, an intelligent local system assistant and diagnostic engineer.
Given a user's request, formulate a structured JSON plan to fulfill the goal using available tools.

AVAILABLE TOOLS:
{json.dumps(tool_declarations, indent=2)}

RESPONSE INSTRUCTIONS:
1. You MUST respond with ONLY a valid JSON object.
2. Structure:
{{
  "intent": "<string intent, e.g. system_diagnosis, check_memory, project_info, general_chat>",
  "goal": "<string describing what to achieve>",
  "steps": [
    {{"tool": "<registered_tool_name>", "parameters": {{}}, "explanation": "<why this tool is chosen>"}}
  ]
}}
3. If the user is asking to diagnose the computer, investigate why it's slow, or inspect health, use "diagnose_system_bottleneck" or a combination of "get_cpu_usage", "get_memory_usage", "get_top_processes", and "get_disk_usage".
4. If no tools are required (e.g. greetings, casual chat, general questions), return "steps": [].
5. Do NOT invent tool names. Only use tools listed under AVAILABLE TOOLS.
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
                options={"temperature": 0.1},
            )
            raw_content = response.get("message", {}).get("content", "")
            logger.debug(f"Planner LLM raw response: {raw_content}")

            parsed = extract_json_from_text(raw_content)
            if parsed and "steps" in parsed:
                valid_steps = []
                for s in parsed.get("steps", []):
                    tool_name = s.get("tool")
                    if tool_name and self.registry.get(tool_name):
                        valid_steps.append(
                            PlanStep(
                                tool=tool_name,
                                parameters=s.get("parameters", {}),
                                explanation=s.get("explanation"),
                            )
                        )
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
