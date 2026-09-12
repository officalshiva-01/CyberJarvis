import json
from typing import List, Dict, Any, Optional
import ollama

from config import DEFAULT_MODEL
from agent.schemas import AgentPlan, ToolResult, AgentResponse
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Evaluator")


class Evaluator:
    """Analyzes tool execution results and formulates both detailed and spoken responses."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def evaluate(
        self,
        user_input: str,
        plan: AgentPlan,
        results: List[ToolResult],
        context_facts: Optional[List[str]] = None,
    ) -> AgentResponse:
        # If no tools were planned, perform conversational completion
        if not results:
            return self._handle_chat_response(user_input, context_facts)

        # Check for specific diagnostic output
        for res in results:
            if res.tool_name == "diagnose_system_bottleneck" and res.success and isinstance(res.output, dict):
                return self._synthesize_diagnostic_report(res.output, plan, results)

        # General tool results synthesis via LLM with fallback
        return self._synthesize_with_llm(user_input, plan, results, context_facts)

    def _synthesize_diagnostic_report(
        self, diag: Dict[str, Any], plan: AgentPlan, results: List[ToolResult]
    ) -> AgentResponse:
        verdict = diag.get("verdict", "HEALTHY")
        summary = diag.get("summary", "")
        metrics = diag.get("metrics", {})
        issues = diag.get("issues", [])
        recommendations = diag.get("recommendations", [])
        top_mem = diag.get("top_memory_consumers", [])
        top_cpu = diag.get("top_cpu_consumers", [])

        # Detailed terminal answer
        lines = [
            f"### System Diagnostic Report: **{verdict}**",
            f"- **CPU Usage**: {metrics.get('cpu_percent', 'N/A')}%",
            f"- **RAM Usage**: {metrics.get('ram_percent', 'N/A')}% ({metrics.get('ram_used_gb', 'N/A')} GB / {metrics.get('ram_total_gb', 'N/A')} GB)",
            f"- **Swap Usage**: {metrics.get('swap_percent', 'N/A')}%",
            f"- **Disk Usage**: {metrics.get('disk_percent', 'N/A')}% ({metrics.get('disk_free_gb', 'N/A')} GB free)",
        ]

        if top_mem:
            procs_str = ", ".join(f"{p['name']} ({p['memory_percent']}%)" for p in top_mem[:3])
            lines.append(f"- **Top Memory Consumers**: {procs_str}")

        if top_cpu:
            procs_str = ", ".join(f"{p['name']} ({p['cpu_percent']}%)" for p in top_cpu[:3])
            lines.append(f"- **Top CPU Consumers**: {procs_str}")

        if issues:
            lines.append("\n**Identified Issues:**")
            for iss in issues:
                lines.append(f"  - {iss}")

        if recommendations:
            lines.append("\n**Recommendations:**")
            for rec in recommendations:
                lines.append(f"  - {rec}")

        detailed_text = "\n".join(lines)

        # Spoken summary for TTS
        if verdict == "HEALTHY":
            spoken = (
                f"Your system is running smoothly. CPU usage is {metrics.get('cpu_percent', 0)}%, "
                f"RAM usage is {metrics.get('ram_percent', 0)}%, and {metrics.get('disk_free_gb', 0)} gigabytes of disk space are free."
            )
        else:
            top_culprit = top_mem[0]['name'] if top_mem else "an active process"
            spoken = (
                f"System slowdown detected. RAM is at {metrics.get('ram_percent', 0)}% "
                f"with {top_culprit} being a leading consumer. CPU is at {metrics.get('cpu_percent', 0)}%. "
                f"{recommendations[0] if recommendations else 'Please check active processes.'}"
            )

        return AgentResponse(
            answer=detailed_text,
            spoken_summary=spoken,
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _synthesize_with_llm(
        self,
        user_input: str,
        plan: AgentPlan,
        results: List[ToolResult],
        context_facts: Optional[List[str]] = None,
    ) -> AgentResponse:
        tool_data = []
        for r in results:
            tool_data.append({
                "tool": r.tool_name,
                "success": r.success,
                "output": r.output if r.success else r.error,
            })

        system_prompt = (
            "You are CyberJarvis, a local AI assistant. "
            "Using the provided tool execution results, directly answer the user's request. "
            "Be clear, precise, and never invent data not present in the results. "
            "Provide two parts in JSON format: 'answer' (markdown format for terminal) and 'spoken_summary' (1-2 clean sentences suitable for text-to-speech)."
        )

        user_content = f"User Request: {user_input}\n\nTool Results:\n{json.dumps(tool_data, default=str)}"

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": 0.2},
            )
            raw = resp.get("message", {}).get("content", "")

            # Attempt JSON extraction
            try:
                from agent.planner import extract_json_from_text
                parsed = extract_json_from_text(raw)
                if parsed and "answer" in parsed:
                    return AgentResponse(
                        answer=parsed["answer"],
                        spoken_summary=parsed.get("spoken_summary", parsed["answer"]),
                        plan=plan,
                        tool_results=results,
                        completed=True,
                    )
            except Exception:
                pass

            return AgentResponse(
                answer=raw,
                spoken_summary=raw[:200],
                plan=plan,
                tool_results=results,
                completed=True,
            )
        except Exception as ex:
            logger.warning(f"Ollama evaluation failed: {ex}. Using deterministic formatter.")

        # Deterministic formatting
        outputs = []
        for r in results:
            outputs.append(f"{r.tool_name}: {r.output if r.success else r.error}")
        text = "\n".join(outputs)
        return AgentResponse(
            answer=text,
            spoken_summary=text[:150],
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _handle_chat_response(
        self, user_input: str, context_facts: Optional[List[str]] = None
    ) -> AgentResponse:
        system_content = "You are CyberJarvis, a helpful and direct local AI assistant."
        if context_facts:
            system_content += "\nRelevant user facts:\n" + "\n".join(f"- {f}" for f in context_facts)

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                ],
            )
            reply = resp.get("message", {}).get("content", "")
            return AgentResponse(
                answer=reply,
                spoken_summary=reply[:200],
                completed=True,
            )
        except Exception as ex:
            msg = f"I am currently unable to contact Ollama: {ex}"
            return AgentResponse(answer=msg, spoken_summary=msg, completed=True)
