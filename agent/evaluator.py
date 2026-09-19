"""
Evidence-Grounded Evaluator for CyberJarvis.
Strictly adheres to:
Tool Results = Evidence
LLM = Interpreter

Enforces structured evidence validation, prevents hallucinated shell scripts/tutorials,
and produces grounded display answers and natural spoken summaries.
"""
import re
import json
from typing import List, Dict, Any, Optional
import ollama

from config import DEFAULT_MODEL
from agent.schemas import AgentPlan, ToolResult, AgentResponse
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Evaluator")


def is_code_requested(user_input: str) -> bool:
    """Check if the user explicitly requested code, scripts, or terminal commands."""
    keywords = ["code", "script", "bash", "python", "snippet", "write a command", "terminal command", "how to write", "command to"]
    lower = user_input.lower()
    return any(k in lower for k in keywords)


def contains_hallucinated_code(text: str) -> bool:
    """Check if the text inappropriately contains bash scripts, shell code, or tutorial commands."""
    forbidden_patterns = [
        r"^#!/bin/bash",
        r"^#!/usr/bin/env",
        r"cat /proc/meminfo",
        r"ps aux \|",
        r"```bash\s*\n\s*#!",
        r"```(?:bash|sh|shell|python)",
        r"free -m\s*&&",
        r"sudo apt",
        r"systemctl restart",
        r"\bimport psutil\b",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, text, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def contains_unrelated_devops_lecture(text: str, user_input: str = "") -> bool:
    """Check if the text contains unsolicited DevOps/SRE lectures when not requested."""
    if user_input and any(w in user_input.lower() for w in ["devops", "kubernetes", "prometheus", "sre", "ci/cd"]):
        return False
    devops_patterns = [
        r"\bkubernetes\b",
        r"\bprometheus\b",
        r"\bgrafana\b",
        r"\bci/cd\b",
        r"\bdevops practice[s]?\b",
        r"\bsre practice[s]?\b",
        r"\bhorizontal pod autoscaler\b",
        r"\bcontainer orchestration\b",
    ]
    for pat in devops_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False



class Evaluator:
    """Evidence-grounded evaluator synthesizing factual answers from tool observations."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def evaluate(
        self,
        user_input: str,
        plan: AgentPlan,
        results: List[ToolResult],
        context_facts: Optional[List[str]] = None,
    ) -> AgentResponse:
        # If no tools were executed, perform general conversational synthesis
        if not results:
            return self._handle_chat_response(user_input, context_facts)

        # 1. Specialized Grounded Handler: diagnose_system_bottleneck
        for res in results:
            if res.tool_name == "diagnose_system_bottleneck" and res.success and isinstance(res.output, dict):
                return self._synthesize_diagnostic_report(res.output, plan, results)

        # 2. Specialized Grounded Handler: process / resource queries (e.g. "what is consuming memory")
        user_lower = user_input.lower()
        if any(w in user_lower for w in ["consuming", "using", "taking", "top process", "offender", "most memory", "most cpu", "heavy"]):
            proc_res = next((r for r in results if r.tool_name == "get_top_processes" and r.success and isinstance(r.output, list)), None)
            if proc_res and proc_res.output:
                return self._synthesize_top_process_answer(proc_res.output, user_lower, plan, results)

        # 3. Specialized Grounded Handler: Network diagnostics
        net_tools = [r for r in results if r.tool_name in ("check_network_connectivity", "check_dns_resolution", "get_network_interfaces")]
        if net_tools and any(w in user_lower for w in ["internet", "network", "dns", "wifi", "ping", "online"]):
            return self._synthesize_network_report(results, plan)

        # 4. Specialized Grounded Handler: Git diagnostics
        git_tools = [r for r in results if r.tool_name in ("get_git_status", "get_git_recent_commits")]
        if git_tools and any(w in user_lower for w in ["git", "branch", "commit", "changes", "repository"]):
            return self._synthesize_git_report(results, plan)

        # 5. General tool synthesis via LLM with strict evidence guardrails
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

        status_emoji = "🟢" if verdict == "HEALTHY" else ("🟡" if verdict == "WARNING" else "🔴")

        lines = [
            f"### System Diagnostic Report: {status_emoji} **{verdict}**",
            f"- **CPU Usage**: {metrics.get('cpu_percent', 'N/A')}%",
            f"- **RAM Usage**: {metrics.get('ram_percent', 'N/A')}% ({metrics.get('ram_used_gb', 'N/A')} GB / {metrics.get('ram_total_gb', 'N/A')} GB)",
            f"- **Swap Usage**: {metrics.get('swap_percent', 'N/A')}%",
            f"- **Disk Usage**: {metrics.get('disk_percent', 'N/A')}% ({metrics.get('disk_free_gb', 'N/A')} GB free)",
        ]

        if top_mem:
            procs_str = ", ".join(f"`{p['name']}` ({p['memory_percent']}%)" for p in top_mem[:3])
            lines.append(f"- **Top Memory Processes**: {procs_str}")

        if top_cpu:
            procs_str = ", ".join(f"`{p['name']}` ({p['cpu_percent']}%)" for p in top_cpu[:3])
            lines.append(f"- **Top CPU Processes**: {procs_str}")

        if issues:
            lines.append("\n**Identified Issues:**")
            for iss in issues:
                lines.append(f"  - {iss}")

        if recommendations:
            lines.append("\n**Recommendations:**")
            for rec in recommendations:
                lines.append(f"  - {rec}")

        display_text = "\n".join(lines)

        # Spoken summary for TTS (Concise natural sentence)
        if verdict == "HEALTHY":
            spoken = (
                f"Your system is currently healthy. CPU usage is {metrics.get('cpu_percent', 0)}%, "
                f"RAM usage is {metrics.get('ram_percent', 0)}%, and {metrics.get('disk_free_gb', 0)} GB disk space is free."
            )
        else:
            top_culprit = top_mem[0]['name'] if top_mem else "an active process"
            spoken = (
                f"System bottleneck detected. RAM is at {metrics.get('ram_percent', 0)}% "
                f"with {top_culprit} being a leading consumer. {recommendations[0] if recommendations else 'Please check running applications.'}"
            )

        return AgentResponse(
            answer=display_text,
            spoken_summary=spoken,
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _synthesize_top_process_answer(
        self, processes: List[Dict[str, Any]], user_lower: str, plan: AgentPlan, results: List[ToolResult]
    ) -> AgentResponse:
        is_cpu = "cpu" in user_lower or "processor" in user_lower
        res_type = "CPU" if is_cpu else "memory"
        pct_key = "cpu_percent" if is_cpu else "memory_percent"

        top_proc = processes[0]
        name = top_proc.get("name", "unknown")
        pid = top_proc.get("pid", "N/A")
        pct = top_proc.get(pct_key, 0.0)

        lines = [f"Among the currently running processes, **{name}** (PID `{pid}`) is consuming the most {res_type} at **{pct}%**."]
        lines.append(f"\n**Top 3 {res_type.upper()} Consumers:**")
        for i, p in enumerate(processes[:3], start=1):
            lines.append(f"{i}. `{p.get('name')}` (PID {p.get('pid')}) — **{p.get(pct_key)}%**")

        display_text = "\n".join(lines)
        spoken = f"Among active processes, {name} is consuming the most {res_type} at {pct} percent."

        return AgentResponse(
            answer=display_text,
            spoken_summary=spoken,
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _synthesize_network_report(self, results: List[ToolResult], plan: AgentPlan) -> AgentResponse:
        conn_res = next((r.output for r in results if r.tool_name == "check_network_connectivity" and r.success), None)
        dns_res = next((r.output for r in results if r.tool_name == "check_dns_resolution" and r.success), None)

        is_connected = conn_res.get("connected", False) if isinstance(conn_res, dict) else False
        dns_success = dns_res.get("success", False) if isinstance(dns_res, dict) else False

        lines = ["### Network Diagnostic Assessment:"]
        if is_connected and dns_success:
            lines.append("🟢 **Status**: Fully Online")
            lines.append(f"- **Internet Connectivity**: Reachable ({conn_res.get('target', '')})")
            lines.append(f"- **DNS Resolution**: Working (Resolved {dns_res.get('domain', '')} to `{dns_res.get('resolved_ip', '')}`)")
            spoken = "Your network is healthy. Both internet connectivity and DNS resolution are working normally."
        elif is_connected and not dns_success:
            lines.append("🟡 **Status**: DNS Failure Detected")
            lines.append(f"- **Internet Connectivity**: Reachable ({conn_res.get('target', '')})")
            lines.append(f"- **DNS Resolution**: Failed ({dns_res.get('error', 'DNS lookup error')})")
            spoken = "Internet connectivity is active, but DNS resolution failed. Check your DNS nameserver configuration."
        else:
            lines.append("🔴 **Status**: Offline / Unreachable")
            lines.append(f"- **Internet Connectivity**: Disconnected ({conn_res.get('status', 'Unreachable') if isinstance(conn_res, dict) else 'Failed'})")
            spoken = "Your computer is currently offline. Internet connectivity could not be established."

        return AgentResponse(
            answer="\n".join(lines),
            spoken_summary=spoken,
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _synthesize_git_report(self, results: List[ToolResult], plan: AgentPlan) -> AgentResponse:
        status_res = next((r.output for r in results if r.tool_name == "get_git_status" and r.success), None)
        commits_res = next((r.output for r in results if r.tool_name == "get_git_recent_commits" and r.success), None)

        lines = ["### Git Repository Status:"]
        if isinstance(status_res, dict):
            branch = status_res.get("branch", "unknown")
            is_clean = status_res.get("is_clean", False)
            changes = status_res.get("changes", [])
            lines.append(f"- **Branch**: `{branch}`")
            lines.append(f"- **Working Tree**: {'Clean' if is_clean else f'{len(changes)} modified/untracked files'}")
            if changes:
                lines.append("\n**Changed Files:**")
                for c in changes[:8]:
                    lines.append(f"  - `{c}`")

        if isinstance(commits_res, list) and commits_res:
            lines.append("\n**Recent Commits:**")
            for c in commits_res[:3]:
                if isinstance(c, dict) and "hash" in c:
                    lines.append(f"- [`{c['hash']}`] {c['message']} ({c['date']})")

        display = "\n".join(lines)
        spoken = f"Git branch is {status_res.get('branch', 'main') if isinstance(status_res, dict) else 'main'}. Repository status checked."
        return AgentResponse(
            answer=display,
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
            raw_out = r.output if r.success else r.error
            out_str = str(raw_out)
            # Bound tool evidence to prevent large prompt bloating
            if len(out_str) > 400:
                truncated_out = out_str[:397] + "..."
            else:
                truncated_out = raw_out
            tool_data.append({
                "tool": r.tool_name,
                "success": r.success,
                "output": truncated_out,
            })

        code_requested = is_code_requested(user_input)

        system_prompt = (
            "You are CyberJarvis, an evidence-grounded local AI assistant.\n"
            "Using ONLY the provided tool execution results, directly answer the user's request.\n"
            "STRICT RULES:\n"
            "1. Tool results are the SOURCE OF TRUTH. Never invent, alter, or hallucinate metrics, numbers, or process names.\n"
            "2. NEVER output Bash or Python code, shell scripts, '#!/bin/bash', or terminal commands unless explicitly requested by the user.\n"
            "3. NEVER produce unrelated DevOps, Kubernetes, Prometheus, or infrastructure explanations.\n"
            "4. Answer directly, factually, and concisely in markdown format.\n"
            "5. Voice summary ('spoken_summary') MUST be concise: 1 to 2 short natural sentences (max 150 characters), no markdown or code.\n"
            "6. Provide JSON with two fields: 'answer' (markdown text) and 'spoken_summary' (concise string)."
        )

        user_content = f"User Request: {user_input}\n\nTool Results:\n{json.dumps(tool_data, default=str)}"

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 250,
                },
            )
            raw = resp.get("message", {}).get("content", "")

            # Guardrails: Check for unrequested code or unrelated DevOps lecture
            if (not code_requested and contains_hallucinated_code(raw)) or contains_unrelated_devops_lecture(raw, user_input):
                logger.warning("Evaluator detected unrequested code or DevOps lecture in LLM output! Rejecting and using deterministic fallback.")
                return self._deterministic_fallback_synthesis(plan, results)

            # Attempt JSON extraction
            from agent.planner import extract_json_from_text
            parsed = extract_json_from_text(raw)
            if parsed and "answer" in parsed:
                ans = parsed["answer"]
                if (not code_requested and contains_hallucinated_code(ans)) or contains_unrelated_devops_lecture(ans, user_input):
                    return self._deterministic_fallback_synthesis(plan, results)
                spoken = parsed.get("spoken_summary", ans[:150])
                spoken = re.sub(r"[*_`#]", "", spoken).strip()
                return AgentResponse(
                    answer=ans,
                    spoken_summary=spoken,
                    plan=plan,
                    tool_results=results,
                    completed=True,
                )

            # Fallback if raw text contains markdown fences or JSON keys
            clean_raw = re.sub(r"```(?:json)?\s*|```", "", raw).strip()
            ans_m = re.search(r"['\"]?answer['\"]?\s*:\s*['\"](.*?)(?:['\"],?\s*['\"]?spoken_summary|$)", clean_raw, re.DOTALL)
            spk_m = re.search(r"['\"]?spoken_summary['\"]?\s*:\s*['\"]([^'\"]+)", clean_raw)
            if ans_m:
                extracted_ans = ans_m.group(1).encode().decode('unicode_escape', errors='ignore').strip()
                extracted_spk = spk_m.group(1).strip() if spk_m else extracted_ans[:150]
                return AgentResponse(
                    answer=extracted_ans,
                    spoken_summary=re.sub(r"[*_`#]", "", extracted_spk).strip(),
                    plan=plan,
                    tool_results=results,
                    completed=True,
                )

            spoken_clean = re.sub(r"```(?:json)?\s*|```|[{}\"']", "", raw)[:150].strip()
            return AgentResponse(
                answer=raw,
                spoken_summary=spoken_clean,
                plan=plan,
                tool_results=results,
                completed=True,
            )
        except Exception as ex:
            logger.warning(f"Ollama evaluator failed ({ex}). Using deterministic formatter.")

        return self._deterministic_fallback_synthesis(plan, results)

    def _deterministic_fallback_synthesis(self, plan: AgentPlan, results: List[ToolResult]) -> AgentResponse:
        lines = []
        for r in results:
            if r.success:
                lines.append(f"**{r.tool_name}**: {r.output}")
            else:
                lines.append(f"❌ **{r.tool_name}**: {r.error}")

        text = "\n\n".join(lines)
        spoken = f"Completed operations for: {', '.join(r.tool_name for r in results)}."
        return AgentResponse(
            answer=text,
            spoken_summary=spoken,
            plan=plan,
            tool_results=results,
            completed=True,
        )

    def _handle_chat_response(
        self, user_input: str, context_facts: Optional[List[str]] = None
    ) -> AgentResponse:
        lower = user_input.lower().strip()
        clean_tokens = set(re.sub(r"[^\w\s]", "", lower).split())
        greeting_words = {"hello", "hi", "hey", "namaste", "greetings"}

        # Rule 1: Greetings must not trigger a long LLM response
        if lower in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "namaste"] or clean_tokens.intersection(greeting_words):
            reply = "Hello! I am CyberJarvis, ready for your commands."
            return AgentResponse(
                answer=reply,
                spoken_summary=reply,
                completed=True,
            )

        code_requested = is_code_requested(user_input)
        system_content = (
            "You are CyberJarvis, a concise and direct local AI assistant. "
            "Answer directly in 1 or 2 concise sentences. "
            "Never output code, shell scripts, or DevOps tutorials unless explicitly requested."
        )
        if context_facts:
            system_content += "\nRelevant user facts:\n" + "\n".join(f"- {f}" for f in context_facts)

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                ],
                options={"temperature": 0.1, "num_predict": 150},
            )
            reply = resp.get("message", {}).get("content", "")

            if (not code_requested and contains_hallucinated_code(reply)) or contains_unrelated_devops_lecture(reply, user_input):
                reply = "I am ready to assist with system diagnostics and voice commands."

            return AgentResponse(
                answer=reply,
                spoken_summary=reply[:150],
                completed=True,
            )
        except Exception as ex:
            msg = f"I am currently unable to contact Ollama: {ex}"
            return AgentResponse(answer=msg, spoken_summary=msg, completed=True)
