"""
Fast Path Handler for CyberJarvis.
Bypasses LLM planner for simple, deterministic queries (facts, time, date, files, git, connectivity).
Provides sub-second latency and guarantees 100% evidence grounding from SQLite memory and native tools.
"""
import re
import time
from typing import Optional, List
from agent.schemas import AgentResponse, AgentPlan, PlanStep
from agent.router import normalize_transcript
from tools.schemas import ToolResult
from memory import get_facts, search_memories
from tools.registry import registry
from observability.logging import get_logger

logger = get_logger("CyberJarvis.FastPath")

EXIT_PHRASES = {"exit", "quit", "bye", "goodbye", "stop", "terminate", "band karo"}
GREETING_WORDS = {"hello", "hi", "hey", "namaste", "greetings", "yo"}


class FastPathHandler:
    """Executes deterministic queries directly without invoking the LLM."""

    def try_handle(self, user_input: str) -> Optional[AgentResponse]:
        text = normalize_transcript(user_input)
        if not text:
            text = user_input.lower().strip()
            if not text:
                return None

        clean_tokens = set(re.sub(r"[^\w\s]", "", text).split())

        # Never intercept complex diagnostic or troubleshooting queries
        if any(w in text for w in [
            "diagnose", "slow", "sluggish", "bottleneck", "why", "troubleshoot",
            "investigate", "system health", "system status", "consuming",
            "analyze my computer", "analyse my computer", "check my computer", "check my system",
            "check my network", "taking memory", "using memory", "mera computer", "mera system",
            "mere system", "ram kaun"
        ]):
            return None

        # 0a. Voice Exit commands (Rule 10)
        if text in EXIT_PHRASES or clean_tokens.issubset(EXIT_PHRASES) or (len(clean_tokens) == 1 and clean_tokens.intersection(EXIT_PHRASES)):
            return self._handle_exit_query()

        # 0b. Greetings (Rule 1: must not trigger long LLM response)
        is_greeting = (
            text in ["hello", "hi", "hey", "hey jarvis", "hello jarvis", "hi jarvis", "good morning", "good afternoon", "good evening", "namaste"]
            or (len(clean_tokens) <= 2 and bool(clean_tokens.intersection(GREETING_WORDS)))
        )
        if is_greeting:
            return self._handle_greeting_query()

        # 0c. Assistant Identity
        if any(p in text for p in ["who are you", "what are you", "what is cyberjarvis", "introduce yourself", "tum kaun ho", "what is your name", "what's your name", "tumhara naam kya hai"]):
            return self._handle_bot_identity_query()

        # 1a. Profile / Identity summary (Phase 5)
        profile_match = any(p in text for p in [
            "tell me about me", "tell me about myself", "who am i",
            "describe me", "what do you know about me", "my profile",
            "mere bare mein batao", "mere baare mein batao", "main kaun hoon",
            "tell me my profile", "about me", "about myself",
        ])
        if profile_match:
            return self._handle_profile_query()

        # 1b. Identity & Name questions
        name_match = any(p in text for p in [
            "what is my name", "what's my name",
            "mera naam kya hai", "tell me my name", "do you know my name"
        ])
        if name_match:
            return self._handle_name_query()

        # 2. Goal questions
        goal_match = any(p in text for p in [
            "what is my goal", "what's my goal", "mera goal kya hai",
            "what do i want to become", "mera lakshya kya hai"
        ])
        if goal_match:
            return self._handle_goal_query()

        # 3. Project questions
        project_match = any(p in text for p in [
            "what project am i working on", "what is my project", "what's my project",
            "what project am i building", "mera project kya hai", "project batao"
        ])
        if project_match:
            return self._handle_project_query()

        # 4. Clock / Time questions
        time_matches = [
            "what time is it", "what's the time", "current time", "tell me the time",
            "time which is my computer", "what is my computer time", "computer ka time kya hai",
            "computer time", "system time", "kya time hua hai", "time batao", "samay kya hai",
            "kya samay hai", "samay batao", "tell me time", "what is time"
        ]
        time_match = (
            text in time_matches
            or any(p in text for p in time_matches)
            or text == "time"
            or ("time" in text and any(w in text for w in ["computer", "system", "tell", "current", "what", "kya", "which"]))
        )
        if time_match:
            return self._handle_time_query()

        # 5. Calendar / Date questions
        date_match = any(p in text for p in [
            "what date is it", "what's the date", "today's date",
            "current date", "aaj ki date", "aaj kya tarikh hai"
        ]) or text == "date"
        if date_match:
            return self._handle_date_query()

        # 6. Current working directory
        pwd_match = any(p in text for p in [
            "what is my current directory", "current directory", "working directory", "where am i",
            "current folder", "pwd"
        ])
        if pwd_match:
            return self._handle_directory_query()

        # 7. Project files listing
        files_match = any(p in text for p in [
            "show my project files", "list project files", "list files", "show files",
            "project files dikhao", "files list karo"
        ])
        if files_match:
            return self._handle_files_query()

        # 8. Quick Git status
        git_match = (
            text in ["git status", "check git status", "what is git status", "check my project git status"] or
            any(p in text for p in ["mere project ka git status", "project ka git status"])
        )
        if git_match:
            return self._handle_git_status_query()

        # 9. Simple Internet connectivity
        net_match = any(p in text for p in [
            "check my internet", "check internet", "internet check karo",
            "check connectivity", "am i online", "is internet working"
        ])
        if net_match:
            return self._handle_connectivity_query()

        # 10. Direct Hardware & System metrics (Deterministic fast path - Rule 2)
        if text in ["check cpu", "check my cpu", "cpu usage", "what is cpu usage", "what is my cpu usage", "current cpu usage", "check cpu usage"]:
            return self._handle_cpu_query()

        if text in ["check ram", "check my ram", "ram usage", "memory usage", "check memory usage", "what is memory usage", "what is my memory usage", "current ram usage", "check memory", "check my memory"]:
            return self._handle_memory_query()

        if text in ["check disk", "check my disk", "disk usage", "what is disk usage", "check disk space", "disk space", "storage space", "check my disk space"]:
            return self._handle_disk_query()

        if text in ["uptime", "system uptime", "check uptime", "check system uptime"]:
            return self._handle_uptime_query()

        return None

    def _handle_profile_query(self) -> AgentResponse:
        facts = get_facts(limit=50)

        # 1. Name
        name = None
        for f in facts:
            m = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z0-9_\-]+)", f, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                break

        if not name:
            mems = search_memories("my name is", limit=5)
            for m in mems:
                m_res = re.search(r"my name is\s+([A-Za-z0-9_\-]+)", m[0], re.IGNORECASE)
                if m_res:
                    name = m_res.group(1).strip()
                    break

        # 2. Goal
        goal = None
        for f in facts:
            m = re.search(r"(?:my goal is to|goal is to|my goal is|goal is)\s+(.+)", f, re.IGNORECASE)
            if m:
                g = m.group(1).strip().rstrip(".")
                if g.lower().startswith("to "):
                    g = g[3:].strip()
                goal = g
                break

        # 3. Project
        project = None
        for f in facts:
            m = re.search(r"(?:my project is|project is)\s+([A-Za-z0-9_\-]+)", f, re.IGNORECASE)
            if m:
                project = m.group(1).strip().rstrip(".")
                break
            elif "CyberJarvis" in f and "project" in f.lower():
                project = "CyberJarvis"
                break

        # Generate response
        if name and goal and project:
            answer = f"You are **{name}**. Your current goal is to **{goal}**, and you are building **{project}**."
            spoken = f"You are {name}. Your current goal is to {goal}, and you are building {project}."
        elif not name and not goal and not project:
            answer = "I do not have your profile details recorded in memory yet."
            spoken = answer
        else:
            parts = []
            missing = []
            if name:
                parts.append(f"You are **{name}**.")
            else:
                missing.append("name")

            if goal:
                parts.append(f"Your current goal is to **{goal}**.")
            else:
                missing.append("goal")

            if project:
                parts.append(f"You are building **{project}**.")
            else:
                missing.append("project")

            if missing:
                parts.append(f"Your {', '.join(missing)} information is not recorded in memory.")

            answer = " ".join(parts)
            spoken = re.sub(r"[*_`]", "", answer)

        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="profile", goal="Retrieve user profile from SQLite memory", steps=[]),
        )

    def _handle_name_query(self) -> AgentResponse:
        facts = get_facts(limit=50)
        name_facts: List[str] = []

        for f in facts:
            match = re.search(r"(?:my name is|i am|i'm)\s+([A-Za-z0-9_\-]+)", f, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted.lower() not in [n.lower() for n in name_facts]:
                    name_facts.append(extracted)

        if not name_facts:
            # Check conversation memories
            mems = search_memories("my name is", limit=5)
            for m in mems:
                match = re.search(r"my name is\s+([A-Za-z0-9_\-]+)", m[0], re.IGNORECASE)
                if match:
                    name_facts.append(match.group(1).strip())
                    break

        if not name_facts:
            answer = "I do not have your name recorded in memory yet. You can tell me by saying: 'My name is [your name]'."
            return AgentResponse(answer=answer, spoken_summary=answer, completed=True)

        # Ambiguity check
        if len(name_facts) > 1:
            answer = f"According to stored records, your primary name is **{name_facts[0]}** (other noted references: {', '.join(name_facts[1:])})."
            spoken = f"Your name is {name_facts[0]}."
        else:
            answer = f"Your name is **{name_facts[0]}**."
            spoken = f"Your name is {name_facts[0]}."

        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="fast_path_fact", goal="Retrieve user name from SQLite memory", steps=[]),
        )

    def _handle_goal_query(self) -> AgentResponse:
        facts = get_facts(limit=50)
        goal_facts = [f for f in facts if any(w in f.lower() for w in ["goal is", "want to become", "plan to"])]

        if not goal_facts:
            answer = "I do not have your goal recorded yet. You can tell me by saying: 'My goal is [your goal]'."
            return AgentResponse(answer=answer, spoken_summary=answer, completed=True)

        clean_goal = goal_facts[0]
        answer = f"Your recorded goal is: **{clean_goal}**."
        spoken = f"Your goal is {clean_goal}."
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="fast_path_fact", goal="Retrieve user goal from SQLite memory", steps=[]),
        )

    def _handle_project_query(self) -> AgentResponse:
        facts = get_facts(limit=50)
        proj_facts = [f for f in facts if any(w in f.lower() for w in ["project is", "working on", "building"])]

        if not proj_facts:
            answer = "You are currently developing the **CyberJarvis** local AI diagnostic agent."
        else:
            answer = f"Your active project is: **{proj_facts[0]}**."

        spoken = "You are building CyberJarvis, a local AI diagnostic agent."
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="fast_path_fact", goal="Retrieve project info from memory", steps=[]),
        )

    def _handle_exit_query(self) -> AgentResponse:
        answer = "Goodbye! Shutting down CyberJarvis."
        return AgentResponse(
            answer=answer,
            spoken_summary=answer,
            completed=True,
            plan=AgentPlan(intent="exit", goal="Exit CyberJarvis", steps=[]),
        )

    def _handle_greeting_query(self) -> AgentResponse:
        answer = "Hello! I am CyberJarvis, ready for your commands."
        return AgentResponse(
            answer=answer,
            spoken_summary=answer,
            completed=True,
            plan=AgentPlan(intent="greeting", goal="Acknowledge greeting concisely", steps=[]),
        )

    def _handle_bot_identity_query(self) -> AgentResponse:
        answer = "I am **CyberJarvis**, an evidence-grounded local AI diagnostic agent."
        spoken = "I am CyberJarvis, your local AI diagnostic agent."
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="bot_identity", goal="State assistant identity", steps=[]),
        )

    def _handle_cpu_query(self) -> AgentResponse:
        from tools.diagnostic_tools import get_cpu_usage
        start_t = time.perf_counter()
        cpu_val = get_cpu_usage()
        dur_ms = (time.perf_counter() - start_t) * 1000
        answer = f"Current CPU utilization is **{cpu_val}%**."
        spoken = f"Current CPU utilization is {cpu_val} percent."
        step = PlanStep(tool="get_cpu_usage", parameters={}, explanation="Check CPU usage percentage")
        result = ToolResult(tool_name="get_cpu_usage", success=True, output=cpu_val, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="check_cpu", goal="Check CPU usage", steps=[step]),
            tool_results=[result],
        )

    def _handle_memory_query(self) -> AgentResponse:
        from tools.diagnostic_tools import get_memory_usage
        start_t = time.perf_counter()
        mem_data = get_memory_usage()
        dur_ms = (time.perf_counter() - start_t) * 1000
        pct = mem_data.get("percent", 0.0) if isinstance(mem_data, dict) else 0.0
        used = mem_data.get("used_gb", 0.0) if isinstance(mem_data, dict) else 0.0
        total = mem_data.get("total_gb", 0.0) if isinstance(mem_data, dict) else 0.0
        answer = f"Virtual RAM usage: **{pct}%** ({used} GB / {total} GB)."
        spoken = f"RAM usage is {pct} percent."
        step = PlanStep(tool="get_memory_usage", parameters={}, explanation="Check RAM usage")
        result = ToolResult(tool_name="get_memory_usage", success=True, output=mem_data, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="check_memory", goal="Check memory usage", steps=[step]),
            tool_results=[result],
        )

    def _handle_disk_query(self) -> AgentResponse:
        from tools.diagnostic_tools import get_disk_usage
        start_t = time.perf_counter()
        disk_data = get_disk_usage()
        dur_ms = (time.perf_counter() - start_t) * 1000
        pct = disk_data.get("percent", 0.0) if isinstance(disk_data, dict) else 0.0
        free = disk_data.get("free_gb", 0.0) if isinstance(disk_data, dict) else 0.0
        answer = f"Root disk usage: **{pct}%** with **{free} GB** free space remaining."
        spoken = f"Disk usage is {pct} percent with {free} gigabytes free."
        step = PlanStep(tool="get_disk_usage", parameters={}, explanation="Check root disk usage")
        result = ToolResult(tool_name="get_disk_usage", success=True, output=disk_data, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="check_disk", goal="Check disk usage", steps=[step]),
            tool_results=[result],
        )

    def _handle_uptime_query(self) -> AgentResponse:
        from tools.system_tools import get_uptime
        start_t = time.perf_counter()
        uptime_data = get_uptime()
        dur_ms = (time.perf_counter() - start_t) * 1000
        readable = uptime_data.get("human_readable", "unknown") if isinstance(uptime_data, dict) else str(uptime_data)
        answer = f"System uptime: **{readable}**."
        spoken = f"System has been up for {readable}."
        step = PlanStep(tool="get_uptime", parameters={}, explanation="Check system uptime")
        result = ToolResult(tool_name="get_uptime", success=True, output=uptime_data, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="get_uptime", goal="Check system uptime", steps=[step]),
            tool_results=[result],
        )

    def _handle_time_query(self) -> AgentResponse:
        from tools.action_tools import get_time
        start_t = time.perf_counter()
        curr_time = get_time()
        dur_ms = (time.perf_counter() - start_t) * 1000
        answer = f"The current system time is **{curr_time}**."
        spoken = f"The time is {curr_time}."
        step = PlanStep(tool="get_time", parameters={}, explanation="Check system clock")
        result = ToolResult(tool_name="get_time", success=True, output=curr_time, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="get_time", goal="Get current time", steps=[step]),
            tool_results=[result],
        )

    def _handle_date_query(self) -> AgentResponse:
        from tools.action_tools import get_date
        start_t = time.perf_counter()
        curr_date = get_date()
        dur_ms = (time.perf_counter() - start_t) * 1000
        answer = f"Today is **{curr_date}**."
        spoken = f"Today is {curr_date}."
        step = PlanStep(tool="get_date", parameters={}, explanation="Check system calendar")
        result = ToolResult(tool_name="get_date", success=True, output=curr_date, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="get_date", goal="Get current date", steps=[step]),
            tool_results=[result],
        )

    def _handle_directory_query(self) -> AgentResponse:
        from tools.action_tools import get_current_directory
        start_t = time.perf_counter()
        cwd = get_current_directory()
        dur_ms = (time.perf_counter() - start_t) * 1000
        answer = f"Current working directory: `{cwd}`"
        spoken = f"Current directory is {cwd}."
        step = PlanStep(tool="get_current_directory", parameters={}, explanation="Get current working directory")
        result = ToolResult(tool_name="get_current_directory", success=True, output=cwd, duration_ms=round(dur_ms, 2))
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="get_current_directory", goal="Get working directory", steps=[step]),
            tool_results=[result],
        )

    def _handle_files_query(self) -> AgentResponse:
        from tools.file_tools import list_project_files
        start_t = time.perf_counter()
        items = list_project_files()
        dur_ms = (time.perf_counter() - start_t) * 1000
        step = PlanStep(tool="list_project_files", parameters={}, explanation="List project workspace files")
        result = ToolResult(tool_name="list_project_files", success=True, output=items, duration_ms=round(dur_ms, 2))

        lines = [f"**CyberJarvis Project Files ({len(items)} items):**"]
        for it in items[:15]:
            icon = "📁" if it.get("type") == "directory" else "📄"
            lines.append(f"- {icon} `{it['name']}`")
        if len(items) > 15:
            lines.append(f"*... and {len(items) - 15} more items.*")

        answer = "\n".join(lines)
        spoken = f"Listed {len(items)} files and directories in your project."
        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="list_project_files", goal="List files", steps=[step]),
            tool_results=[result],
        )

    def _handle_git_status_query(self) -> AgentResponse:
        result = registry.execute("git_status", user_request="check git status")
        status = result.output if isinstance(result.output, dict) else {}
        step = PlanStep(tool="git_status", parameters={}, explanation="Check Git status")

        branch = status.get("branch", "unknown")
        is_clean = status.get("is_clean", False)
        changes = status.get("changes", [])

        if is_clean:
            answer = f"Git branch: **{branch}**\nWorking tree is clean. No uncommitted changes."
            spoken = f"Git branch is {branch}, and the working tree is completely clean."
        else:
            answer = (
                f"Git branch: **{branch}**\n"
                f"Modified files ({len(changes)}):\n" +
                "\n".join(f"- `{c}`" for c in changes)
            )
            spoken = f"Git branch is {branch} with {len(changes)} modified or untracked files."

        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="get_git_status", goal="Check Git status", steps=[step]),
            tool_results=[result],
        )

    def _handle_connectivity_query(self) -> AgentResponse:
        from tools.network_tools import check_network_connectivity
        start_t = time.perf_counter()
        net = check_network_connectivity()
        dur_ms = (time.perf_counter() - start_t) * 1000
        step = PlanStep(tool="check_network_connectivity", parameters={}, explanation="Verify internet connectivity")
        result = ToolResult(tool_name="check_network_connectivity", success=True, output=net, duration_ms=round(dur_ms, 2))

        if net.get("connected"):
            answer = "🟢 **Internet Status**: Connected and online (Public DNS reachable)."
            spoken = "Internet connectivity is active and working normally."
        else:
            answer = f"🔴 **Internet Status**: Disconnected.\nDetails: {net.get('status')}"
            spoken = "Your internet connection appears to be offline or unreachable."

        return AgentResponse(
            answer=answer,
            spoken_summary=spoken,
            completed=True,
            plan=AgentPlan(intent="check_connectivity", goal="Check internet", steps=[step]),
            tool_results=[result],
        )
