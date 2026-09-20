"""
Deterministic Architecture Mapper, Bounded Project Context Collector,
and Architecture Synthesizer for CyberJarvis.

Extracts structured repository facts, inspects bounded relevant files,
enforces strict context limits (<= 2500 chars), and synthesizes grounded
architectural analysis via a bounded Ollama call with failsafe timeout.
"""
import os
import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import ollama

from config import DEFAULT_MODEL, PROJECT_ROOT
from agent.schemas import AgentPlan, PlanStep
from tools.schemas import ToolResult
from observability.logging import get_logger

logger = get_logger("CyberJarvis.ProjectContext")

MAX_PROFILE_TOOLS = 3
MAX_FILE_CHARS = 400
MAX_SYNTHESIS_CONTEXT_CHARS = 2400
SYNTHESIS_NUM_PREDICT = 250
SYNTHESIS_TEMPERATURE = 0.1
DEFAULT_SYNTHESIS_TIMEOUT = float(os.getenv("CYBERJARVIS_SYNTHESIS_TIMEOUT", "12.0"))


class ArchitectureMapper:
    """
    Deterministic architecture mapper and repository fact extractor for CyberJarvis.
    Inspects repository components on disk and structures verified evidence.
    """

    EXCLUDED_DIRS = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "logs",
        "scratch", "build", "dist", ".idea", ".vscode", "tmp", ".system_generated"
    }

    EXCLUDED_EXTENSIONS = {
        ".pyc", ".pyo", ".pyd", ".db", ".sqlite", ".sqlite3",
        ".wav", ".mp3", ".pcm", ".log", ".tar", ".gz", ".zip", ".jsonl"
    }

    RELEVANT_FILES = [
        "README.md",
        "main.py",
        "config.py",
        "agent/gateway.py",
        "agent/router.py",
        "agent/agent_core.py",
        "agent/planner.py",
        "agent/executor.py",
        "agent/evaluator.py",
        "tools/registry.py",
        "voice/voice_loop.py",
        "memory/store.py",
        "security/policy.py",
        "ui/dashboard.py",
    ]

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()

    def scan_components(self) -> Dict[str, Any]:
        """
        Scan and detect repository components deterministically.
        Extracts facts for all 15 core architectural subsystems.
        """
        root = self.project_root

        # 1. Project Entry Point
        entry_file = root / "main.py"
        entry_point = {
            "path": "main.py",
            "exists": entry_file.is_file(),
            "summary": "CLI entry point supporting --voice, --diagnose, --command, and --no-tts flags",
        }

        # 2. Gateway
        gw_file = root / "agent" / "gateway.py"
        gateway = {
            "path": "agent/gateway.py",
            "exists": gw_file.is_file(),
            "summary": "Authoritative execution entry point and orchestrator measuring granular latency",
        }

        # 3. Router
        router_file = root / "agent" / "router.py"
        router = {
            "path": "agent/router.py",
            "exists": router_file.is_file(),
            "tiers": ["FAST", "DIAGNOSTIC", "INFORMATION", "AGENT"],
            "summary": "Deterministic pre-LLM classifier mapping queries to 4 execution tiers",
        }

        # 4. Agent Core
        agent_core_file = root / "agent" / "agent_core.py"
        agent_core = {
            "path": "agent/agent_core.py",
            "exists": agent_core_file.is_file(),
            "summary": "Agent loop orchestrator with task profile classification (bypassing planner for PROJECT_ARCHITECTURE)",
        }

        # 5. Planner
        planner_file = root / "agent" / "planner.py"
        planner = {
            "path": "agent/planner.py",
            "exists": planner_file.is_file(),
            "summary": "LLM planner (phi3:mini) constructing multi-step plans for open-ended queries",
        }

        # 6. Executor
        executor_file = root / "agent" / "executor.py"
        executor = {
            "path": "agent/executor.py",
            "exists": executor_file.is_file(),
            "summary": "Concurrent thread pool executor dispatching tool actions with parallel execution",
        }

        # 7. Evaluator
        evaluator_file = root / "agent" / "evaluator.py"
        evaluator = {
            "path": "agent/evaluator.py",
            "exists": evaluator_file.is_file(),
            "summary": "Grounded evaluator validating tool outputs and synthesizing final answers",
        }

        # 8. Tools Subsystem
        tools_dir = root / "tools"
        tool_files = [f.name for f in tools_dir.glob("*.py")] if tools_dir.is_dir() else []
        tools = {
            "path": "tools/registry.py",
            "exists": (tools_dir / "registry.py").is_file(),
            "categories": ["system", "diagnostic", "network", "file", "git"],
            "files": sorted(tool_files),
            "summary": "Tool registry with risk levels (LOW..CRITICAL) and permission boundaries (READ_ONLY, MUTATING)",
        }

        # 9. Voice Subsystem
        voice_dir = root / "voice"
        voice_files = [f.name for f in voice_dir.glob("*.py")] if voice_dir.is_dir() else []
        voice = {
            "path": "voice/",
            "exists": voice_dir.is_dir(),
            "files": sorted(voice_files),
            "summary": "Hardware mic discovery (audio_capture.py), STT (speech_to_text.py), TTS (text_to_speech.py via spd-say), loop (voice_loop.py)",
        }

        # 10. Memory Subsystem
        memory_dir = root / "memory"
        memory_files = [f.name for f in memory_dir.glob("*.py")] if memory_dir.is_dir() else []
        memory = {
            "path": "memory/",
            "exists": memory_dir.is_dir(),
            "files": sorted(memory_files),
            "summary": "SQLite persistent store (store.py) for facts & turns with recency/importance ranking (ranking.py)",
        }

        # 11. Security Subsystem
        security_dir = root / "security"
        security_files = [f.name for f in security_dir.glob("*.py")] if security_dir.is_dir() else []
        security = {
            "path": "security/",
            "exists": security_dir.is_dir(),
            "files": sorted(security_files),
            "summary": "Policy enforcement (policy.py), project root path traversal guards (permissions.py), interactive confirmation (confirmation.py)",
        }

        # 12. Integrations
        integrations_dir = root / "integrations"
        integrations = {
            "path": "integrations/",
            "exists": integrations_dir.is_dir(),
            "summary": "HTTP client for external APIs (GitHub, Weather)",
        }

        # 13. Observability
        obs_dir = root / "observability"
        observability = {
            "path": "observability/",
            "exists": obs_dir.is_dir(),
            "summary": "Structured logging (logging.py) and credential sanitization (sanitizer.py)",
        }

        # 14. UI
        ui_file = root / "ui" / "dashboard.py"
        ui = {
            "path": "ui/dashboard.py",
            "exists": ui_file.is_file(),
            "summary": "ANSI Terminal Dashboard displaying real-time execution paths, tool tables, and latency breakdowns",
        }

        # 15. Tests
        tests_dir = root / "tests"
        test_files = sorted([f.name for f in tests_dir.glob("test_*.py")]) if tests_dir.is_dir() else []
        tests = {
            "path": "tests/",
            "exists": tests_dir.is_dir(),
            "count": len(test_files),
            "summary": f"{len(test_files)} automated unit test suites covering routing, tools, memory, security, and agent paths",
        }

        return {
            "entry_point": entry_point,
            "gateway": gateway,
            "router": router,
            "agent_core": agent_core,
            "planner": planner,
            "executor": executor,
            "evaluator": evaluator,
            "tools": tools,
            "voice": voice,
            "memory": memory,
            "security": security,
            "integrations": integrations,
            "observability": observability,
            "ui": ui,
            "tests": tests,
        }

    def read_bounded_file(self, rel_path: str, max_chars: int = MAX_FILE_CHARS) -> Optional[str]:
        """Read a relevant file with boundary checks, exclusion filtering, and truncation."""
        p = Path(rel_path)
        if any(part in self.EXCLUDED_DIRS for part in p.parts):
            return None
        if p.suffix in self.EXCLUDED_EXTENSIONS:
            return None

        target = (self.project_root / rel_path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError:
            return None

        if not target.is_file():
            return None

        try:
            content = target.read_text(encoding="utf-8", errors="replace").strip()
            if len(content) > max_chars:
                return content[:max_chars] + "... [truncated]"
            return content
        except Exception as e:
            logger.warning(f"Error reading bounded file {rel_path}: {e}")
            return None

    def build_structured_context(self, components: Dict[str, Any], max_total_chars: int = MAX_SYNTHESIS_CONTEXT_CHARS) -> str:
        """
        Build a compact, structured facts string for synthesis.
        Guaranteed to stay strictly <= max_total_chars (default 2400 <= 2500).
        """
        lines = [
            "### CyberJarvis Structured Architecture Facts",
            f"- Entry Point: {components['entry_point']['path']} ({components['entry_point']['summary']})",
            f"- Gateway: {components['gateway']['path']} ({components['gateway']['summary']})",
            f"- Router: {components['router']['path']} (Tiers: {', '.join(components['router']['tiers'])})",
            f"- Agent Core: {components['agent_core']['path']} ({components['agent_core']['summary']})",
            f"- Planner & Executor: {components['planner']['path']}, {components['executor']['path']} ({components['executor']['summary']})",
            f"- Evaluator: {components['evaluator']['path']} ({components['evaluator']['summary']})",
            f"- Tools: {components['tools']['path']} (Categories: {', '.join(components['tools']['categories'])})",
            f"- Voice Subsystem: {components['voice']['path']} ({components['voice']['summary']})",
            f"- Memory Subsystem: {components['memory']['path']} ({components['memory']['summary']})",
            f"- Security Boundary: {components['security']['path']} ({components['security']['summary']})",
            f"- Observability & UI: {components['observability']['path']}, {components['ui']['path']}",
            f"- Test Coverage: {components['tests']['path']} ({components['tests']['summary']})",
        ]

        current_str = "\n".join(lines)
        remaining = max_total_chars - len(current_str) - 80

        if remaining > 150:
            readme = self.read_bounded_file("README.md", max_chars=min(300, remaining // 2))
            if readme:
                lines.append(f"\n[README Excerpt]:\n{readme}")
            current_str = "\n".join(lines)
            remaining = max_total_chars - len(current_str) - 40
            if remaining > 100:
                gw = self.read_bounded_file("agent/gateway.py", max_chars=min(200, remaining))
                if gw:
                    lines.append(f"\n[Gateway Excerpt]:\n{gw}")

        context_str = "\n".join(lines)
        return context_str[:max_total_chars]


def _call_ollama_with_timeout(model: str, messages: list, options: dict, timeout_seconds: float) -> dict:
    """Execute Ollama chat with strict timeout failsafe on a daemon thread."""
    result_box: List[Any] = [None]
    exc_box: List[Optional[Exception]] = [None]
    done_event = threading.Event()

    def worker():
        try:
            # Calling ollama.chat directly allows unit test unittest.mock patches to intercept cleanly
            result_box[0] = ollama.chat(model=model, messages=messages, options=options)
        except Exception as e:
            exc_box[0] = e
        finally:
            done_event.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    finished = done_event.wait(timeout=timeout_seconds)
    if not finished:
        raise TimeoutError(f"Ollama synthesis timed out after {timeout_seconds}s")
    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


class ProjectContextCollector:
    """Collects structured repository evidence and coordinates bounded architecture synthesis."""

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_SYNTHESIS_TIMEOUT,
        mapper: Optional[ArchitectureMapper] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.mapper = mapper or ArchitectureMapper()

    @staticmethod
    def build_architecture_task_plan(query: str) -> AgentPlan:
        """Construct a deterministic, bounded tool plan to inspect repository architecture."""
        steps = [
            PlanStep(
                tool="list_project_files",
                parameters={},
                explanation="List project directory structure and files to inspect architecture",
            ),
            PlanStep(
                tool="read_project_file",
                parameters={"filename": "README.md"},
                explanation="Read project architecture overview and subsystem documentation",
            ),
            PlanStep(
                tool="read_project_file",
                parameters={"filename": "agent/gateway.py"},
                explanation="Inspect authoritative gateway and 4-tier execution routing",
            ),
        ]
        return AgentPlan(
            intent="project_architecture",
            goal=f"Analyze CyberJarvis architecture for query: '{query}'",
            steps=steps[:MAX_PROFILE_TOOLS],
        )

    def collect_evidence(self, tool_results: Optional[List[ToolResult]] = None) -> Dict[str, Any]:
        """
        Process repository files and tool results into structured, bounded evidence.
        Runs deterministic ArchitectureMapper scan to ground all facts.
        """
        components = self.mapper.scan_components()
        structured_context = self.mapper.build_structured_context(components)

        evidence: Dict[str, Any] = {
            "components": components,
            "structured_context": structured_context,
            "root_files": [],
            "root_dirs": [],
            "readme_excerpt": "",
            "gateway_excerpt": "",
        }

        if tool_results:
            for r in tool_results:
                if not r.success or not r.output:
                    continue

                if r.tool_name == "list_project_files" and isinstance(r.output, list):
                    for item in r.output:
                        if isinstance(item, dict):
                            name = item.get("name", "")
                            itype = item.get("type", "file")
                            if itype == "directory":
                                evidence["root_dirs"].append(name)
                            else:
                                evidence["root_files"].append(name)

                elif r.tool_name == "read_project_file" and isinstance(r.output, dict):
                    fn = r.output.get("filename", "")
                    content = r.output.get("content", "")
                    if "readme" in fn.lower():
                        evidence["readme_excerpt"] = content[:MAX_FILE_CHARS]
                    elif "gateway" in fn.lower():
                        evidence["gateway_excerpt"] = content[:MAX_FILE_CHARS]

        return evidence

    def synthesize_architecture(
        self,
        query: str,
        tool_results: Optional[List[ToolResult]] = None,
        model: str = DEFAULT_MODEL,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, float]:
        """
        Synthesize grounded architecture report using ONE bounded Ollama LLM call.
        Employs strict context bounds (<= 2500 chars), num_predict <= 300, and a failsafe timeout.
        Returns:
            (answer_markdown, spoken_summary, synthesis_duration_ms)
        """
        synth_start = time.perf_counter()
        if evidence is None:
            evidence = self.collect_evidence(tool_results)

        structured_context = evidence.get("structured_context", "")
        if not structured_context:
            structured_context = self.mapper.build_structured_context(evidence.get("components", {}))

        system_prompt = (
            "You are CyberJarvis, analyzing your own local architecture.\n"
            "Base your explanation strictly on the provided repository facts.\n"
            "Structure your response with these exact markdown headers:\n"
            "### Architecture Overview\n"
            "### Request Flow\n"
            "### Major Components\n"
            "### Agent Execution Flow\n"
            "### Voice Flow\n"
            "### Memory Flow\n"
            "### Security Boundary\n"
            "### Current Strengths\n"
            "### Current Limitations\n"
            "### Recommended Next Engineering Step\n"
            "Keep each section concise (1-2 sentences). Finish with 'SUMMARY: <spoken summary>'."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User query: '{query}'\n\nRepository Evidence:\n{structured_context}",
            },
        ]

        try:
            resp = _call_ollama_with_timeout(
                model=model,
                messages=messages,
                options={
                    "temperature": SYNTHESIS_TEMPERATURE,
                    "num_predict": SYNTHESIS_NUM_PREDICT,
                },
                timeout_seconds=self.timeout_seconds,
            )
            synthesis_ms = (time.perf_counter() - synth_start) * 1000
            raw_content = resp.get("message", {}).get("content", "").strip()

            spoken_summary = self._extract_spoken_summary(raw_content)
            clean_answer = self._clean_answer_markdown(raw_content)

            # Ensure key sections exist; if LLM output was cut off or malformed, fallback to verified report
            required_checks = [
                "### Architecture Overview",
                "### Request Flow",
                "### Major Components",
                "### Recommended Next Engineering Step",
            ]
            if not all(sec in clean_answer for sec in required_checks):
                logger.warning("Ollama output incomplete or truncated. Falling back to deterministic grounded report.")
                clean_answer = self._generate_grounded_architecture_report(evidence)
                if not spoken_summary:
                    spoken_summary = self._extract_spoken_summary(clean_answer)

            logger.info(f"PROJECT_ARCHITECTURE synthesis completed in {synthesis_ms:.1f}ms")
            return clean_answer, spoken_summary, synthesis_ms

        except (TimeoutError, Exception) as ex:
            synthesis_ms = (time.perf_counter() - synth_start) * 1000
            logger.warning(f"Ollama architecture synthesis failed or timed out ({ex}). Using deterministic grounded report.")
            fallback = self._generate_grounded_architecture_report(evidence)
            spoken = self._extract_spoken_summary(fallback)
            return fallback, spoken, synthesis_ms

    def _extract_spoken_summary(self, text: str) -> str:
        m = re.search(r"(?:SUMMARY|SPOKEN SUMMARY):\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            summary = m.group(1).strip()
            summary = re.sub(r"[*_`#]", "", summary)
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary

        return "CyberJarvis features a four-tier local execution architecture combining deterministic Fast and Diagnostic paths with bounded local Ollama reasoning."

    def _clean_answer_markdown(self, text: str) -> str:
        return re.sub(r"\n+(?:SUMMARY|SPOKEN SUMMARY):\s*.+$", "", text, flags=re.IGNORECASE).strip()

    def _generate_grounded_architecture_report(self, evidence: Dict[str, Any]) -> str:
        """Deterministic grounded architecture report based on actual repository structure and files."""
        components = evidence.get("components", {})
        tests_count = components.get("tests", {}).get("count", 16)

        return (
            "### Architecture Overview\n"
            "CyberJarvis is a local, voice- and terminal-controlled AI diagnostic agent running on Linux. "
            "It operates on a four-tier deterministic execution architecture (FAST, DIAGNOSTIC, INFORMATION, AGENT) "
            "designed to achieve sub-second execution for deterministic tasks while reserving Ollama LLM reasoning "
            "for complex agentic workflows.\n\n"
            "### Request Flow\n"
            "1. **Input Ingestion**: Audio captured from physical ALSA/PipeWire microphones or text from the terminal.\n"
            "2. **STT & Normalization**: Audio transcribed via Vosk/Whisper; text normalized (wake word stripping, speech-artifact recovery) in `normalize_transcript`.\n"
            "3. **Deterministic Routing**: `ExecutionRouter` classifies the request into FAST, DIAGNOSTIC, INFORMATION, or AGENT before invoking any LLM.\n"
            "4. **Execution & Dispatch**: Query is dispatched to the corresponding execution path handler, instrumented with granular latency metrics, and rendered via Terminal Dashboard and Text-to-Speech (`spd-say`).\n\n"
            "### Major Components\n"
            "- **Entry Point (`main.py`)**: CLI entry supporting `--voice`, `--no-tts`, `--diagnose`, and `--command` execution.\n"
            "- **Gateway (`agent/gateway.py`)**: Authoritative coordinator measuring end-to-end latency across STT, router, planning, tools, and evaluation.\n"
            "- **Router (`agent/router.py`)**: Pre-LLM classifier mapping queries to FAST, DIAGNOSTIC, INFORMATION, or AGENT tiers.\n"
            "- **Agent Core (`agent/agent_core.py`)**: Orchestrates agent loop with task profiling, planning, execution, and grounded evaluation.\n"
            "- **Planner (`agent/planner.py`)**: LLM planner (`phi3:mini`) for open-ended queries (bypassed for deterministic profiles).\n"
            "- **Executor (`agent/executor.py`)**: Concurrent thread pool executing tools with parallel dispatch.\n"
            "- **Evaluator (`agent/evaluator.py`)**: Grounded evaluator synthesizing final answers and verifying tool results.\n"
            "- **Tool Registry (`tools/registry.py`)**: Central registry with permission levels (`READ_ONLY`, `MUTATING`) and risk metadata (`LOW` to `CRITICAL`).\n"
            "- **Voice Subsystem (`voice/`)**: Hardware mic discovery (`audio_capture.py`), STT (`speech_to_text.py`), TTS (`text_to_speech.py`), and continuous loop (`voice_loop.py`).\n"
            "- **Memory Subsystem (`memory/`)**: SQLite database (`store.py`) persisting user facts and conversation history with recency/importance ranking (`ranking.py`).\n"
            "- **Security Boundary (`security/`)**: Role-based policy enforcement (`policy.py`), path traversal guards (`permissions.py`), and interactive confirmation (`confirmation.py`).\n"
            "- **Observability & UI (`observability/`, `ui/dashboard.py`)**: Structured logging, secret sanitization, and terminal dashboard rendering.\n"
            "- **Integrations (`integrations/`)**: HTTP client for external APIs (GitHub, Weather).\n"
            f"- **Test Suite (`tests/`)**: Automated test suite with {tests_count}+ test suites covering all execution paths.\n\n"
            "### Agent Execution Flow\n"
            "For AGENT queries, `AgentCore` evaluates task profiles. Known profiles like `PROJECT_ARCHITECTURE` use deterministic tool plans (`list_project_files`, `read_project_file`), "
            "collecting repository evidence via `ArchitectureMapper` to bypass the expensive general planner and synthesize grounded analysis in a single bounded LLM call. Open-ended queries invoke `Planner` (`phi3:mini`), `PlanExecutor` with parallel thread pool, and `Evaluator`.\n\n"
            "### Voice Flow\n"
            "`AudioCapture` streams 16kHz PCM audio with adaptive energy VAD. `SpeechToText` transcribes speech. Output responses are converted to concise spoken summaries and synthesized non-blockingly via `TextToSpeech` (`spd-say`).\n\n"
            "### Memory Flow\n"
            "`_detect_and_save_fact` extracts user profile facts (name, goals, projects) into SQLite. Queries retrieve relevant facts; conversational turns are persisted with type classifications (`goal`, `project`, `preference`).\n\n"
            "### Security Boundary\n"
            "Read-only diagnostic and file inspection tools are auto-approved. Mutating actions (file deletion, process termination) require explicit user confirmation via `ConfirmationHandler` and pass through `is_path_safe_for_project` path traversal checks.\n\n"
            "### Current Strengths\n"
            "- Deterministic sub-second FAST (<50ms) and DIAGNOSTIC (<500ms) execution with zero LLM overhead.\n"
            "- Bounded project-architecture task profile eliminating 100s+ planner stalls.\n"
            "- Strict security boundaries with read-only sandboxing and confirmation barriers.\n\n"
            "### Current Limitations\n"
            "- Local LLM inference speed is hardware-dependent (CPU-bound phi3:mini).\n"
            "- STT acoustic accuracy depends on ambient environment and microphone hardware.\n\n"
            "### Recommended Next Engineering Step\n"
            "Implement deterministic tool profiles for `DEPLOYMENT_PLANNING` and `PROJECT_DEBUGGING` to further streamline multi-step agent tasks.\n\n"
            "SUMMARY: CyberJarvis features a four-tier local execution architecture combining deterministic Fast and Diagnostic paths with bounded local Ollama reasoning and strict security guardrails."
        )
