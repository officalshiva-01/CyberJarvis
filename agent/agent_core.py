import time
from typing import Optional, Callable
from agent.schemas import AgentResponse, AgentPlan, PlanStep
from agent.state import ConversationState, Turn
from agent.planner import Planner
from agent.executor import PlanExecutor
from agent.evaluator import Evaluator
from agent.task_profile import AgentTaskProfile, classify_agent_task_profile
from agent.project_context import ProjectContextCollector
from memory import init_db, save_memory, save_fact, get_facts
from security.policy import SecurityPolicy
from security.confirmation import ConfirmationHandler
from observability.logging import get_logger

logger = get_logger("CyberJarvis.AgentCore")


class AgentCore:
    """The central agent orchestrator coordinating the entire understand-plan-execute-observe-evaluate loop."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        executor: Optional[PlanExecutor] = None,
        evaluator: Optional[Evaluator] = None,
        project_collector: Optional[ProjectContextCollector] = None,
        state: Optional[ConversationState] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        init_db()
        self.state = state or ConversationState()
        self.planner = planner or Planner()
        self.evaluator = evaluator or Evaluator()
        self.project_collector = project_collector or ProjectContextCollector()
        self.on_status = on_status_change or (lambda s: None)

        def step_callback(tool_name: str):
            self.on_status(f"🔧 Running: {tool_name}")

        self.executor = executor or PlanExecutor(
            policy=SecurityPolicy(),
            confirmation_handler=ConfirmationHandler(),
            step_callback=step_callback,
        )

    def process(self, user_input: str) -> AgentResponse:
        user_input_clean = user_input.strip()
        if not user_input_clean:
            return AgentResponse(answer="Please provide an instruction.", spoken_summary="Please provide an instruction.")

        logger.info(f"Processing user input: '{user_input_clean}'")

        # 0. Arbitrary Shell Command Defense Guard
        from security.policy import is_arbitrary_shell_attempt
        from security.audit import audit_logger
        is_shell, shell_reason = is_arbitrary_shell_attempt(user_input_clean)
        if is_shell:
            logger.warning(f"Arbitrary shell command attempt blocked: '{user_input_clean}' ({shell_reason})")
            audit_logger.log_event(
                user_request=user_input_clean,
                tool="arbitrary_shell_execution",
                arguments={"raw_input": user_input_clean},
                permission_level="admin",
                risk_level="critical",
                confirmation_status="BLOCKED",
                result=None,
                duration_ms=0.0,
                success=False,
                error=shell_reason,
            )
            return AgentResponse(
                answer="🚫 **Security Policy Violation**: Arbitrary shell command execution is strictly forbidden.\nCyberJarvis executes only registered, permissioned tools.",
                spoken_summary="Security violation. Arbitrary shell commands are strictly forbidden.",
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
            )

        # 1. Check for personal facts to save
        self._detect_and_save_fact(user_input_clean)

        # 2. Agent Task Profile Classification
        classify_start = time.perf_counter()
        task_profile = classify_agent_task_profile(user_input_clean)
        classification_ms = (time.perf_counter() - classify_start) * 1000

        # Specialized Safe Actions: PROJECT_TESTING
        if task_profile == AgentTaskProfile.PROJECT_TESTING:
            self.on_status("🧪 Running project automated tests...")
            plan = AgentPlan(
                intent="project_testing",
                goal=user_input_clean,
                steps=[PlanStep(tool="run_tests", parameters={}, explanation="Execute pytest suite")],
            )
            tool_results = self.executor.execute_plan(plan)
            res = tool_results[0] if tool_results else None
            success = res.success if res else False
            output = res.output if (res and isinstance(res.output, dict)) else {}
            exit_code = output.get("exit_code", 0 if success else 1)
            stdout = output.get("stdout", "")
            target = output.get("target", "tests")

            if success and exit_code == 0:
                answer = f"✅ **Tests Passed**: All tests passed successfully.\n- Suite: `{target}`\n- Exit Code: `0`\n- Status: **VERIFIED**"
                if stdout:
                    answer += f"\n\n```\n{stdout}\n```"
                spoken_summary = "All tests in your project passed successfully."
            else:
                err_msg = output.get("error") or (res.error if res else "Test execution failed")
                answer = f"❌ **Tests Failed**: Automated tests failed with exit code `{exit_code}`.\n- Suite: `{target}`\n- Error: {err_msg}"
                if stdout:
                    answer += f"\n\n```\n{stdout}\n```"
                spoken_summary = "Automated tests encountered failures."

            self.on_status("✅ Complete")
            response = AgentResponse(
                answer=answer,
                spoken_summary=spoken_summary,
                plan=plan,
                tool_results=tool_results,
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
                project_profile=task_profile.value,
                classification_ms=round(classification_ms, 2),
            )
            turn = Turn(user_input=user_input_clean, plan=plan, tool_results=tool_results, assistant_response=response.answer)
            self.state.add_turn(turn)
            save_memory(user_input_clean, response.answer, memory_type="project")
            return response

        # Specialized Safe Actions: PROJECT_BACKUP
        if task_profile == AgentTaskProfile.PROJECT_BACKUP:
            self.on_status("📦 Creating project backup archive...")
            plan = AgentPlan(
                intent="project_backup",
                goal=user_input_clean,
                steps=[PlanStep(tool="create_project_backup", parameters={}, explanation="Create compressed archive in backups/")],
            )
            tool_results = self.executor.execute_plan(plan)
            res = tool_results[0] if tool_results else None
            success = res.success if res else False
            output = res.output if (res and isinstance(res.output, dict)) else {}

            if success and output.get("verified"):
                b_path = output.get("backup_path", "")
                b_size = output.get("size_mb", 0.0)
                b_count = output.get("files_archived", 0)
                answer = (
                    f"📦 **Project Backup Created & Verified**\n\n"
                    f"- **Location**: `{b_path}`\n"
                    f"- **Size**: **{b_size} MB**\n"
                    f"- **Files Archived**: {b_count}\n"
                    f"- **Integrity Verification**: `VERIFIED (size > 0, valid tar.gz)`"
                )
                spoken_summary = f"Project backup was successfully created and verified, size {b_size} megabytes."
            else:
                err_msg = res.error if res else "Backup creation cancelled or failed."
                if "rejected by user" in err_msg.lower():
                    answer = "⚠️ **Backup Cancelled**: Operation was not authorized by the user."
                    spoken_summary = "Project backup cancelled."
                else:
                    answer = f"❌ **Backup Failed**: {err_msg}"
                    spoken_summary = "Project backup creation failed."

            self.on_status("✅ Complete")
            response = AgentResponse(
                answer=answer,
                spoken_summary=spoken_summary,
                plan=plan,
                tool_results=tool_results,
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
                project_profile=task_profile.value,
                classification_ms=round(classification_ms, 2),
            )
            turn = Turn(user_input=user_input_clean, plan=plan, tool_results=tool_results, assistant_response=response.answer)
            self.state.add_turn(turn)
            save_memory(user_input_clean, response.answer, memory_type="project")
            return response

        # Deterministic Profiles: PROJECT_ARCHITECTURE, PROJECT_INSPECTION, PROJECT_DEBUGGING, DEPLOYMENT_PLANNING
        deterministic_statuses = {
            AgentTaskProfile.PROJECT_ARCHITECTURE: ("📁 Inspecting project architecture...", "🔍 Extracting architecture evidence...", "🧠 Synthesizing architecture analysis..."),
            AgentTaskProfile.PROJECT_INSPECTION: ("📁 Inspecting project structure...", "🔍 Extracting inspection evidence...", "🧠 Synthesizing project inspection..."),
            AgentTaskProfile.PROJECT_DEBUGGING: ("🐞 Debugging project issues...", "🔍 Extracting debugging evidence...", "🧠 Synthesizing debugging analysis..."),
            AgentTaskProfile.DEPLOYMENT_PLANNING: ("🚀 Planning deployment architecture...", "🔍 Extracting deployment evidence...", "🧠 Synthesizing deployment plan..."),
        }

        if task_profile in deterministic_statuses:
            scan_msg, evidence_msg, synth_msg = deterministic_statuses[task_profile]
            logger.info(f"Executing deterministic AGENT profile: {task_profile.value}")
            self.on_status(scan_msg)

            # 1. Repository Scan: plan & execute bounded inspection tools
            scan_start = time.perf_counter()
            plan = self.project_collector.build_task_plan(task_profile, user_input_clean)
            tool_results = self.executor.execute_plan(plan)
            repo_scan_ms = (time.perf_counter() - scan_start) * 1000

            # 2. Evidence Extraction: structured repository facts extraction
            self.on_status(evidence_msg)
            evidence_start = time.perf_counter()
            evidence = self.project_collector.collect_evidence_for_profile(task_profile, tool_results)
            evidence_ms = (time.perf_counter() - evidence_start) * 1000

            # 3. LLM Synthesis: bounded Ollama call with timeout failsafe
            self.on_status(synth_msg)
            synth_res = self.project_collector.synthesize_profile(
                profile=task_profile,
                query=user_input_clean,
                tool_results=tool_results,
                evidence=evidence,
            )
            clean_answer, spoken_summary, synthesis_ms = synth_res
            synthesis_status = getattr(synth_res, "status", "SUCCESS")
            fallback_used = getattr(synth_res, "fallback_used", False)
            self.on_status("✅ Complete")

            response = AgentResponse(
                answer=clean_answer,
                spoken_summary=spoken_summary,
                plan=plan,
                tool_results=tool_results,
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
                project_profile=task_profile.value,
                classification_ms=round(classification_ms, 2),
                context_collection_ms=round(evidence_ms, 2),
                repo_scan_ms=round(repo_scan_ms, 2),
                evidence_extraction_ms=round(evidence_ms, 2),
                synthesis_ms=round(synthesis_ms, 2),
                synthesis_status=synthesis_status,
                fallback_used=fallback_used,
            )

            # Persist to conversational state & long-term memory
            turn = Turn(
                user_input=user_input_clean,
                plan=plan,
                tool_results=tool_results,
                assistant_response=response.answer,
            )
            self.state.add_turn(turn)
            save_memory(user_input_clean, response.answer, memory_type="project")
            return response

        # 3. General Open-Ended Planning Phase
        self.on_status("🧠 Planning...")
        plan_start = time.perf_counter()
        plan = self.planner.plan(user_input_clean, state=self.state)
        planning_ms = (time.perf_counter() - plan_start) * 1000
        logger.info(f"Generated plan in {planning_ms:.1f}ms: intent='{plan.intent}', goal='{plan.goal}', steps={len(plan.steps)}")

        # 4. Execution phase
        tool_results = []
        if plan.steps:
            tool_results = self.executor.execute_plan(plan)

        # 5. Evaluation / Synthesis phase
        facts = get_facts(limit=10)
        if not tool_results:
            logger.info("No tools executed. Skipping heavy evaluator synthesis.")
            evaluator_ms = 0.0
            response = self.evaluator._handle_chat_response(user_input_clean, facts)
            response.plan = plan
            response.tool_results = []
            response.planning_ms = round(planning_ms, 2)
            response.evaluator_ms = 0.0
            response.project_profile = task_profile.value
            response.classification_ms = round(classification_ms, 2)
            self.on_status("✅ Complete")
        else:
            self.on_status("📊 Analyzing...")
            eval_start = time.perf_counter()
            response = self.evaluator.evaluate(
                user_input=user_input_clean,
                plan=plan,
                results=tool_results,
                context_facts=facts,
            )
            evaluator_ms = (time.perf_counter() - eval_start) * 1000
            response.planning_ms = round(planning_ms, 2)
            response.evaluator_ms = round(evaluator_ms, 2)
            response.project_profile = task_profile.value
            response.classification_ms = round(classification_ms, 2)
            self.on_status("✅ Diagnosis ready" if "diagnose" in plan.intent else "✅ Complete")

        # 6. Persist to conversational state & long-term memory
        turn = Turn(
            user_input=user_input_clean,
            plan=plan,
            tool_results=tool_results,
            assistant_response=response.answer,
        )
        self.state.add_turn(turn)

        mem_type = self._classify_memory_type(user_input_clean)
        save_memory(user_input_clean, response.answer, memory_type=mem_type)

        return response

    def _detect_and_save_fact(self, text: str):
        import re
        patterns = [
            r"^my name is (.+)$",
            r"^i am (.+)$",
            r"^i'm (.+)$",
            r"^my project is (.+)$",
            r"^my goal is (.+)$",
            r"^i want to become (.+)$",
            r"^i use (.+)$",
        ]
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                save_fact(text)
                logger.info(f"Persisted user fact: '{text}'")
                break

    def _classify_memory_type(self, text: str) -> str:
        lower = text.lower()
        if any(lower.startswith(p) for p in ["my goal is", "i want to become", "i plan to"]):
            return "goal"
        if any(lower.startswith(p) for p in ["my project is", "i am working on", "i'm building"]):
            return "project"
        if any(lower.startswith(p) for p in ["i prefer", "i like", "i don't like", "i dislike"]):
            return "preference"
        return "conversation"
