import time
from typing import Optional, Callable
from agent.schemas import AgentResponse
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

        # 1. Check for personal facts to save
        self._detect_and_save_fact(user_input_clean)

        # 2. Agent Task Profile Classification
        classify_start = time.perf_counter()
        task_profile = classify_agent_task_profile(user_input_clean)
        classification_ms = (time.perf_counter() - classify_start) * 1000

        # Special Deterministic Profile: PROJECT_ARCHITECTURE
        if task_profile == AgentTaskProfile.PROJECT_ARCHITECTURE:
            logger.info(f"Executing deterministic AGENT profile: {task_profile.value}")
            self.on_status("📁 Inspecting project architecture...")
            context_start = time.perf_counter()
            plan = self.project_collector.build_architecture_task_plan(user_input_clean)
            context_collection_ms = (time.perf_counter() - context_start) * 1000

            # Execute deterministic inspection tools (list_project_files, read_project_file)
            tool_results = self.executor.execute_plan(plan)

            # Single bounded Ollama synthesis call (<=500 tokens)
            self.on_status("🧠 Synthesizing architecture analysis...")
            clean_answer, spoken_summary, synthesis_ms = self.project_collector.synthesize_architecture(
                query=user_input_clean,
                tool_results=tool_results,
            )
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
                context_collection_ms=round(context_collection_ms, 2),
                synthesis_ms=round(synthesis_ms, 2),
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
