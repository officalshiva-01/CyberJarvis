import time
from typing import Optional, Callable
from agent.schemas import AgentResponse
from agent.state import ConversationState, Turn
from agent.planner import Planner
from agent.executor import PlanExecutor
from agent.evaluator import Evaluator
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
        state: Optional[ConversationState] = None,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        init_db()
        self.state = state or ConversationState()
        self.planner = planner or Planner()
        self.evaluator = evaluator or Evaluator()
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

        # 2. Planning phase
        self.on_status("🧠 Planning...")
        plan_start = time.perf_counter()
        plan = self.planner.plan(user_input_clean, state=self.state)
        planning_ms = (time.perf_counter() - plan_start) * 1000
        logger.info(f"Generated plan in {planning_ms:.1f}ms: intent='{plan.intent}', goal='{plan.goal}', steps={len(plan.steps)}")

        # 3. Execution phase
        tool_results = []
        if plan.steps:
            tool_results = self.executor.execute_plan(plan)

        # 4. Evaluation / Synthesis phase
        self.on_status("📊 Analyzing...")
        eval_start = time.perf_counter()
        facts = get_facts(limit=10)
        response = self.evaluator.evaluate(
            user_input=user_input_clean,
            plan=plan,
            results=tool_results,
            context_facts=facts,
        )
        evaluator_ms = (time.perf_counter() - eval_start) * 1000
        response.planning_ms = round(planning_ms, 2)
        response.evaluator_ms = round(evaluator_ms, 2)
        self.on_status("✅ Diagnosis ready" if "diagnose" in plan.intent else "✅ Complete")

        # 5. Persist to conversational state & long-term memory
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
