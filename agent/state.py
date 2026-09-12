from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from agent.schemas import AgentPlan, ToolResult


class Turn(BaseModel):
    user_input: str
    plan: Optional[AgentPlan] = None
    tool_results: List[ToolResult] = Field(default_factory=list)
    assistant_response: str = ""


class ConversationState:
    """Maintains short-term conversational context and diagnostic history."""

    def __init__(self, max_turns: int = 5):
        self.turns: List[Turn] = []
        self.max_turns = max_turns
        self.latest_diagnosis: Optional[Dict[str, Any]] = None

    def add_turn(self, turn: Turn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Cache latest diagnostic result for conversational follow-ups
        for res in turn.tool_results:
            if res.tool_name in ("diagnose_system_bottleneck", "get_system_status", "get_top_processes"):
                if res.success and isinstance(res.output, dict):
                    self.latest_diagnosis = res.output

    def get_recent_dialogue(self) -> List[Dict[str, str]]:
        messages = []
        for t in self.turns:
            messages.append({"role": "user", "content": t.user_input})
            if t.assistant_response:
                messages.append({"role": "assistant", "content": t.assistant_response})
        return messages
