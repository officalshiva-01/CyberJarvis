from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from security.policy import RiskLevel, PermissionLevel


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.READ_ONLY
    is_read_only: bool = True
    risk_level: RiskLevel = RiskLevel.LOW


class PlanStep(BaseModel):
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None


class AgentPlan(BaseModel):
    intent: str
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class AgentResponse(BaseModel):
    answer: str
    spoken_summary: Optional[str] = None
    plan: Optional[AgentPlan] = None
    tool_results: List[ToolResult] = Field(default_factory=list)
    completed: bool = True
