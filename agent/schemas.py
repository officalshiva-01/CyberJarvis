from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from tools.schemas import ToolParameter, ToolDefinition, ToolResult
from security.policy import RiskLevel, PermissionLevel


class PlanStep(BaseModel):
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None


class AgentPlan(BaseModel):
    intent: str
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str
    spoken_summary: Optional[str] = None
    plan: Optional[AgentPlan] = None
    tool_results: List[ToolResult] = Field(default_factory=list)
    completed: bool = True
    planning_ms: float = 0.0
    evaluator_ms: float = 0.0
    diagnostic_report: Optional[Any] = None
    project_profile: Optional[str] = None
    classification_ms: float = 0.0
    context_collection_ms: float = 0.0
    repo_scan_ms: float = 0.0
    evidence_extraction_ms: float = 0.0
    synthesis_ms: float = 0.0
    synthesis_status: Optional[str] = None
    fallback_used: Optional[bool] = None

