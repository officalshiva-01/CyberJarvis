from typing import Dict, Any, Optional
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


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
