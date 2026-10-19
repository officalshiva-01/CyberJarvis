from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from security.policy import RiskLevel, PermissionLevel
from tools.isolation import IsolationMode


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
    timeout: float = 10.0
    requires_confirmation: bool = False
    reversible: bool = False
    isolation: Union[IsolationMode, str] = IsolationMode.PROCESS

    @property
    def read_only(self) -> bool:
        return self.is_read_only

    @property
    def permission(self) -> str:
        return self.permission_level.value.upper()

    @property
    def risk(self) -> str:
        return self.risk_level.value.upper()

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "permission": self.permission,
            "permission_level": self.permission_level.value,
            "risk": self.risk,
            "risk_level": self.risk_level.value,
            "timeout": self.timeout,
            "isolation": self.isolation.value if isinstance(self.isolation, IsolationMode) else str(self.isolation),
            "read_only": self.is_read_only,
            "requires_confirmation": self.requires_confirmation,
            "reversible": self.reversible,
        }


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    verified: bool = True
    verification_details: Optional[Dict[str, Any]] = None
