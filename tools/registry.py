import time
import inspect
from typing import Callable, Dict, Any, Optional, List

from tools.schemas import ToolDefinition, ToolResult
from security.policy import RiskLevel, PermissionLevel, SecurityPolicy
from security.confirmation import ConfirmationHandler
from observability.logging import get_logger

logger = get_logger("CyberJarvis.ToolRegistry")


class RegisteredTool:
    def __init__(self, definition: ToolDefinition, func: Callable):
        self.definition = definition
        self.func = func

    def __call__(self, **kwargs):
        sig = inspect.signature(self.func)
        filtered_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in kwargs:
                filtered_kwargs[param_name] = kwargs[param_name]
            elif param.default != inspect.Parameter.empty:
                filtered_kwargs[param_name] = param.default

        return self.func(**filtered_kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        permission_level: PermissionLevel = PermissionLevel.READ_ONLY,
        is_read_only: bool = True,
        risk_level: RiskLevel = RiskLevel.LOW,
    ):
        def decorator(func: Callable):
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or {},
                permission_level=permission_level,
                is_read_only=is_read_only,
                risk_level=risk_level,
            )
            self._tools[name] = RegisteredTool(definition=tool_def, func=func)
            logger.debug(f"Registered tool '{name}' (risk={risk_level.value}, read_only={is_read_only})")
            return func
        return decorator

    def get(self, name: str) -> Optional[RegisteredTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool_definitions(self) -> List[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def get_tool_declarations_for_llm(self) -> List[Dict[str, Any]]:
        """Return formatted declarations for LLM prompt context."""
        declarations = []
        for name, tool in self._tools.items():
            declarations.append({
                "name": name,
                "description": tool.definition.description,
                "parameters": tool.definition.parameters,
                "read_only": tool.definition.is_read_only,
                "risk": tool.definition.risk_level.value,
            })
        return declarations

    def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        policy: Optional[SecurityPolicy] = None,
        confirmation_handler: Optional[ConfirmationHandler] = None,
    ) -> ToolResult:
        parameters = parameters or {}
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: '{tool_name}'",
                duration_ms=0.0,
            )

        # 1. Security & Policy check
        sec_policy = policy or SecurityPolicy()
        decision = sec_policy.evaluate(
            tool_name=tool_name,
            permission_level=tool.definition.permission_level,
            risk_level=tool.definition.risk_level,
            is_read_only=tool.definition.is_read_only,
        )

        if not decision.allowed:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Security Policy Denied: {decision.reason}",
                duration_ms=0.0,
            )

        if decision.requires_confirmation:
            handler = confirmation_handler or ConfirmationHandler()
            confirmed = handler.confirm(tool_name, decision.reason, parameters)
            if not confirmed:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Action '{tool_name}' rejected by user.",
                    duration_ms=0.0,
                )

        # 2. Execute deterministic function
        start_time = time.perf_counter()
        try:
            output = tool(**parameters)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as ex:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error executing tool '{tool_name}': {ex}", exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(ex),
                duration_ms=round(duration_ms, 2),
            )


# Global tool registry singleton
registry = ToolRegistry()
