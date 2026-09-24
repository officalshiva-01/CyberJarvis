import time
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Dict, Any, Optional, List

from tools.schemas import ToolDefinition, ToolResult
from security.policy import RiskLevel, PermissionLevel, SecurityPolicy
from security.confirmation import ConfirmationHandler
from security.audit import audit_logger
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
        timeout: float = 10.0,
        requires_confirmation: Optional[bool] = None,
        reversible: bool = False,
    ):
        def decorator(func: Callable):
            if requires_confirmation is None:
                req_conf = (
                    (not is_read_only)
                    or (permission_level in (PermissionLevel.MUTATING, PermissionLevel.PRIVILEGED, PermissionLevel.ADMIN))
                    or (risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL))
                )
            else:
                req_conf = bool(requires_confirmation)

            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or {},
                permission_level=permission_level,
                is_read_only=is_read_only,
                risk_level=risk_level,
                timeout=timeout,
                requires_confirmation=req_conf,
                reversible=reversible,
            )
            self._tools[name] = RegisteredTool(definition=tool_def, func=func)
            logger.debug(f"Registered tool '{name}' (risk={risk_level.value}, read_only={is_read_only}, timeout={timeout}s, confirm={req_conf})")
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
                "timeout": tool.definition.timeout,
                "requires_confirmation": tool.definition.requires_confirmation,
                "reversible": tool.definition.reversible,
            })
        return declarations

    def execute(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        policy: Optional[SecurityPolicy] = None,
        confirmation_handler: Optional[ConfirmationHandler] = None,
        user_request: str = "",
    ) -> ToolResult:
        parameters = parameters or {}
        tool = self._tools.get(tool_name)
        if not tool:
            err = f"Unknown tool: '{tool_name}'"
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=parameters,
                permission_level="unknown",
                risk_level="high",
                confirmation_status="BLOCKED",
                result=None,
                duration_ms=0.0,
                success=False,
                error=err,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=err,
                duration_ms=0.0,
                verified=False,
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
            err = f"Security Policy Denied: {decision.reason}"
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=parameters,
                permission_level=tool.definition.permission_level.value,
                risk_level=tool.definition.risk_level.value,
                confirmation_status="BLOCKED",
                result=None,
                duration_ms=0.0,
                success=False,
                error=err,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=err,
                duration_ms=0.0,
                verified=False,
            )

        # 2. Confirmation Check
        confirmation_status = "NOT_REQUIRED"
        if decision.requires_confirmation or tool.definition.requires_confirmation:
            handler = confirmation_handler or ConfirmationHandler()
            confirmed = handler.confirm(tool_name, decision.reason, parameters)
            if not confirmed:
                err = f"Action '{tool_name}' rejected by user."
                audit_logger.log_event(
                    user_request=user_request,
                    tool=tool_name,
                    arguments=parameters,
                    permission_level=tool.definition.permission_level.value,
                    risk_level=tool.definition.risk_level.value,
                    confirmation_status="REJECTED",
                    result=None,
                    duration_ms=0.0,
                    success=False,
                    error=err,
                )
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=err,
                    duration_ms=0.0,
                    verified=False,
                )
            confirmation_status = "CONFIRMED"

        # 3. Execute deterministic function with bounded timeout
        timeout_sec = getattr(tool.definition, "timeout", 10.0) or 10.0
        start_time = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool, **parameters)
                output = future.result(timeout=timeout_sec)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # 4. Result Verification
            verified = True
            success = True
            if isinstance(output, dict):
                if "verified" in output:
                    verified = bool(output["verified"])
                if output.get("error") and not output.get("success"):
                    success = False
                elif "success" in output:
                    success = bool(output["success"]) and verified
                elif not verified:
                    success = False

            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=parameters,
                permission_level=tool.definition.permission_level.value,
                risk_level=tool.definition.risk_level.value,
                confirmation_status=confirmation_status,
                result=output,
                duration_ms=duration_ms,
                success=success,
                error=output.get("error") if isinstance(output, dict) else None,
            )

            return ToolResult(
                tool_name=tool_name,
                success=success,
                output=output,
                error=output.get("error") if (isinstance(output, dict) and not success) else None,
                duration_ms=round(duration_ms, 2),
                verified=verified,
            )
        except TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err = f"Tool '{tool_name}' timed out after {timeout_sec}s."
            logger.warning(err)
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=parameters,
                permission_level=tool.definition.permission_level.value,
                risk_level=tool.definition.risk_level.value,
                confirmation_status=confirmation_status,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=err,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=err,
                duration_ms=round(duration_ms, 2),
                verified=False,
            )
        except Exception as ex:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error executing tool '{tool_name}': {ex}", exc_info=True)
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=parameters,
                permission_level=tool.definition.permission_level.value,
                risk_level=tool.definition.risk_level.value,
                confirmation_status=confirmation_status,
                result=None,
                duration_ms=duration_ms,
                success=False,
                error=str(ex),
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(ex),
                duration_ms=round(duration_ms, 2),
                verified=False,
            )


# Global tool registry singleton
registry = ToolRegistry()
