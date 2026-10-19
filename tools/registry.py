import time
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable, Dict, Any, Optional, List, Union

from tools.schemas import ToolDefinition, ToolResult
from tools.isolation import IsolationMode, execute_in_process, execute_in_thread
from security.policy import RiskLevel, PermissionLevel, SecurityPolicy
from security.confirmation import ConfirmationHandler
from security.audit import audit_logger
from security.tool_validation import validate_tool_parameters
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
        isolation: Union[IsolationMode, str] = IsolationMode.PROCESS,
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

            if isinstance(isolation, str):
                try:
                    iso_val = IsolationMode(isolation.lower().strip())
                except ValueError:
                    iso_val = isolation  # Keep invalid value so execute() can fail closed!
            elif isinstance(isolation, IsolationMode):
                iso_val = isolation
            else:
                iso_val = str(isolation)

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
                isolation=iso_val,
            )
            self._tools[name] = RegisteredTool(definition=tool_def, func=func)
            logger.debug(f"Registered tool '{name}' (risk={risk_level.value}, read_only={is_read_only}, timeout={timeout}s, confirm={req_conf}, isolation={iso_val})")
            return func
        return decorator

    @property
    def tools(self) -> Dict[str, RegisteredTool]:
        return self._tools

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
            iso_str = (
                tool.definition.isolation.value
                if isinstance(tool.definition.isolation, IsolationMode)
                else str(tool.definition.isolation)
            )
            declarations.append({
                "name": name,
                "description": tool.definition.description,
                "parameters": tool.definition.parameters,
                "read_only": tool.definition.is_read_only,
                "risk": tool.definition.risk_level.value,
                "timeout": tool.definition.timeout,
                "isolation": iso_str,
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
        if parameters is None:
            raw_parameters: Dict[str, Any] = {}
        elif not isinstance(parameters, dict):
            err = f"Tool parameters for '{tool_name}' must be a dictionary, got {type(parameters).__name__}."
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments={"invalid_type": type(parameters).__name__},
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
        else:
            raw_parameters = parameters

        tool = self._tools.get(tool_name)
        if not tool:
            err = f"Unknown tool: '{tool_name}'"
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=raw_parameters,
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

        # 0. Runtime parameter validation (happens BEFORE confirmation and BEFORE execution)
        is_valid, validated_params, val_err = validate_tool_parameters(tool.definition, raw_parameters)
        if not is_valid:
            audit_logger.log_event(
                user_request=user_request,
                tool=tool_name,
                arguments=raw_parameters,
                permission_level=tool.definition.permission_level.value,
                risk_level=tool.definition.risk_level.value,
                confirmation_status="BLOCKED",
                result=None,
                duration_ms=0.0,
                success=False,
                error=val_err,
            )
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=val_err,
                duration_ms=0.0,
                verified=False,
            )

        parameters = validated_params

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

        # 3. Check isolation mode and fail closed on unsupported modes
        isolation_val = getattr(tool.definition, "isolation", IsolationMode.PROCESS)
        iso_str = (
            isolation_val.value
            if isinstance(isolation_val, IsolationMode)
            else str(isolation_val).lower().strip()
        )
        if iso_str not in (IsolationMode.PROCESS.value, IsolationMode.THREAD.value):
            err = f"Unsupported isolation mode '{isolation_val}' for tool '{tool_name}'. Allowed modes are 'process' and 'thread'."
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

        # 4. Execute deterministic function with bounded timeout under isolated boundary
        timeout_sec = getattr(tool.definition, "timeout", 10.0) or 10.0
        try:
            if iso_str == IsolationMode.PROCESS.value:
                exec_ok, output, exec_err, duration_ms = execute_in_process(tool, parameters, timeout_sec)
            else:
                exec_ok, output, exec_err, duration_ms = execute_in_thread(tool, parameters, timeout_sec)
        except Exception as runner_ex:
            duration_ms = 0.0
            exec_ok = False
            output = None
            exec_err = f"Isolation runner failure for '{tool_name}': {runner_ex}"

        if not exec_ok:
            is_timeout = "timed out after" in (exec_err or "")
            if is_timeout:
                err = f"Tool '{tool_name}' timed out after {timeout_sec}s."
                logger.warning(err)
            else:
                err = exec_err or "Tool execution failed"
                logger.error(f"Error executing tool '{tool_name}': {err}")

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

        # 5. Result Verification
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


# Global tool registry singleton
registry = ToolRegistry()
