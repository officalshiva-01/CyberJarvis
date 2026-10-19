"""
Runtime Parameter Validation Layer for CyberJarvis ToolRegistry.
Validates parameters against tool schemas before confirmation and execution.
Ensures zero leakage of secrets in validation error messages.
"""
from typing import Dict, Any, Tuple, Optional
from observability.logging import sanitize_data


def _check_basic_type(val: Any, expected_type: str) -> bool:
    """
    Validate val against declared basic schema type:
    string, integer, number, boolean, object/dict, array/list.
    Rejects booleans masquerading as integers or numbers.
    """
    t = expected_type.lower().strip()
    if t in ("string", "str"):
        return isinstance(val, str)
    elif t in ("integer", "int"):
        # Booleans are subclasses of int in Python, explicitly reject them
        return isinstance(val, int) and not isinstance(val, bool)
    elif t in ("number", "float"):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    elif t in ("boolean", "bool"):
        return isinstance(val, bool)
    elif t in ("object", "dict"):
        return isinstance(val, dict)
    elif t in ("array", "list"):
        return isinstance(val, (list, tuple))
    # If type is unspecified or unknown, permit
    return True


def validate_tool_parameters(
    tool_def: Any,
    parameters: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate runtime parameters against tool definition metadata.

    Requirements:
    - Validate parameters is a dictionary.
    - Reject unknown/unexpected parameters instead of silently ignoring them.
    - Enforce required parameters.
    - Apply defaults only when parameters are absent and a default exists.
    - Validate declared basic types.
    - Reject type mismatches before confirmation and execution.
    - Never leak secret values in error messages.

    Returns:
        (is_valid, validated_params_with_defaults, error_message)
    """
    tool_name = getattr(tool_def, "name", "unknown_tool")

    if parameters is None:
        raw_params: Dict[str, Any] = {}
    elif not isinstance(parameters, dict):
        err = f"Tool parameters for '{tool_name}' must be a dictionary, got {type(parameters).__name__}."
        return False, {}, str(sanitize_data(err))
    else:
        raw_params = parameters

    declared_params: Dict[str, Any] = getattr(tool_def, "parameters", {}) or {}

    # 1. Reject unknown/unexpected parameters
    for param_name in raw_params:
        if param_name not in declared_params:
            raw_err = f"Unexpected parameter '{param_name}' for tool '{tool_name}'."
            return False, {}, str(sanitize_data(raw_err))

    validated_params: Dict[str, Any] = {}

    # 2. Check each declared parameter
    for param_name, param_spec in declared_params.items():
        if hasattr(param_spec, "dict"):
            spec = param_spec.dict()
        elif hasattr(param_spec, "__dict__"):
            spec = vars(param_spec)
        elif isinstance(param_spec, dict):
            spec = param_spec
        else:
            spec = {}

        declared_type = str(spec.get("type", "")).strip()
        has_default = "default" in spec
        default_val = spec.get("default")

        if "required" in spec:
            is_required = bool(spec["required"])
        else:
            is_required = not has_default

        if param_name not in raw_params:
            if has_default:
                validated_params[param_name] = default_val
            elif is_required:
                raw_err = f"Missing required parameter '{param_name}' for tool '{tool_name}'."
                return False, {}, str(sanitize_data(raw_err))
        else:
            val = raw_params[param_name]
            # Type validation: reject type mismatch without leaking val
            if declared_type and not _check_basic_type(val, declared_type):
                actual_type = type(val).__name__
                raw_err = (
                    f"Parameter '{param_name}' for tool '{tool_name}' "
                    f"expected {declared_type}, got {actual_type}."
                )
                return False, {}, str(sanitize_data(raw_err))

            validated_params[param_name] = val

    return True, validated_params, None
