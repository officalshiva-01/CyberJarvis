import os
import datetime as dt

from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel


@registry.register(
    name="get_time",
    description="Get the current local system time formatted as HH:MM:SS AM/PM.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_time() -> str:
    return dt.datetime.now().strftime("%I:%M:%S %p")


@registry.register(
    name="get_date",
    description="Get today's date formatted with weekday, month, day, and year.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_date() -> str:
    return dt.datetime.now().strftime("%A, %B %d, %Y")


@registry.register(
    name="get_current_directory",
    description="Get the current working directory path.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_current_directory() -> str:
    return os.getcwd()
