import os
import platform
import time
import datetime as dt
import psutil

from tools.registry import registry
from tools.isolation import IsolationMode
from security.policy import RiskLevel, PermissionLevel


@registry.register(
    name="get_cpu_usage",
    description="Get the current CPU utilization percentage across all cores.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_cpu_usage() -> float:
    return float(psutil.cpu_percent(interval=0.5))


@registry.register(
    name="get_memory_usage",
    description="Get total, used, available RAM and utilization percentage.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_memory_usage() -> dict:
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024 ** 3), 2),
        "used_gb": round(mem.used / (1024 ** 3), 2),
        "available_gb": round(mem.available / (1024 ** 3), 2),
        "percent": round(mem.percent, 1),
    }


@registry.register(
    name="get_swap_usage",
    description="Get swap memory utilization details.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_swap_usage() -> dict:
    swap = psutil.swap_memory()
    return {
        "total_gb": round(swap.total / (1024 ** 3), 2),
        "used_gb": round(swap.used / (1024 ** 3), 2),
        "free_gb": round(swap.free / (1024 ** 3), 2),
        "percent": round(swap.percent, 1),
    }


@registry.register(
    name="get_disk_usage",
    description="Get disk space usage for the root filesystem.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_disk_usage(path: str = "/") -> dict:
    disk = psutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(disk.total / (1024 ** 3), 2),
        "used_gb": round(disk.used / (1024 ** 3), 2),
        "free_gb": round(disk.free / (1024 ** 3), 2),
        "percent": round(disk.percent, 1),
    }


@registry.register(
    name="get_load_average",
    description="Get system load averages over 1, 5, and 15 minutes.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_load_average() -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
        cpu_count = psutil.cpu_count() or 1
        return {
            "load_1m": round(load1, 2),
            "load_5m": round(load5, 2),
            "load_15m": round(load15, 2),
            "cpu_cores": cpu_count,
            "is_overloaded": load1 > (cpu_count * 1.5),
        }
    except (AttributeError, OSError):
        return {"error": "Load average not supported on this platform"}


@registry.register(
    name="get_uptime",
    description="Get system uptime and boot timestamp.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_uptime() -> dict:
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    boot_str = dt.datetime.fromtimestamp(boot_time).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "boot_time": boot_str,
        "uptime_hours": hours,
        "uptime_minutes": minutes,
        "human_readable": f"{hours} hours, {minutes} minutes",
    }


@registry.register(
    name="get_os_info",
    description="Get local operating system, architecture, kernel, and Python version.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_os_info() -> dict:
    return {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


@registry.register(
    name="get_system_status",
    description="Get combined system status including CPU, memory, swap, and disk metrics.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_system_status() -> dict:
    return {
        "cpu_percent": get_cpu_usage(),
        "memory": get_memory_usage(),
        "swap": get_swap_usage(),
        "disk": get_disk_usage(),
        "load": get_load_average(),
    }
