import psutil
from typing import List, Dict, Any, Optional

from tools.registry import registry
from tools.system_tools import get_cpu_usage, get_memory_usage, get_swap_usage, get_disk_usage
from security.policy import RiskLevel, PermissionLevel


@registry.register(
    name="get_top_processes",
    description="Get top processes ordered by memory or CPU usage.",
    parameters={
        "limit": {"type": "integer", "description": "Number of processes to return (default 5)", "default": 5},
        "sort_by": {"type": "string", "description": "'memory' or 'cpu' (default 'memory')", "default": "memory"},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_top_processes(limit: int = 5, sort_by: str = "memory") -> List[Dict[str, Any]]:
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status']):
        try:
            info = proc.info
            processes.append({
                "pid": info['pid'],
                "name": info['name'] or "unknown",
                "user": info['username'] or "unknown",
                "cpu_percent": round(info['cpu_percent'] or 0.0, 1),
                "memory_percent": round(info['memory_percent'] or 0.0, 1),
                "status": info['status'] or "unknown",
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    sort_key = "cpu_percent" if sort_by.lower() == "cpu" else "memory_percent"
    processes.sort(key=lambda p: p[sort_key], reverse=True)
    return processes[:limit]


@registry.register(
    name="diagnose_system_bottleneck",
    description="Perform an end-to-end diagnosis of computer slowdowns, analyzing CPU, RAM, Swap, Disk, and offending processes.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def diagnose_system_bottleneck() -> Dict[str, Any]:
    cpu = get_cpu_usage()
    mem = get_memory_usage()
    swap = get_swap_usage()
    disk = get_disk_usage()
    top_mem_procs = get_top_processes(limit=5, sort_by="memory")
    top_cpu_procs = get_top_processes(limit=5, sort_by="cpu")

    issues = []
    recommendations = []

    # Threshold checks
    is_ram_critical = mem["percent"] >= 85.0
    is_ram_warning = 75.0 <= mem["percent"] < 85.0
    is_cpu_critical = cpu >= 85.0
    is_cpu_warning = 70.0 <= cpu < 85.0
    is_disk_critical = disk["percent"] >= 90.0
    is_disk_warning = 80.0 <= disk["percent"] < 90.0
    is_swap_heavy = swap["percent"] >= 60.0

    if is_ram_critical:
        top_name = top_mem_procs[0]['name'] if top_mem_procs else 'process'
        top_pct = top_mem_procs[0]['memory_percent'] if top_mem_procs else 0
        issues.append(f"Critical RAM utilization at {mem['percent']}%. Top consumer is '{top_name}' ({top_pct}%).")
        recommendations.append(f"Consider closing high-memory application: {top_name}.")
    elif is_ram_warning:
        issues.append(f"Elevated RAM usage at {mem['percent']}%.")

    if is_swap_heavy:
        issues.append(f"Heavy Swap thrashing: {swap['percent']}% swap used ({swap['used_gb']} GB). Indicates RAM shortage.")
        recommendations.append("Free physical RAM to reduce disk swap thrashing.")

    if is_cpu_critical:
        top_name = top_cpu_procs[0]['name'] if top_cpu_procs else 'process'
        top_pct = top_cpu_procs[0]['cpu_percent'] if top_cpu_procs else 0
        issues.append(f"High CPU utilization at {cpu}%. Top thread is '{top_name}' ({top_pct}%).")
        recommendations.append(f"Inspect or throttle heavy CPU process '{top_name}'.")
    elif is_cpu_warning:
        issues.append(f"Moderate CPU activity at {cpu}%.")

    if is_disk_critical:
        issues.append(f"Critically low disk space: {disk['percent']}% full ({disk['free_gb']} GB free).")
        recommendations.append("Clean cache, temporary files, or remove unused large archives.")
    elif is_disk_warning:
        issues.append(f"Disk space is getting low: {disk['percent']}% full.")

    if not issues:
        status_verdict = "HEALTHY"
        summary = "System resources are running within normal thresholds. No severe bottlenecks detected."
    else:
        status_verdict = "DEGRADED" if (is_ram_critical or is_cpu_critical or is_disk_critical) else "WARNING"
        summary = "System slowdown detected: " + "; ".join(issues)

    return {
        "verdict": status_verdict,
        "summary": summary,
        "metrics": {
            "cpu_percent": cpu,
            "ram_percent": mem["percent"],
            "ram_used_gb": mem["used_gb"],
            "ram_total_gb": mem["total_gb"],
            "swap_percent": swap["percent"],
            "disk_percent": disk["percent"],
            "disk_free_gb": disk["free_gb"],
        },
        "issues": issues,
        "recommendations": recommendations,
        "top_memory_consumers": top_mem_procs,
        "top_cpu_consumers": top_cpu_procs,
    }
