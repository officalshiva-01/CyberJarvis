"""
Deterministic Diagnostic Analyzer for CyberJarvis (Tier 2).
Calculates health status, primary bottleneck, secondary issues, and recommendations
using strict deterministic thresholds directly from actual tool outputs.

Zero LLM calls, zero hallucinated code/commands, 100% grounded in real system telemetry.
"""
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from tools.schemas import ToolResult
from observability.logging import get_logger

logger = get_logger("CyberJarvis.DiagnosticAnalyzer")


class DiagnosticReport(BaseModel):
    """Structured report returned by the deterministic diagnostic analyzer."""
    status: str  # "healthy", "warning", "critical", "unknown"
    observations: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    primary_issue: str = "none"
    secondary_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    spoken_summary: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    top_memory_consumers: List[Dict[str, Any]] = Field(default_factory=list)
    top_cpu_consumers: List[Dict[str, Any]] = Field(default_factory=list)


class DiagnosticAnalyzer:
    """Dedicated deterministic diagnostic analyzer evaluating system telemetry."""

    # Thresholds
    CPU_WARNING = 70.0
    CPU_CRITICAL = 85.0

    RAM_WARNING = 75.0
    RAM_CRITICAL = 85.0

    SWAP_WARNING = 30.0
    SWAP_CRITICAL = 60.0

    DISK_WARNING = 80.0
    DISK_CRITICAL = 90.0

    LOAD_WARNING_FACTOR = 1.0
    LOAD_CRITICAL_FACTOR = 1.5

    def analyze(self, results: List[ToolResult], user_input: str = "") -> Tuple[DiagnosticReport, str]:
        """
        Analyze tool results deterministically and produce a structured DiagnosticReport
        and formatted Markdown answer.
        """
        user_lower = user_input.lower().strip()

        # Extract metric data from tool results
        cpu_val: Optional[float] = None
        mem_data: Dict[str, Any] = {}
        swap_data: Dict[str, Any] = {}
        disk_data: Dict[str, Any] = {}
        load_data: Dict[str, Any] = {}
        net_data: Dict[str, Any] = {}
        dns_data: Dict[str, Any] = {}
        top_mem_procs: List[Dict[str, Any]] = []
        top_cpu_procs: List[Dict[str, Any]] = []

        for r in results:
            if not r.success:
                continue
            out = r.output
            if r.tool_name == "get_cpu_usage" and isinstance(out, (int, float)):
                cpu_val = float(out)
            elif r.tool_name == "get_memory_usage" and isinstance(out, dict):
                mem_data = out
            elif r.tool_name == "get_swap_usage" and isinstance(out, dict):
                swap_data = out
            elif r.tool_name == "get_disk_usage" and isinstance(out, dict):
                disk_data = out
            elif r.tool_name == "get_load_average" and isinstance(out, dict):
                load_data = out
            elif r.tool_name == "check_network_connectivity" and isinstance(out, dict):
                net_data = out
            elif r.tool_name == "check_dns_resolution" and isinstance(out, dict):
                dns_data = out
            elif r.tool_name == "get_top_processes" and isinstance(out, list):
                # Distinguish between sort_by memory vs cpu
                if out and "memory_percent" in out[0] and "cpu_percent" in out[0]:
                    # Check if sorted primarily by cpu or memory
                    if len(out) >= 2 and out[0]["cpu_percent"] >= out[1]["cpu_percent"] and out[0]["memory_percent"] <= out[1]["memory_percent"]:
                        top_cpu_procs = out
                    elif not top_mem_procs:
                        top_mem_procs = out
                    else:
                        top_cpu_procs = out

        # Fallback if top processes weren't split
        if not top_cpu_procs and top_mem_procs:
            top_cpu_procs = sorted(top_mem_procs, key=lambda p: p.get("cpu_percent", 0.0), reverse=True)

        # 1. Specialized query: "what is consuming the most memory/cpu"
        if any(w in user_lower for w in ["consuming the most", "using the most", "taking the most", "most memory", "most cpu", "top process", "offender"]):
            return self._analyze_process_query(user_lower, top_mem_procs, top_cpu_procs, mem_data, cpu_val)

        # 2. Comprehensive diagnosis
        observations: List[str] = []
        findings: List[str] = []
        issues: List[Tuple[str, str, int]] = []  # (category, description, severity: 2=critical, 1=warning)
        recommendations: List[str] = []

        metrics_summary: Dict[str, Any] = {}

        # --- CPU Analysis ---
        if cpu_val is not None:
            metrics_summary["cpu_percent"] = cpu_val
            cpu_status = "normal"
            top_cpu_name = top_cpu_procs[0].get("name", "process") if top_cpu_procs else "unknown"
            top_cpu_pct = top_cpu_procs[0].get("cpu_percent", 0.0) if top_cpu_procs else 0.0

            if cpu_val >= self.CPU_CRITICAL:
                cpu_status = "critical"
                issues.append((
                    "CPU",
                    f"High CPU utilization at {cpu_val}%. Top active thread is '{top_cpu_name}' ({top_cpu_pct}%).",
                    2
                ))
                recommendations.append(f"Inspect or throttle heavy CPU consumer: '{top_cpu_name}'.")
            elif cpu_val >= self.CPU_WARNING:
                cpu_status = "warning"
                issues.append((
                    "CPU",
                    f"Elevated CPU activity at {cpu_val}%.",
                    1
                ))
                recommendations.append("Monitor high-utilization processes if performance degrades further.")
            else:
                findings.append(f"CPU utilization is normal ({cpu_val}%).")

            observations.append(f"CPU: {cpu_val}% ({cpu_status})")

        # --- RAM Analysis ---
        if mem_data:
            ram_pct = mem_data.get("percent", 0.0)
            ram_used = mem_data.get("used_gb", 0.0)
            ram_total = mem_data.get("total_gb", 0.0)
            metrics_summary["ram_percent"] = ram_pct
            metrics_summary["ram_used_gb"] = ram_used
            metrics_summary["ram_total_gb"] = ram_total

            ram_status = "normal"
            top_mem_name = top_mem_procs[0].get("name", "process") if top_mem_procs else "unknown"
            top_mem_pct = top_mem_procs[0].get("memory_percent", 0.0) if top_mem_procs else 0.0

            if ram_pct >= self.RAM_CRITICAL:
                ram_status = "critical"
                issues.append((
                    "RAM",
                    f"Critical RAM utilization at {ram_pct}% ({ram_used} GB / {ram_total} GB). Top consumer is '{top_mem_name}' ({top_mem_pct}%).",
                    2
                ))
                recommendations.append(f"Consider closing high-memory application: '{top_mem_name}'.")
            elif ram_pct >= self.RAM_WARNING:
                ram_status = "warning"
                issues.append((
                    "RAM",
                    f"Elevated RAM usage at {ram_pct}% ({ram_used} GB / {ram_total} GB).",
                    1
                ))
                recommendations.append("Close background browser tabs or idle applications to free RAM.")
            else:
                findings.append(f"RAM usage is normal ({ram_pct}%).")

            observations.append(f"RAM: {ram_pct}% ({ram_used} GB / {ram_total} GB) ({ram_status})")

        # --- SWAP Analysis ---
        if swap_data:
            swap_pct = swap_data.get("percent", 0.0)
            swap_used = swap_data.get("used_gb", 0.0)
            metrics_summary["swap_percent"] = swap_pct
            metrics_summary["swap_used_gb"] = swap_used

            swap_status = "normal"
            if swap_pct >= self.SWAP_CRITICAL:
                swap_status = "critical"
                issues.append((
                    "SWAP",
                    f"Heavy swap thrashing at {swap_pct}% ({swap_used} GB used). Physical RAM exhaustion detected.",
                    2
                ))
                recommendations.append("Free physical RAM immediately to prevent disk I/O thrashing.")
            elif swap_pct >= self.SWAP_WARNING:
                swap_status = "warning"
                issues.append((
                    "SWAP",
                    f"Moderate swap pressure at {swap_pct}% ({swap_used} GB used).",
                    1
                ))
            else:
                findings.append(f"Swap memory is normal ({swap_pct}%).")

            observations.append(f"Swap: {swap_pct}% ({swap_used} GB used) ({swap_status})")

        # --- DISK Analysis ---
        if disk_data:
            disk_pct = disk_data.get("percent", 0.0)
            disk_free = disk_data.get("free_gb", 0.0)
            metrics_summary["disk_percent"] = disk_pct
            metrics_summary["disk_free_gb"] = disk_free

            disk_status = "normal"
            if disk_pct >= self.DISK_CRITICAL or disk_free < 5.0:
                disk_status = "critical"
                issues.append((
                    "DISK",
                    f"Critically low root disk space at {disk_pct}% ({disk_free} GB free).",
                    2
                ))
                recommendations.append("Clean package caches, system logs, or remove unused large archives.")
            elif disk_pct >= self.DISK_WARNING:
                disk_status = "warning"
                issues.append((
                    "DISK",
                    f"Disk storage is getting tight: {disk_pct}% full ({disk_free} GB free).",
                    1
                ))
                recommendations.append("Review disk usage and clear temporary files.")
            else:
                findings.append(f"Disk storage is normal ({disk_pct}% used, {disk_free} GB free).")

            observations.append(f"Disk: {disk_pct}% ({disk_free} GB free) ({disk_status})")

        # --- LOAD Analysis ---
        if load_data and "load_1m" in load_data:
            l1 = load_data.get("load_1m", 0.0)
            l5 = load_data.get("load_5m", 0.0)
            l15 = load_data.get("load_15m", 0.0)
            cores = load_data.get("cpu_cores", 1)
            metrics_summary["load_1m"] = l1
            metrics_summary["cpu_cores"] = cores

            load_status = "normal"
            if l1 > (cores * self.LOAD_CRITICAL_FACTOR):
                load_status = "critical"
                issues.append((
                    "LOAD",
                    f"System capacity overloaded: 1-minute load is {l1} on {cores} CPU cores (threshold {cores * self.LOAD_CRITICAL_FACTOR:.1f}).",
                    2
                ))
                recommendations.append("Wait for high-compute background jobs to finish or reduce task concurrency.")
            elif l1 > (cores * self.LOAD_WARNING_FACTOR):
                load_status = "warning"
                issues.append((
                    "LOAD",
                    f"Elevated system load: {l1} on {cores} CPU cores.",
                    1
                ))
            else:
                findings.append(f"System load is balanced ({l1} on {cores} cores).")

            observations.append(f"Load: {l1}, {l5}, {l15} ({cores} cores) ({load_status})")

        # --- NETWORK Analysis ---
        if net_data:
            is_connected = net_data.get("connected", False)
            metrics_summary["network_connected"] = is_connected
            if not is_connected:
                issues.append((
                    "NETWORK",
                    f"Network connectivity failure: {net_data.get('status', 'Unreachable')}.",
                    2
                ))
                recommendations.append("Check physical network cable, Wi-Fi connectivity, or default gateway.")
                observations.append("Network: Disconnected / Unreachable")
            else:
                observations.append(f"Network: Connected ({net_data.get('target', '1.1.1.1:53')})")
                findings.append("Internet connectivity is active.")

        if dns_data:
            dns_ok = dns_data.get("success", False)
            metrics_summary["dns_success"] = dns_ok
            if not dns_ok and net_data.get("connected", False):
                issues.append((
                    "DNS",
                    f"DNS resolution failure: {dns_data.get('error', 'Unable to resolve domain')}.",
                    1
                ))
                recommendations.append("Verify system DNS nameserver configuration in /etc/resolv.conf.")
                observations.append("DNS: Failed")
            elif dns_ok:
                observations.append(f"DNS: Working (Resolved {dns_data.get('domain', '')})")

        # Add top process observations
        if top_mem_procs:
            top_names = [f"`{p['name']}` ({p.get('memory_percent', 0)}%)" for p in top_mem_procs[:3]]
            observations.append(f"Top RAM Consumers: {', '.join(top_names)}")
        if top_cpu_procs:
            top_names = [f"`{p['name']}` ({p.get('cpu_percent', 0)}%)" for p in top_cpu_procs[:3]]
            observations.append(f"Top CPU Consumers: {', '.join(top_names)}")

        # --- Determine Overall Status & Primary Issue ---
        if not metrics_summary:
            status = "unknown"
            primary_issue = "Unable to read system telemetry metrics."
            spoken_summary = "System diagnostic data could not be retrieved."
            recommendations.append("Ensure CyberJarvis has permission to access system metrics.")
        elif any(sev == 2 for _, _, sev in issues):
            status = "critical"
            # Sort critical issues: RAM > CPU > SWAP > DISK > LOAD > NETWORK
            crit_issues = [desc for _, desc, sev in issues if sev == 2]
            primary_issue = crit_issues[0]
            secondary_issues = [desc for _, desc, _ in issues if desc != primary_issue]

            # Spoken summary (under 2 sentences)
            top_culprit = ""
            if "RAM" in primary_issue and top_mem_procs:
                top_culprit = f", primarily consumed by {top_mem_procs[0]['name']}"
            elif "CPU" in primary_issue and top_cpu_procs:
                top_culprit = f", driven by {top_cpu_procs[0]['name']}"

            spoken_summary = f"System bottleneck detected. {primary_issue.split('.')[0]}{top_culprit}. {recommendations[0] if recommendations else 'Check running applications.'}"

        elif any(sev == 1 for _, _, sev in issues):
            status = "warning"
            warn_issues = [desc for _, desc, sev in issues if sev == 1]
            primary_issue = warn_issues[0]
            secondary_issues = [desc for _, desc, _ in issues if desc != primary_issue]

            spoken_summary = f"System warning: {primary_issue.split('.')[0]}. Performance may be slightly degraded."

        else:
            status = "healthy"
            primary_issue = "none"
            secondary_issues = []
            findings.insert(0, "All monitored system telemetry metrics are within normal operational limits.")
            recommendations.append("System is operating normally. No remediation required.")
            spoken_summary = "Your system is currently healthy. CPU, memory, swap and disk usage are within normal ranges."

        # Compile findings
        for _, desc, _ in issues:
            findings.append(desc)

        report = DiagnosticReport(
            status=status,
            observations=observations,
            findings=findings,
            primary_issue=primary_issue,
            secondary_issues=secondary_issues,
            recommendations=recommendations,
            spoken_summary=spoken_summary,
            metrics=metrics_summary,
            top_memory_consumers=top_mem_procs,
            top_cpu_consumers=top_cpu_procs,
        )

        display_markdown = self._format_markdown_report(report)
        return report, display_markdown

    def _analyze_process_query(
        self,
        user_lower: str,
        top_mem_procs: List[Dict[str, Any]],
        top_cpu_procs: List[Dict[str, Any]],
        mem_data: Dict[str, Any],
        cpu_val: Optional[float],
    ) -> Tuple[DiagnosticReport, str]:
        """Dedicated grounded synthesis for queries asking which process is consuming memory or CPU."""
        is_cpu = "cpu" in user_lower or "processor" in user_lower
        procs = top_cpu_procs if is_cpu else top_mem_procs
        res_type = "CPU" if is_cpu else "memory"
        pct_key = "cpu_percent" if is_cpu else "memory_percent"

        if not procs:
            answer = f"No active process metrics could be retrieved for {res_type}."
            spoken = f"Unable to read process {res_type} usage."
            report = DiagnosticReport(
                status="unknown",
                observations=[],
                findings=[answer],
                primary_issue="none",
                recommendations=["Check psutil process access permissions."],
                spoken_summary=spoken,
            )
            return report, answer

        top = procs[0]
        name = top.get("name", "unknown")
        pid = top.get("pid", "N/A")
        pct = top.get(pct_key, 0.0)

        lines = [
            f"Among the currently running processes, **{name}** (PID `{pid}`) is consuming the most {res_type} at **{pct}%**."
        ]

        if not is_cpu and mem_data:
            lines.append(f"*(Total system RAM usage: {mem_data.get('percent', 0.0)}% — {mem_data.get('used_gb', 0.0)} GB / {mem_data.get('total_gb', 0.0)} GB)*\n")
        elif is_cpu and cpu_val is not None:
            lines.append(f"*(Total system CPU utilization: {cpu_val}%)*\n")

        lines.append(f"**Top 3 {res_type.upper()} Consumers:**")
        for i, p in enumerate(procs[:3], start=1):
            lines.append(f"{i}. `{p.get('name')}` (PID `{p.get('pid')}`) — **{p.get(pct_key)}%**")

        display_markdown = "\n".join(lines)
        spoken_summary = f"Among active processes, {name} is consuming the most {res_type} at {pct} percent."

        report = DiagnosticReport(
            status="healthy" if pct < 50.0 else ("warning" if pct < 80.0 else "critical"),
            observations=[f"Top {res_type}: {name} ({pct}%)"],
            findings=[f"{name} is consuming {pct}% of {res_type}."],
            primary_issue="none" if pct < 50.0 else f"High {res_type} consumption by {name} ({pct}%)",
            recommendations=[f"Inspect or terminate process {name} if not needed."] if pct >= 50.0 else ["No action required."],
            spoken_summary=spoken_summary,
            top_memory_consumers=top_mem_procs,
            top_cpu_consumers=top_cpu_procs,
        )
        return report, display_markdown

    def _format_markdown_report(self, report: DiagnosticReport) -> str:
        status_icons = {
            "healthy": "🟢 **HEALTHY**",
            "warning": "🟡 **WARNING**",
            "critical": "🔴 **CRITICAL**",
            "unknown": "⚪ **UNKNOWN**",
        }
        verdict_str = status_icons.get(report.status, report.status.upper())

        lines = [f"### System Diagnostic Report: {verdict_str}"]

        # Detailed metrics list
        m = report.metrics
        if "cpu_percent" in m:
            lines.append(f"- **CPU Usage**: {m['cpu_percent']}%")
        if "ram_percent" in m:
            lines.append(f"- **RAM Usage**: {m['ram_percent']}% ({m.get('ram_used_gb', 'N/A')} GB / {m.get('ram_total_gb', 'N/A')} GB)")
        if "swap_percent" in m:
            lines.append(f"- **Swap Usage**: {m['swap_percent']}% ({m.get('swap_used_gb', '0')} GB used)")
        if "disk_percent" in m:
            lines.append(f"- **Disk Usage**: {m['disk_percent']}% ({m.get('disk_free_gb', 'N/A')} GB free)")
        if "load_1m" in m:
            lines.append(f"- **Load Average**: {m['load_1m']} (1m) on {m.get('cpu_cores', 1)} CPU cores")
        if "network_connected" in m:
            net_status = "Online" if m["network_connected"] else "Offline / Disconnected"
            lines.append(f"- **Network**: {net_status}")

        # Top processes
        if report.top_memory_consumers:
            procs_str = ", ".join(f"`{p['name']}` ({p.get('memory_percent', 0)}%)" for p in report.top_memory_consumers[:3])
            lines.append(f"- **Top Memory Processes**: {procs_str}")
        if report.top_cpu_consumers:
            procs_str = ", ".join(f"`{p['name']}` ({p.get('cpu_percent', 0)}%)" for p in report.top_cpu_consumers[:3])
            lines.append(f"- **Top CPU Processes**: {procs_str}")

        # Primary issue
        lines.append(f"\n**Primary Issue:**\n- {report.primary_issue.capitalize() if report.primary_issue != 'none' else 'None detected'}")

        # Findings
        if report.findings:
            lines.append("\n**Findings:**")
            for f in report.findings[:4]:
                lines.append(f"- {f}")

        # Recommendations
        if report.recommendations:
            lines.append("\n**Recommendations:**")
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)
