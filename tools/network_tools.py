import socket
import psutil
from typing import Dict, Any, List

from tools.registry import registry
from tools.schemas import IsolationMode
from security.policy import RiskLevel, PermissionLevel


@registry.register(
    name="check_network_connectivity",
    description="Test internet connectivity by connecting to a reliable public host.",
    parameters={
        "host": {"type": "string", "description": "Host IP to ping/connect (default '1.1.1.1')", "default": "1.1.1.1"},
        "port": {"type": "integer", "description": "Port to test (default 53 DNS)", "default": 53},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
)
def check_network_connectivity(host: str = "1.1.1.1", port: int = 53) -> Dict[str, Any]:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.5)
        sock.connect((host, port))
        return {
            "connected": True,
            "target": f"{host}:{port}",
            "status": "Online and reachable",
        }
    except Exception as ex:
        return {
            "connected": False,
            "target": f"{host}:{port}",
            "status": f"Unreachable ({type(ex).__name__}): {ex}",
        }
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


@registry.register(
    name="check_dns_resolution",
    description="Test DNS resolution capability for domain names.",
    parameters={
        "domain": {"type": "string", "description": "Domain to resolve (default 'github.com')", "default": "github.com"},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
)
def check_dns_resolution(domain: str = "github.com") -> Dict[str, Any]:
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(1.5)
        resolved_ip = socket.gethostbyname(domain)
        return {
            "success": True,
            "domain": domain,
            "resolved_ip": resolved_ip,
        }
    except Exception as ex:
        return {
            "success": False,
            "domain": domain,
            "error": str(ex),
        }
    finally:
        socket.setdefaulttimeout(old_timeout)


@registry.register(
    name="get_network_interfaces",
    description="List active network interfaces, IP addresses, and operational status.",
    parameters={},
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.THREAD,
)
def get_network_interfaces() -> List[Dict[str, Any]]:
    interfaces = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for iface_name, addr_list in addrs.items():
        stat = stats.get(iface_name)
        ip_addresses = []
        for addr in addr_list:
            if addr.family == socket.AF_INET:
                ip_addresses.append(addr.address)

        interfaces.append({
            "interface": iface_name,
            "is_up": stat.isup if stat else False,
            "speed_mbps": stat.speed if stat else 0,
            "ipv4": ip_addresses,
        })
    return interfaces
