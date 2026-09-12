import sys
from pathlib import Path

# Add project root to sys.path if needed
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools

# Mapping of legacy tool names to registered tool names
TOOL_NAME_MAP = {
    "cpu": "get_cpu_usage",
    "memory": "get_memory_usage",
    "disk": "get_disk_usage",
    "list_files": "list_project_files",
    "read_file": "read_project_file",
    "get_time": "get_time",
    "get_date": "get_date",
    "get_current_directory": "get_current_directory",
}


def run_tool(tool_name: str, *args):
    """
    Backward-compatible tool runner.
    Maps legacy names and dispatches through the unified ToolRegistry.
    """
    registered_name = TOOL_NAME_MAP.get(tool_name, tool_name)
    kwargs = {}
    if registered_name == "read_project_file" and args:
        kwargs["filename"] = args[0]
    elif registered_name == "get_top_processes" and args:
        kwargs["limit"] = args[0]

    result = tools.registry.execute(registered_name, parameters=kwargs)
    if result.success:
        return result.output
    else:
        return {"error": result.error}


if __name__ == "__main__":
    print("CPU:", run_tool("cpu"))
    print("Memory:", run_tool("memory"))
    print("Disk:", run_tool("disk"))
    print("Files:", run_tool("list_files"))
    print("Time:", run_tool("get_time"))
    print("Date:", run_tool("get_date"))
    print("Directory:", run_tool("get_current_directory"))
