try:
    from services.file_tools import (
        list_project_files,
        read_project_file,
    )
    from services.system_tools import (
        get_cpu_usage,
        get_memory_usage,
        get_disk_usage,
    )
except ModuleNotFoundError:
    from file_tools import (
        list_project_files,
        read_project_file,
    )
    from system_tools import (
        get_cpu_usage,
        get_memory_usage,
        get_disk_usage,
    )


def run_tool(tool_name: str, *args):
    tools = {
        "cpu": get_cpu_usage,
        "memory": get_memory_usage,
        "disk": get_disk_usage,
        "list_files": list_project_files,
        "read_file": read_project_file,
    }

    tool = tools.get(tool_name)

    if tool is None:
        return {
            "error": f"Unknown or unauthorized tool: {tool_name}"
        }

    try:
        return tool(*args)
    except Exception as error:
        return {
            "error": str(error)
        }


if __name__ == "__main__":
    print("CPU:", run_tool("cpu"))
    print("Memory:", run_tool("memory"))
    print("Disk:", run_tool("disk"))
    print("Files:", run_tool("list_files"))
