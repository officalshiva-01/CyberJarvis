try:
    from services.tool_selector import select_tool
except ModuleNotFoundError:
    from tool_selector import select_tool


def route_intent(system_tool, file_tool, action_tool):
    """
    Determine the requested intent and validate it
    through the tool selector.
    """

    if system_tool:
        return select_tool("system")

    if file_tool:
        return select_tool("file")

    if action_tool:
        return select_tool("action")

    return select_tool("chat")


if __name__ == "__main__":
    tests = [
        ("cpu", None, None),
        (None, "read_file", None),
        (None, None, "get_time"),
        (None, None, None),
    ]

    for system_tool, file_tool, action_tool in tests:
        print(
            route_intent(
                system_tool,
                file_tool,
                action_tool,
            )
        )
