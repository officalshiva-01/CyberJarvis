def route_intent(system_tool, file_tool, action_tool):
    if system_tool:
        return "system"

    if file_tool:
        return "file"

    if action_tool:
        return "action"

    return "chat"
