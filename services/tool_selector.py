ALLOWED_INTENTS = {
    "system",
    "file",
    "action",
    "chat",
}


def select_tool(intent: str):
    """
    Convert a validated intent into the corresponding tool category.
    """

    if intent not in ALLOWED_INTENTS:
        return None

    return intent


if __name__ == "__main__":
    tests = [
        "system",
        "file",
        "action",
        "chat",
        "delete",
    ]

    for test in tests:
        print(test, "->", select_tool(test))
