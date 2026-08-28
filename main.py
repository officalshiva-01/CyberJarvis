import re
import ollama
from memory import init_db, save_memory, get_recent_memories
from services.tool_dispatcher import run_tool
from services.intent_router import route_intent


MODEL = "phi3:mini"

EXIT_COMMANDS = {
    "exit",
    "quit",
    "bye",
    "/bye",
}


def detect_action_tool(text: str):
    """
    Detect safe read-only local action requests.

    Returns:
        "get_time", "get_date", "get_current_directory"
        or None
    """

    text_lower = text.lower().strip()

    time_keywords = [
        "what time is it",
        "current time",
        "tell me the time",
        "time now",
    ]

    date_keywords = [
        "what is today's date",
        "what is the date",
        "current date",
        "today's date",
        "date today",
    ]

    directory_keywords = [
        "current directory",
        "where am i",
        "what directory am i in",
        "working directory",
        "current folder",
    ]

    if any(keyword in text_lower for keyword in time_keywords):
        return "get_time"

    if any(keyword in text_lower for keyword in date_keywords):
        return "get_date"

    if any(keyword in text_lower for keyword in directory_keywords):
        return "get_current_directory"

    return None


def detect_system_tool(text: str):
    """
    Detect simple system-monitoring requests.

    Returns:
        "cpu", "memory", "disk", or None
    """

    text = text.lower()

    cpu_keywords = [
        "cpu",
        "processor usage",
        "processor load",
    ]

    memory_keywords = [
        "ram",
        "memory usage",
        "memory utilization",
    ]

    disk_keywords = [
        "disk",
        "storage",
        "disk space",
        "free space",
    ]

    if any(keyword in text for keyword in cpu_keywords):
        return "cpu"

    if any(keyword in text for keyword in memory_keywords):
        return "memory"

    if any(keyword in text for keyword in disk_keywords):
        return "disk"

    return None


def detect_file_tool(text: str):
    """
    Detect safe read-only project file requests.

    Returns:
        ("list_files", None)
        ("read_file", filename)
        or (None, None)
    """

    text_lower = text.lower().strip()

    list_keywords = [
        "list files",
        "show files",
        "what files",
        "which files",
        "files in my project",
        "project files",
        "files in cyberjarvis",
    ]

    if any(keyword in text_lower for keyword in list_keywords):
        return "list_files", None

    match = re.search(
        r"\b(?:read|show|open|display|contents?\s+of)\s+([A-Za-z0-9_./-]+)",
        text_lower,
    )

    if match:
        filename = match.group(1).removeprefix("./")
        return "read_file", filename

    return None, None


def build_context():
    memories = get_recent_memories(limit=10)

    context = []

    for user_message, assistant_response in reversed(memories):
        context.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        context.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

    return context


def ask_ultron(user_input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Ultron, a helpful local AI assistant. "
                "Answer clearly and concisely. "
                "Use the provided conversation context. "
                "Do not invent system statistics."
            ),
        }
    ]

    messages.extend(build_context())

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    response = ollama.chat(
        model=MODEL,
        messages=messages,
    )

    return response["message"]["content"]


def format_system_result(tool_name, result):
    if tool_name == "cpu":
        return f"Current CPU usage is {result}%."

    if tool_name == "memory":
        return (
            f"RAM usage is {result['percent']}%. "
            f"{result['used_gb']} GB of {result['total_gb']} GB is currently used."
        )

    if tool_name == "disk":
        return (
            f"Disk usage is {result['percent']}%. "
            f"{result['free_gb']} GB is free out of {result['total_gb']} GB."
        )

    return str(result)


def format_file_result(tool_name, result):
    if isinstance(result, dict) and "error" in result:
        return f"I couldn't access that file: {result['error']}"

    if tool_name == "list_files":
        if not result:
            return "The CyberJarvis project directory is empty."

        lines = []

        for item in result:
            icon = "📁" if item["type"] == "directory" else "📄"
            lines.append(f"{icon} {item['name']}")

        return "Files in the CyberJarvis project:\n" + "\n".join(lines)

    if tool_name == "read_file":
        content = result.get("content", "")
        max_chars = 12000

        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Output truncated for safety.]"

        return f"Contents of {result['filename']}:\n\n{content}"

    return str(result)


def format_action_result(tool_name, result):
    if isinstance(result, dict) and "error" in result:
        return f"I couldn't perform that action: {result['error']}"

    if tool_name == "get_time":
        return f"The current time is {result}."

    if tool_name == "get_date":
        return f"Today is {result}."

    if tool_name == "get_current_directory":
        return f"The current directory is {result}."

    return str(result)


def main():
    init_db()

    print("\n🤖 CyberJarvis AI")
    print("Persistent memory: ON")
    print("System tools: CPU / RAM / Disk")
    print("Type 'exit', 'quit', 'bye', or '/bye' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nUltron: Goodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Ultron: Goodbye.")
            break

        try:
            system_tool = detect_system_tool(user_input)
            file_tool, filename = detect_file_tool(user_input)
            action_tool = detect_action_tool(user_input)

            intent = route_intent(system_tool, file_tool, action_tool)

            if intent == "system":
                print("Ultron: Checking system...")

                result = run_tool(system_tool)

                if isinstance(result, dict) and "error" in result:
                    answer = f"I couldn't check that: {result['error']}"
                else:
                    answer = format_system_result(system_tool, result)

            elif intent == "file":
                print("Ultron: Checking project files...")

                if file_tool == "read_file":
                    result = run_tool(file_tool, filename)
                else:
                    result = run_tool(file_tool)

                answer = format_file_result(file_tool, result)

            elif intent == "action":
                print("Ultron: Checking...")

                result = run_tool(action_tool)
                answer = format_action_result(action_tool, result)

            else:
                print("Ultron: ", end="", flush=True)
                answer = ask_ultron(user_input)

            print(answer)
            print()

            # Save conversation to persistent memory
            save_memory(user_input, answer)

        except Exception as error:
            print(f"\n❌ Error: {error}\n")


if __name__ == "__main__":
    main()
