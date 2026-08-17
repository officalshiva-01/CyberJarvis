import ollama

from memory import init_db, save_memory, get_recent_memories
from services.tool_dispatcher import run_tool


MODEL = "phi3:mini"

EXIT_COMMANDS = {
    "exit",
    "quit",
    "bye",
    "/bye",
}


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
            # --------------------------------
            # STEP 4C: TOOL ROUTING
            # --------------------------------

            tool_name = detect_system_tool(user_input)

            if tool_name:
                print("Ultron: Checking system...")

                result = run_tool(tool_name)

                if isinstance(result, dict) and "error" in result:
                    answer = f"I couldn't check that: {result['error']}"
                else:
                    answer = format_system_result(tool_name, result)

            else:
                # --------------------------------
                # NORMAL AI REQUEST
                # --------------------------------

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
