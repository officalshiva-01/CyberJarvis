from typing import Optional, Callable
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Security")


def format_confirmation_prompt(tool_name: str, reason: str, details: Optional[dict] = None) -> str:
    """Generate user-facing confirmation prompt for security-gated actions."""
    if tool_name == "create_project_backup":
        return "This will create a backup archive of ~/CyberJarvis. Proceed? [y/N]"
    if tool_name == "write_project_file":
        fn = details.get("filename", "file") if details else "file"
        return f"This will write or update project file '{fn}'. Proceed? [y/N]"
    if tool_name == "restart_project_service":
        svc = details.get("service_name", "service") if details else "service"
        return f"This will restart project service '{svc}'. Proceed? [y/N]"
    if tool_name == "delete_project_file":
        fn = details.get("filename", "file") if details else "file"
        return f"This will delete project file '{fn}'. Proceed? [y/N]"
    return f"This will execute '{tool_name}' ({reason}). Proceed? [y/N]"


class ConfirmationHandler:
    """Manages user confirmation prompts for mutating and high-risk actions."""

    def __init__(self, voice_prompt_fn: Optional[Callable[[str], bool]] = None):
        self.voice_prompt_fn = voice_prompt_fn

    def confirm(self, tool_name: str, reason: str, details: Optional[dict] = None) -> bool:
        prompt_text = format_confirmation_prompt(tool_name, reason, details)
        logger.warning(f"Requesting confirmation for '{tool_name}': {prompt_text}")

        # If custom voice/interactive prompt function provided:
        if self.voice_prompt_fn:
            return self.voice_prompt_fn(prompt_text)

        # Standard terminal confirmation
        print(f"\n⚠️  {prompt_text}", end=" ", flush=True)
        try:
            choice = input().strip().lower()
            authorized = choice in ("yes", "y")
            if authorized:
                logger.info(f"Action '{tool_name}' was confirmed by user.")
            else:
                logger.info(f"Action '{tool_name}' was rejected by user.")
            return authorized
        except (KeyboardInterrupt, EOFError):
            print("\nAction cancelled.")
            return False
