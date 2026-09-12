from typing import Optional, Callable
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Security")


class ConfirmationHandler:
    """Manages user confirmation prompts for mutating and high-risk actions."""

    def __init__(self, voice_prompt_fn: Optional[Callable[[str], bool]] = None):
        self.voice_prompt_fn = voice_prompt_fn

    def confirm(self, tool_name: str, reason: str, details: Optional[dict] = None) -> bool:
        prompt_text = f"Security Alert: '{tool_name}' requires confirmation. Reason: {reason}"
        if details:
            prompt_text += f"\nDetails: {details}"

        logger.warning(f"Requesting confirmation for '{tool_name}': {reason}")

        # If custom voice/interactive prompt function provided:
        if self.voice_prompt_fn:
            return self.voice_prompt_fn(prompt_text)

        # Standard terminal confirmation
        print(f"\n⚠️  {prompt_text}")
        try:
            choice = input("Do you authorize this action? (yes/no): ").strip().lower()
            authorized = choice in ("yes", "y")
            if authorized:
                logger.info(f"Action '{tool_name}' was confirmed by user.")
            else:
                logger.info(f"Action '{tool_name}' was rejected by user.")
            return authorized
        except (KeyboardInterrupt, EOFError):
            print("\nAction cancelled.")
            return False
