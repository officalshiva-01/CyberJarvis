import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DB_PATH = Path(os.getenv("CYBERJARVIS_DB_PATH", str(PROJECT_ROOT / "cyberjarvis.db"))).resolve()
LOG_DIR = PROJECT_ROOT / "logs"

# Ollama Model Configuration
DEFAULT_MODEL = os.getenv("CYBERJARVIS_MODEL", "phi3:mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# External APIs
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

# Security Policy
AUTO_APPROVE_READ_ONLY = True
REQUIRE_CONFIRMATION_FOR_MUTATING = True

# Voice Engine Settings
TTS_ENGINE = os.getenv("CYBERJARVIS_TTS", "spd-say")  # Options: spd-say, pyttsx3, none
STT_ENGINE = os.getenv("CYBERJARVIS_STT", "speech_recognition") # Options: speech_recognition, text

# Timeout & Limit settings
HTTP_TIMEOUT_SECONDS = 10.0
COMMAND_TIMEOUT_SECONDS = 15.0
MAX_TOOL_ITERATIONS = 8
MAX_FILE_READ_CHARS = 12000
