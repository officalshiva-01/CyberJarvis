# 🤖 CyberJarvis 2.0

> **Production-Oriented, Voice-Controlled, Local AI Agent with Real Hardware Diagnostics, Planning, Memory, and Security Guardrails.**

CyberJarvis is an autonomous local AI system running on Linux that transforms natural language and voice requests into multi-step action plans, inspects local computer health, executes secure deterministic tools, evaluates diagnostic evidence, and responds via speech and clean markdown.

---

## 🏛️ Core Architectural Principle

> **"RULES ARE GUARDRAILS, NOT THE BRAIN."**

Rather than hundreds of hardcoded `if/elif` keyword checks, CyberJarvis operates on an **Agentic Loop**:

```text
User (Voice or Text)
  │
  ▼
Speech-to-Text (STT)
  │
  ▼
Agent Core & Short-Term State
  │
  ▼
Planner (Local Ollama / phi3:mini)
  │
  ▼
Structured Plan & Tool Selection (JSON + Pydantic)
  │
  ▼
Security & Policy Validation (Read-Only auto-approved, Mutating guarded)
  │
  ▼
Deterministic Tool Execution (psutil, network, files, git, APIs)
  │
  ▼
Structured Observation Collection
  │
  ▼
Evaluator & Synthesis (LLM Analysis + Bottleneck Detection)
  │
  ▼
Persistent Memory (SQLite Facts & Typed Recency/Importance Ranking)
  │
  ▼
Response Dispatch: Terminal Output + Text-to-Speech (TTS)
```

---

## 📁 Project Architecture

```text
CyberJarvis/
├── main.py                  # Unified entry point (Terminal, Voice, and One-shot modes)
├── config.py                # Central configuration (paths, models, timeouts, API keys)
│
├── agent/                   # Agentic Core Subsystem
│   ├── schemas.py           # Pydantic schemas (AgentPlan, PlanStep, AgentResponse)
│   ├── state.py             # Conversational state & multi-turn memory
│   ├── planner.py           # Ollama structured plan generator & fallback planner
│   ├── executor.py          # Plan execution engine with step-by-step observation handling
│   ├── evaluator.py         # Diagnostic synthesis and spoken summary generator
│   └── agent_core.py        # Central agent loop orchestrator
│
├── tools/                   # Unified Tool Registry Subsystem
│   ├── schemas.py           # ToolDefinition, ToolResult, ToolParameter
│   ├── registry.py          # Central registry with permission & risk metadata
│   ├── system_tools.py      # psutil CPU, RAM, Swap, Disk, Load, Uptime, OS info
│   ├── diagnostic_tools.py  # Deep bottleneck diagnosis, top CPU/RAM processes
│   ├── network_tools.py     # Connectivity test, DNS resolution, interface stats
│   ├── file_tools.py        # Sandboxed project file listing, reading, and deletion
│   ├── git_tools.py         # Working tree status, branches, recent commit logs
│   └── action_tools.py      # Time, date, working directory helpers
│
├── security/                # Policy & Guardrails Subsystem
│   ├── policy.py            # RiskLevel (LOW, MED, HIGH, CRIT) & PermissionLevel
│   ├── permissions.py       # Path traversal checks & process kill protection
│   └── confirmation.py      # Interactive user confirmation handler for mutating actions
│
├── memory/                  # Persistent SQLite Memory Subsystem
│   ├── store.py             # SQLite initialization & automatic column migrations
│   ├── retrieval.py         # Typed memory retrieval, search, and persistent facts
│   └── ranking.py           # Importance and recency ranking algorithms
│
├── voice/                   # Voice I/O Subsystem
│   ├── speech_to_text.py    # Speech-to-Text with microphone & keyboard fallback
│   ├── text_to_speech.py    # Text-to-Speech via Linux spd-say / pyttsx3 / terminal
│   └── voice_loop.py        # Continuous conversational voice loop
│
├── integrations/            # External API Subsystem
│   ├── api_client.py        # Robust HTTP client (httpx, timeout, error handling)
│   ├── github_api.py        # Remote GitHub repo and commit querying
│   └── weather_api.py       # Live weather queries without mandatory keys
│
├── observability/           # Telemetry & Logging Subsystem
│   ├── logging.py           # Structured logger with secret sanitization
│   └── tracing.py           # Performance and latency span tracing
│
└── tests/                   # Automated Unit & Integration Tests
    ├── test_tools.py
    ├── test_security.py
    ├── test_memory.py
    ├── test_planner.py
    ├── test_agent_core.py
    ├── test_integrations.py
    └── test_voice.py
```

---

## 🚀 Quick Start

### 1. Requirements & Prerequisites
- Linux OS (tested on Ubuntu/Debian)
- Python 3.10+
- [Ollama](https://ollama.com/) running locally with `phi3:mini`:
  ```bash
  ollama run phi3:mini
  ```
- Speech Dispatcher (preinstalled on most Linux desktop distributions):
  ```bash
  which spd-say
  ```

### 2. Run Modes

#### A. Interactive Terminal Agent (Default)
```bash
python main.py
```
*Supports real-time agent status indicators (`🧠 Planning...`, `🔧 Running: <tool>`, `📊 Analyzing...`, `✅ Diagnosis ready`), detailed markdown outputs, and spoken responses.*

#### B. Voice Assistant Mode
```bash
python main.py --voice
```
*Listens to voice input from your microphone, runs the complete agentic planning and diagnosis, and speaks the verdict back to you.*

#### C. Instant Diagnostic One-Shot
```bash
python main.py --diagnose
```
*Immediately runs comprehensive hardware, memory, and process diagnostics without entering interactive mode.*

#### D. Disable Audio Output
```bash
python main.py --no-tts
```

---

## 🛡️ Security & Guardrails

1. **Read-Only Auto-Approval**: Diagnostic queries (CPU, RAM, disk, processes, git logs, file reads) execute automatically without interrupting the user.
2. **Mutating Action Guardrails**: Mutating actions (e.g. deleting files, killing processes) require explicit user confirmation before execution.
3. **Critical Operation Blocking**: Destructive system operations (formatting drives, modifying root system files) are blocked at the policy layer.
4. **Sandboxed File Access**: File operations are strictly locked to the project root with path traversal validation preventing access to sensitive files (`.ssh`, `/etc/shadow`, etc.).
5. **Never Arbitrary Shell**: The LLM cannot execute raw arbitrary shell commands; all actions route strictly through verified deterministic tools.

---

## 🧪 Running Automated Tests

Run the full test suite (27 unit and integration tests):

```bash
.venv/bin/python -m unittest discover tests
```

---

## 💾 Persistent Memory

CyberJarvis maintains long-term knowledge in `cyberjarvis.db`:
- **Personal Facts**: Statements like *"My name is Shivaji"* or *"My goal is to become a DevOps engineer"* are automatically extracted and indexed.
- **Typed Memories**: Categorized by type (`conversation`, `goal`, `project`, `preference`).
- **Relevance & Importance**: Ranked using both priority weight and timestamp recency for contextual grounding.
