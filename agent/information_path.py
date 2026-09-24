"""
Tier-3 Lightweight Information / Direct-LLM Path Handler for CyberJarvis.
Handles general informational, educational, and conceptual questions directly
via a single focused Ollama LLM call without executing local tools or invoking
the heavy Planner/Evaluator pipeline.
"""
import time
import re
from typing import Optional, List, Tuple, Dict, Any
import ollama

from config import DEFAULT_MODEL
from agent.schemas import AgentResponse, AgentPlan
from memory import get_facts
from observability.logging import get_logger

logger = get_logger("CyberJarvis.InformationPath")

DEFAULT_INFO_TIMEOUT = 15.0  # Hard timeout in seconds


class InformationPathHandler:
    """Executes informational queries directly using a single focused Ollama LLM call."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_INFO_TIMEOUT):
        self.model = model
        self.timeout = timeout

    def handle(self, user_input: str) -> Tuple[AgentResponse, float]:
        """
        Execute direct LLM generation for informational request.
        Returns:
            (response, llm_ms)
        """
        start_t = time.perf_counter()
        clean_input = user_input.strip()

        # Focused system prompt: direct answers, no tools, structured markdown
        system_prompt = (
            "You are CyberJarvis, a concise and highly knowledgeable local AI assistant.\n"
            "Directly answer the user's question with accurate, well-structured, practical guidance in Markdown.\n"
            "STRICT RULES:\n"
            "1. Answer the question directly and factually without introductory filler, conversational preamble, or addressing the user.\n"
            "2. Do NOT mention the user's name, personal background, personal goals, or the CyberJarvis project unless the user explicitly asks for personalized advice.\n"
            "3. Do NOT mention, recommend, or hallucinate system tools or terminal commands unless specifically asked.\n"
            "4. Keep the response clear, structured, and focused.\n"
            "5. Provide a very concise 1-2 sentence spoken summary at the end under 'SUMMARY: <text>'."
        )

        # Contextual relevance from memory ONLY if the user explicitly asks for personalized advice (Requirement 12 & 16)
        from memory.context_builder import MemoryContextBuilder
        mem_builder = MemoryContextBuilder()
        info_mem_ctx = mem_builder.build_information_context(clean_input)
        if info_mem_ctx:
            system_prompt += f"\nUser background context (use only because user requested personalized advice):\n{info_mem_ctx}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clean_input},
        ]

        try:
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "num_predict": 400,  # Bound output tokens to prevent long runaway generation
                },
            )
            llm_ms = (time.perf_counter() - start_t) * 1000
            raw_answer = resp.get("message", {}).get("content", "").strip()

            spoken_summary = self._extract_spoken_summary(raw_answer, clean_input)
            clean_answer = self._clean_answer_markdown(raw_answer)

            plan = AgentPlan(
                intent="information",
                goal=clean_input,
                steps=[],
            )

            response = AgentResponse(
                answer=clean_answer,
                spoken_summary=spoken_summary,
                plan=plan,
                tool_results=[],
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
            )
            logger.info(f"Information Path completed in {llm_ms:.1f}ms for: '{clean_input}'")
            return response, llm_ms

        except Exception as ex:
            llm_ms = (time.perf_counter() - start_t) * 1000
            logger.warning(f"Ollama direct chat failed ({ex}). Falling back to helpful offline answer.")
            fallback_answer = self._generate_fallback_info(clean_input)
            spoken = self._extract_spoken_summary(fallback_answer, clean_input)
            return AgentResponse(
                answer=fallback_answer,
                spoken_summary=spoken,
                plan=AgentPlan(intent="information", goal=clean_input, steps=[]),
                tool_results=[],
                completed=True,
                planning_ms=0.0,
                evaluator_ms=0.0,
            ), llm_ms

    def _extract_spoken_summary(self, text: str, user_input: str) -> str:
        # Check if SUMMARY: tag exists
        m = re.search(r"(?:SUMMARY|SPOKEN SUMMARY):\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            summary = m.group(1).strip()
            summary = re.sub(r"[*_`#]", "", summary)
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return summary

        # Extract first 1 or 2 clean sentences
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        first_line = lines[0] if lines else f"Here is information regarding {user_input}."
        first_line = re.sub(r"[*_`#]", "", first_line)
        sentences = re.split(r"(?<=[.!?])\s+", first_line)
        spoken = " ".join(sentences[:2])
        if len(spoken) > 200:
            spoken = spoken[:197] + "..."
        return spoken if spoken else f"Here is information about {user_input}."

    def _clean_answer_markdown(self, text: str) -> str:
        # Remove trailing SUMMARY: line from written markdown answer
        cleaned = re.sub(r"\n+(?:SUMMARY|SPOKEN SUMMARY):\s*.+$", "", text, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else text

    def _generate_fallback_info(self, query: str) -> str:
        lower = query.lower()
        if "developer" in lower:
            return (
                "### Becoming a Software Developer\n\n"
                "To become a software developer:\n"
                "1. **Learn Core Fundamentals**: Master one core programming language (Python, JavaScript, Go, or Java).\n"
                "2. **Data Structures & Algorithms**: Understand arrays, hash maps, trees, and algorithmic complexity.\n"
                "3. **Build Real Projects**: Develop web apps, APIs, or automation tools and host them on GitHub.\n"
                "4. **Version Control & Tooling**: Master Git, Linux command line, and debugging.\n"
                "5. **Databases & Networking**: Understand SQL/NoSQL databases and HTTP/REST fundamentals.\n"
                "6. **Deploy & Share**: Deploy applications to cloud platforms and contribute to open source."
            )
        if "devops" in lower:
            return (
                "### What is DevOps?\n\n"
                "**DevOps** is a set of practices, cultural philosophies, and tools that automates and integrates "
                "the processes between software development and IT teams.\n\n"
                "- **CI/CD**: Continuous Integration and Continuous Delivery for rapid, automated deployments.\n"
                "- **Infrastructure as Code**: Managing infrastructure with code using Terraform or Ansible.\n"
                "- **Containerization**: Using Docker and Kubernetes for consistent environments.\n"
                "- **Monitoring & Observability**: Tracking system health with Prometheus, Grafana, and logs."
            )
        if "python" in lower:
            return (
                "### What is Python?\n\n"
                "**Python** is a high-level, interpreted programming language known for its clear, readable syntax "
                "and versatility. It is widely used in web development, automation, data science, machine learning, "
                "and DevOps scripting."
            )
        return (
            f"### Information: {query}\n\n"
            f"Here is information regarding **{query}**.\n\n"
            "Focus on understanding core fundamentals, practical project implementation, and industry best practices."
        )
