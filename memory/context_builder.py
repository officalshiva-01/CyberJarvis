"""
MemoryContextBuilder for CyberJarvis Phase 7: Advanced Memory 2.0.
Provides deterministic, bounded, deduplicated context injection for LLM planning & synthesis.
Enforces strict <= 2500 character limits, profile-specific type prioritization, and generic query isolation.
"""
import re
from typing import Optional, List, Dict, Set, Union
from memory.models import MemoryRecord, MemoryType
from memory.retrieval import retrieve_memories
from observability.logging import get_logger

logger = get_logger("CyberJarvis.MemoryContextBuilder")

DEFAULT_MAX_CONTEXT_CHARS = 2500

EXPLICIT_PERSONALIZATION_TRIGGERS = [
    "for me", "my career", "my goal", "my background", "give me advice",
    "advise me", "based on my", "tailor for me", "personalized",
    "for my situation", "about me", "my project", "my roadmap",
    "how should i learn", "what should i learn", "what career",
]


class MemoryContextBuilder:
    """Builds bounded, deduplicated memory context for Agent profiles and information routing."""

    def __init__(
        self,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        project: str = "CyberJarvis",
        db_path: Optional[str] = None,
    ):
        self.max_chars = max_chars
        self.project = project
        self.db_path = db_path

    def should_inject_personal_memory(self, user_input: str) -> bool:
        """
        Check if an informational query explicitly requests personalized guidance.
        Generic queries ('How does Python work?') MUST return False.
        """
        if not user_input:
            return False
        lower = user_input.lower().strip()
        return any(trig in lower for trig in EXPLICIT_PERSONALIZATION_TRIGGERS)

    def build_information_context(self, user_input: str) -> str:
        """
        Build memory context specifically for the INFORMATION path.
        Strict rule: Returns empty string for generic informational queries.
        Only injects USER_PROFILE context if the user explicitly asked for personalized advice.
        """
        if not self.should_inject_personal_memory(user_input):
            return ""

        records = retrieve_memories(
            query=user_input,
            memory_type=MemoryType.USER_PROFILE.value,
            min_importance=3,
            limit=5,
            db_path=self.db_path,
        )
        if not records:
            records = retrieve_memories(
                memory_type=MemoryType.USER_PROFILE.value,
                min_importance=3,
                limit=5,
                db_path=self.db_path,
            )
        if not records:
            return ""

        lines = ["User Background & Goals:"]
        for r in records:
            lines.append(f"- {r.content}")

        context = "\n".join(lines)
        return context[:800]  # Bounded for information path

    def build_context(
        self,
        query: str,
        task_profile: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        limit: int = 12,
    ) -> str:
        """
        Build comprehensive, deduplicated, bounded memory context for Agent profiles.
        Strictly limits total output to <= max_chars (default 2500).
        """
        # Determine priority memory types based on task profile
        types_to_fetch = memory_types
        if not types_to_fetch and task_profile:
            profile_upper = task_profile.upper()
            if "ARCHITECTURE" in profile_upper:
                types_to_fetch = [MemoryType.PROJECT.value, MemoryType.DECISION.value]
            elif "INSPECTION" in profile_upper:
                types_to_fetch = [MemoryType.PROJECT.value, MemoryType.TASK.value, MemoryType.ISSUE.value]
            elif "DEBUGGING" in profile_upper:
                types_to_fetch = [MemoryType.ISSUE.value, MemoryType.TASK.value, MemoryType.DECISION.value]
            elif "DEPLOYMENT" in profile_upper:
                types_to_fetch = [MemoryType.PROJECT.value, MemoryType.DECISION.value, MemoryType.ISSUE.value]
            elif "PLANNING" in profile_upper:
                types_to_fetch = [MemoryType.PROJECT.value, MemoryType.TASK.value, MemoryType.USER_PROFILE.value]

        # Retrieve candidates
        candidates = retrieve_memories(
            query=query,
            memory_type=types_to_fetch,
            project=self.project,
            task_id=task_id,
            limit=limit,
            db_path=self.db_path,
        )

        if not candidates:
            return ""

        # Deduplicate candidates by normalized text content
        deduped: List[MemoryRecord] = []
        seen_keys: Set[str] = set()

        for c in candidates:
            norm_key = re.sub(r"[^\w\s]", "", c.content.lower()).strip()
            if norm_key and norm_key not in seen_keys:
                seen_keys.add(norm_key)
                deduped.append(c)

        if not deduped:
            return ""

        # Categorize by memory type
        sections: Dict[str, List[str]] = {
            "PROJECT": [],
            "DECISION": [],
            "TASK": [],
            "ISSUE": [],
            "USER_PROFILE": [],
            "SESSION": [],
        }

        for rec in deduped:
            m_type = rec.memory_type
            if m_type in sections:
                sections[m_type].append(rec.content)
            else:
                sections["PROJECT"].append(rec.content)

        # Assemble markdown blocks
        section_titles = {
            "PROJECT": "Durable Project State",
            "DECISION": "Key Architecture & Technical Decisions",
            "TASK": "Active Tasks & Continuity",
            "ISSUE": "Known Issues & Performance Limits",
            "USER_PROFILE": "User Profile & Context",
        }

        blocks: List[str] = []
        for stype, title in section_titles.items():
            items = sections.get(stype, [])
            if items:
                b_lines = [f"### {title}"]
                for item in items:
                    b_lines.append(f"- {item}")
                blocks.append("\n".join(b_lines))

        full_context = "\n\n".join(blocks).strip()

        # Enforce strict maximum character boundary (max_chars <= 2500)
        if len(full_context) > self.max_chars:
            logger.info(f"Memory context truncated from {len(full_context)} to {self.max_chars} chars.")
            truncated = full_context[:self.max_chars]
            # Cleanly break at last newline if possible
            last_nl = truncated.rfind("\n")
            if last_nl > self.max_chars * 0.8:
                full_context = truncated[:last_nl].strip()
            else:
                full_context = truncated.strip()

        return full_context
