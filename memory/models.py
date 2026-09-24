"""
Memory Models and Types for CyberJarvis Phase 7: Advanced Memory 2.0.
Defines supported memory types, confidence scales, status lifecycle, and the unified MemoryRecord model.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Optional, Dict, Any, Tuple


class MemoryType(str, Enum):
    USER_PROFILE = "USER_PROFILE"
    PROJECT = "PROJECT"
    TASK = "TASK"
    DECISION = "DECISION"
    ISSUE = "ISSUE"
    SESSION = "SESSION"
    TOOL_RESULT = "TOOL_RESULT"

    @classmethod
    def normalize(cls, raw_type: Optional[str]) -> str:
        """Normalize type string to canonical MemoryType name with legacy backward-compatibility aliases."""
        if not raw_type:
            return cls.SESSION.value
        clean = raw_type.strip().upper()

        # Legacy aliases
        if clean in ("CONVERSATION", "CHAT", "INTERACTION"):
            return cls.SESSION.value
        if clean in ("FACT", "PROFILE", "USER", "PREFERENCE", "GOAL", "USER_FACT"):
            return cls.USER_PROFILE.value
        if clean in ("GENERAL", "ARCH", "ARCHITECTURE", "REPO"):
            return cls.PROJECT.value
        if clean in ("BUG", "ERROR", "FAILURE", "LIMITATION", "BOTTLENECK"):
            return cls.ISSUE.value
        if clean in ("TODO", "JOB", "ACTION"):
            return cls.TASK.value

        for member in cls:
            if clean == member.value:
                return member.value

        return clean


class Confidence(float, Enum):
    HIGH = 1.0         # Explicit user assertions ("My name is...", "My goal is...")
    EXPLICIT_USER = 1.0
    TOOL = 0.9         # Tool-derived verifiable facts (git commit, pytest exit, system specs)
    FACT = 0.9
    MEDIUM = 0.7       # Structured system observations
    INFERRED = 0.5     # LLM assumptions or soft inferences
    LOW = 0.3          # Uncertain or tentative observations


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class MemoryRecord:
    """Unified durable agent-grade memory record."""
    id: Optional[int] = None
    memory_type: str = MemoryType.SESSION.value
    content: str = ""
    importance: int = 3
    confidence: float = Confidence.HIGH
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source: str = "user"
    project: Optional[str] = "CyberJarvis"
    task_id: Optional[str] = None
    status: str = MemoryStatus.ACTIVE.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None

    def __post_init__(self):
        self.memory_type = MemoryType.normalize(self.memory_type)
        if not self.content and self.user_message:
            if self.assistant_response:
                self.content = f"User: {self.user_message}\nAssistant: {self.assistant_response}"
            else:
                self.content = self.user_message
        if not self.user_message and self.content:
            self.user_message = self.content
        if not self.assistant_response:
            self.assistant_response = ""

        # Clamp importance to 1..5
        try:
            self.importance = max(1, min(5, int(self.importance)))
        except (ValueError, TypeError):
            self.importance = 3

        # Clamp confidence to 0.0..1.0
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (ValueError, TypeError):
            self.confidence = 1.0

        if not isinstance(self.metadata, dict):
            if isinstance(self.metadata, str):
                try:
                    self.metadata = json.loads(self.metadata)
                except Exception:
                    self.metadata = {}
            else:
                self.metadata = {}

    def to_legacy_tuple(self) -> Tuple[str, str, str, str, int]:
        """Convert to legacy (user_message, assistant_response, created_at, memory_type, importance) tuple."""
        return (
            self.user_message or self.content,
            self.assistant_response or "",
            self.created_at or "",
            self.memory_type,
            self.importance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "project": self.project,
            "task_id": self.task_id,
            "status": self.status,
            "metadata": dict(self.metadata),
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
        }

    @classmethod
    def from_row(cls, row: Tuple[Any, ...], column_names: Tuple[str, ...]) -> "MemoryRecord":
        """Build MemoryRecord dynamically from a sqlite row tuple and cursor.description."""
        row_dict = dict(zip(column_names, row))
        raw_meta = row_dict.get("metadata", "{}")
        parsed_meta = {}
        if isinstance(raw_meta, str) and raw_meta.strip():
            try:
                parsed_meta = json.loads(raw_meta)
            except Exception:
                parsed_meta = {}
        elif isinstance(raw_meta, dict):
            parsed_meta = raw_meta

        return cls(
            id=row_dict.get("id"),
            memory_type=row_dict.get("memory_type", MemoryType.SESSION.value),
            content=row_dict.get("content") or row_dict.get("user_message", ""),
            importance=row_dict.get("importance", 3),
            confidence=row_dict.get("confidence", 1.0),
            created_at=row_dict.get("created_at"),
            updated_at=row_dict.get("updated_at") or row_dict.get("created_at"),
            source=row_dict.get("source", "user"),
            project=row_dict.get("project", "CyberJarvis"),
            task_id=row_dict.get("task_id"),
            status=row_dict.get("status", MemoryStatus.ACTIVE.value),
            metadata=parsed_meta,
            user_message=row_dict.get("user_message"),
            assistant_response=row_dict.get("assistant_response"),
        )
