"""
CyberJarvis Phase 7: Advanced Memory 2.0 Subsystem.
Exports durable models, retrieval, ranking, lifecycle management,
context building, and backward-compatible APIs.
"""
from memory.store import init_db, get_connection
from memory.models import MemoryType, MemoryRecord, Confidence, MemoryStatus
from memory.sanitizer import sanitize_memory_content, sanitize_memory_metadata
from memory.retrieval import (
    retrieve_memories,
    save_memory,
    get_recent_memories,
    get_memories_by_type,
    search_memories,
    search_and_rank_memories,
    save_fact,
    get_facts,
)
from memory.ranking import rank_memories, rank_memory_records, score_memory
from memory.lifecycle import (
    create_memory,
    get_memory_by_id,
    update_memory,
    supersede_memory,
    archive_memory,
    delete_memory,
    merge_duplicate_memories,
)
from memory.extraction import extract_memory_from_text, auto_extract_and_save_memory
from memory.context_builder import MemoryContextBuilder

__all__ = [
    # Database
    "init_db",
    "get_connection",
    # Models
    "MemoryType",
    "MemoryRecord",
    "Confidence",
    "MemoryStatus",
    # Sanitization
    "sanitize_memory_content",
    "sanitize_memory_metadata",
    # Retrieval
    "retrieve_memories",
    "save_memory",
    "get_recent_memories",
    "get_memories_by_type",
    "search_memories",
    "search_and_rank_memories",
    "save_fact",
    "get_facts",
    # Ranking
    "rank_memories",
    "rank_memory_records",
    "score_memory",
    # Lifecycle
    "create_memory",
    "get_memory_by_id",
    "update_memory",
    "supersede_memory",
    "archive_memory",
    "delete_memory",
    "merge_duplicate_memories",
    # Extraction
    "extract_memory_from_text",
    "auto_extract_and_save_memory",
    # Context Builder
    "MemoryContextBuilder",
]
