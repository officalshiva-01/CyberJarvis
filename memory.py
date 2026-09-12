"""
CyberJarvis Memory Shim.
Preserves backward-compatibility while routing to the modular memory package.
"""
from memory.store import DB_PATH, get_connection, init_db
from memory.retrieval import (
    save_memory,
    get_recent_memories,
    get_memories_by_type,
    search_memories,
    search_and_rank_memories,
    save_fact,
    get_facts,
)
from memory.ranking import rank_memories

__all__ = [
    "DB_PATH",
    "init_db",
    "get_connection",
    "save_memory",
    "get_recent_memories",
    "get_memories_by_type",
    "search_memories",
    "search_and_rank_memories",
    "save_fact",
    "get_facts",
    "rank_memories",
]

if __name__ == "__main__":
    init_db()
    print(f"Memory database initialized at: {DB_PATH}")
