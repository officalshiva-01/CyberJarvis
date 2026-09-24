"""
Advanced Memory Retrieval Engine for CyberJarvis Phase 7: Advanced Memory 2.0.
Supports:
- Exact retrieval (exact content / phrase matches)
- Keyword retrieval (tokenized content & metadata matching)
- Typed retrieval (USER_PROFILE, PROJECT, TASK, DECISION, ISSUE, SESSION, TOOL_RESULT)
- Project-scoped retrieval
- Task-scoped retrieval
- Multi-factor ranking integration
- 100% backward compatibility with legacy facts and memory tuple APIs.
"""
from typing import List, Tuple, Optional, Any, Union
import sqlite3

from memory.store import get_connection
from memory.models import MemoryRecord, MemoryType, MemoryStatus, Confidence
from memory.ranking import rank_memory_records, rank_memories
from memory.sanitizer import sanitize_memory_content
from observability.logging import get_logger

logger = get_logger("CyberJarvis.MemoryRetrieval")


def retrieve_memories(
    query: Optional[str] = None,
    memory_type: Optional[Union[str, List[str]]] = None,
    project: Optional[str] = None,
    task_id: Optional[str] = None,
    status: Optional[str] = MemoryStatus.ACTIVE.value,
    exact_match: bool = False,
    min_confidence: float = 0.0,
    min_importance: int = 1,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> List[MemoryRecord]:
    """
    Unified high-performance memory retrieval engine (<50ms target, zero Ollama overhead).
    Supports exact, keyword, typed, project-scoped, and task-scoped retrieval.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    conditions: List[str] = []
    params: List[Any] = []

    # 1. Status Filter
    if status is not None:
        conditions.append("status = ?")
        params.append(status.lower().strip())

    # 2. Type Filter (supports single type or list of types)
    if memory_type:
        if isinstance(memory_type, (list, tuple, set)):
            norm_types = [MemoryType.normalize(t) for t in memory_type]
            placeholders = ", ".join(["?"] * len(norm_types))
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(norm_types)
        else:
            norm_t = MemoryType.normalize(memory_type)
            conditions.append("memory_type = ?")
            params.append(norm_t)

    # 3. Project Scope
    if project:
        conditions.append("project = ?")
        params.append(project)

    # 4. Task Scope
    if task_id:
        conditions.append("task_id = ?")
        params.append(task_id)

    # 5. Importance & Confidence Bounds
    if min_importance > 1:
        conditions.append("importance >= ?")
        params.append(min_importance)
    if min_confidence > 0.0:
        conditions.append("confidence >= ?")
        params.append(min_confidence)

    # 6. Query Filter: Exact vs Keyword
    if query and query.strip():
        clean_q = query.strip()
        if exact_match:
            conditions.append("(content = ? OR user_message = ?)")
            params.extend([clean_q, clean_q])
        else:
            # Keyword matching over content, user_message, assistant_response, and metadata
            stop_words = {
                "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from",
                "by", "for", "with", "about", "against", "between", "into", "through", "during",
                "before", "after", "above", "below", "to", "in", "on", "off", "over", "under",
                "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
                "does", "did", "can", "could", "should", "would", "will", "shall", "may", "might",
                "must", "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he",
                "him", "his", "she", "her", "it", "its", "they", "them", "their", "what", "which",
                "who", "whom", "this", "that", "these", "those", "how", "why", "where"
            }
            raw_tokens = [w.strip() for w in clean_q.split() if len(w.strip()) > 1]
            meaningful = [w for w in raw_tokens if w.lower() not in stop_words]
            words = meaningful if meaningful else raw_tokens
            if words:
                keyword_clauses = []
                for w in words[:8]:  # Bounded to top 8 tokens for sub-50ms performance
                    keyword_clauses.append(
                        "(content LIKE ? OR user_message LIKE ? OR assistant_response LIKE ? OR metadata LIKE ?)"
                    )
                    like_pattern = f"%{w}%"
                    params.extend([like_pattern, like_pattern, like_pattern, like_pattern])
                conditions.append(f"({' OR '.join(keyword_clauses)})")
            else:
                conditions.append("(content LIKE ? OR user_message LIKE ?)")
                params.extend([f"%{clean_q}%", f"%{clean_q}%"])

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    # Fetch candidate pool (up to 3x limit) to allow multi-factor ranking
    fetch_limit = max(limit * 3, 30)
    sql = f"SELECT * FROM memories {where_sql} ORDER BY id DESC LIMIT ?"
    params.append(fetch_limit)

    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    conn.close()

    records = [MemoryRecord.from_row(r, cols) for r in rows]

    # Target type for ranking
    target_t = None
    if memory_type and isinstance(memory_type, str):
        target_t = MemoryType.normalize(memory_type)

    # Rank and slice to limit
    ranked = rank_memory_records(records, query=query, target_type=target_t)
    return ranked[:limit]


# =====================================================================
# Backward-Compatible APIs
# =====================================================================

def save_memory(
    user_message: str,
    assistant_response: str,
    memory_type: str = "conversation",
    importance: int = 3,
    db_path: Optional[str] = None,
) -> int:
    """Legacy API for saving conversation memories."""
    from memory.lifecycle import create_memory
    norm_type = MemoryType.normalize(memory_type)
    content = sanitize_memory_content(user_message.strip())
    rec = create_memory(
        content=content,
        memory_type=norm_type,
        importance=importance,
        confidence=Confidence.HIGH,
        source="user",
        user_message=user_message,
        assistant_response=assistant_response,
        db_path=db_path,
    )
    return rec.id


def get_recent_memories(limit: int = 10, db_path: Optional[str] = None) -> List[Tuple[str, str, str, str, int]]:
    """Legacy API returning tuples: (user_message, assistant_response, created_at, memory_type, importance)."""
    records = retrieve_memories(limit=limit, db_path=db_path)
    return [r.to_legacy_tuple() for r in records]


def get_memories_by_type(
    memory_type: str, limit: int = 10, db_path: Optional[str] = None
) -> List[Tuple[str, str, str, str, int]]:
    """Legacy API returning typed memories as tuples."""
    records = retrieve_memories(memory_type=memory_type, limit=limit, db_path=db_path)
    return [r.to_legacy_tuple() for r in records]


def search_memories(
    query: str, limit: int = 10, memory_type: Optional[str] = None, db_path: Optional[str] = None
) -> List[Tuple[str, str, str, str, int]]:
    """Legacy API for searching memories returning tuples."""
    if not query or not query.strip():
        return []
    records = retrieve_memories(query=query, memory_type=memory_type, limit=limit, db_path=db_path)
    return [r.to_legacy_tuple() for r in records]


def search_and_rank_memories(
    query: str, limit: int = 10, memory_type: Optional[str] = None, db_path: Optional[str] = None
) -> List[Tuple[str, str, str, str, int]]:
    """Legacy API returning ranked tuples."""
    tuples = search_memories(query, limit=limit, memory_type=memory_type, db_path=db_path)
    return rank_memories(tuples)


def save_fact(fact: str, db_path: Optional[str] = None):
    """
    Legacy API for storing facts.
    Saves to facts table AND creates a high-confidence USER_PROFILE memory.
    """
    clean_fact = sanitize_memory_content(fact.strip())
    if not clean_fact:
        return

    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO facts (fact) VALUES (?)",
        (clean_fact,),
    )
    fid = cur.lastrowid
    conn.commit()
    conn.close()

    # Also persist as USER_PROFILE memory
    from memory.lifecycle import create_memory
    existing = retrieve_memories(
        query=clean_fact,
        memory_type=MemoryType.USER_PROFILE.value,
        exact_match=True,
        db_path=db_path,
    )
    if not existing:
        rec = create_memory(
            content=clean_fact,
            memory_type=MemoryType.USER_PROFILE.value,
            importance=5,
            confidence=Confidence.HIGH,
            source="user",
            metadata={"key": "fact", "legacy_synced": True},
            db_path=db_path,
        )
        if not fid:
            fid = rec.id

    return fid


def get_facts(limit: int = 20, db_path: Optional[str] = None) -> List[str]:
    """Legacy API returning facts list."""
    conn = get_connection(db_path)
    cursor = conn.execute(
        "SELECT fact FROM facts ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    facts = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Also include high-importance USER_PROFILE memories if not already in facts
    user_memories = retrieve_memories(
        memory_type=MemoryType.USER_PROFILE.value,
        min_importance=4,
        limit=limit,
        db_path=db_path,
    )
    seen = {f.lower().strip() for f in facts}
    for m in user_memories:
        clean_c = m.content.strip()
        if clean_c.lower() not in seen:
            facts.append(clean_c)
            seen.add(clean_c.lower())

    return facts[:limit]
