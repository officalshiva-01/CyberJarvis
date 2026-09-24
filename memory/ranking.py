"""
Advanced Ranking Algorithm for CyberJarvis Phase 7: Advanced Memory 2.0.
Ranks memories by:
1. Relevance (keyword overlap & substring match against query)
2. Importance (1 to 5 scale)
3. Recency (exponential / bucketed time decay)
4. Confidence (0.0 to 1.0 scale)
5. Memory-type match (bonus for matching requested domain)

Guarantees 100% backward compatibility with legacy rank_memories() tuple API.
"""
import re
import time
from datetime import datetime, timezone
from typing import List, Tuple, Any, Optional, Union

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "am", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those", "my", "your", "our",
    "what", "where", "when", "how", "why", "who", "which",
    "me", "you", "we", "they", "i", "do", "does", "did", "and", "or",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize and filter common stopwords for relevance scoring."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


def calculate_relevance(query: str, content: str) -> float:
    """Calculate normalized lexical relevance score in [0.0, 1.0]."""
    if not query or not content:
        return 0.0

    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0

    c_tokens = set(_tokenize(content))
    matches = sum(1 for t in q_tokens if t in c_tokens)
    overlap_ratio = matches / len(q_tokens)

    # Substring bonus if exact query phrase is inside content
    clean_q = query.lower().strip()
    clean_c = content.lower()
    sub_bonus = 0.25 if clean_q in clean_c else 0.0

    return min(1.0, overlap_ratio + sub_bonus)


def calculate_recency(timestamp_str: Optional[str]) -> float:
    """Calculate recency score in [0.2, 1.0] based on ISO/SQL timestamp."""
    if not timestamp_str:
        return 0.5

    try:
        # SQLite CURRENT_TIMESTAMP format is 'YYYY-MM-DD HH:MM:SS' or ISO 'YYYY-MM-DDTHH:MM:SS'
        clean_ts = timestamp_str.replace("T", " ").split(".")[0]
        dt = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S")
        age_seconds = max(0.0, time.time() - dt.replace(tzinfo=timezone.utc).timestamp())

        # Time decay buckets
        if age_seconds < 3600:       # < 1 hour
            return 1.0
        elif age_seconds < 86400:    # < 24 hours
            return 0.90
        elif age_seconds < 604800:   # < 7 days
            return 0.75
        elif age_seconds < 2592000:  # < 30 days
            return 0.50
        else:                        # older
            return 0.30
    except Exception:
        return 0.50


def score_memory(
    record: Any,
    query: Optional[str] = None,
    target_type: Optional[str] = None,
) -> float:
    """
    Compute multi-factor score for a MemoryRecord.
    Weights:
      With query:   Relevance: 0.35, Importance: 0.25, Recency: 0.15, Confidence: 0.15, Type Match: 0.10
      Without query: Importance: 0.35, Recency: 0.30, Confidence: 0.20, Type Match: 0.15
    """
    content = getattr(record, "content", "") or getattr(record, "user_message", "")
    importance = float(getattr(record, "importance", 3)) / 5.0  # normalize 1..5 -> 0.2..1.0
    confidence = float(getattr(record, "confidence", 1.0))      # 0.0..1.0
    recency = calculate_recency(getattr(record, "created_at", None))

    mem_type = getattr(record, "memory_type", "SESSION")
    type_match = 1.0 if (target_type and mem_type.upper() == target_type.upper()) else (0.5 if not target_type else 0.1)

    if query and query.strip():
        relevance = calculate_relevance(query, content)
        total = (
            0.35 * relevance +
            0.25 * importance +
            0.15 * recency +
            0.15 * confidence +
            0.10 * type_match
        )
    else:
        total = (
            0.35 * importance +
            0.30 * recency +
            0.20 * confidence +
            0.15 * type_match
        )

    return round(total, 4)


def rank_memory_records(
    records: List[Any],
    query: Optional[str] = None,
    target_type: Optional[str] = None,
) -> List[Any]:
    """Rank list of MemoryRecord objects using multi-factor scoring."""
    if not records:
        return []

    return sorted(
        records,
        key=lambda r: score_memory(r, query=query, target_type=target_type),
        reverse=True,
    )


def rank_memories(memories: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """
    Legacy backward-compatible tuple ranking function.
    Preserves exact behavior for (user_message, assistant_response, created_at, memory_type, importance).
    Higher importance is preferred.
    Newer memories are preferred when importance is equal.
    """
    return sorted(
        memories,
        key=lambda memory: (
            memory[4] if len(memory) > 4 else 3,
            memory[2] if len(memory) > 2 else "",
        ),
        reverse=True,
    )
