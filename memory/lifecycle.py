"""
Memory Lifecycle Manager for CyberJarvis Phase 7: Advanced Memory 2.0.
Implements:
- create (with secret sanitization and confidence assignment)
- update
- supersede
- archive
- delete (with confirmation guard for importance >= 4)
- merge duplicates
"""
import json
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List, Union

from memory.store import get_connection
from memory.models import MemoryRecord, MemoryType, MemoryStatus, Confidence
from memory.sanitizer import sanitize_memory_content, sanitize_memory_metadata
from observability.logging import get_logger

logger = get_logger("CyberJarvis.MemoryLifecycle")


def create_memory(
    content: str,
    memory_type: Union[MemoryType, str] = MemoryType.SESSION,
    importance: int = 3,
    confidence: float = Confidence.HIGH,
    source: str = "user",
    project: Optional[str] = "CyberJarvis",
    task_id: Optional[str] = None,
    status: str = "active",
    metadata: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
    db_path: Optional[str] = None,
) -> MemoryRecord:
    """Create and persist a new sanitized MemoryRecord to SQLite."""
    clean_content = sanitize_memory_content(content.strip() if content else "")
    if not clean_content and user_message:
        clean_content = sanitize_memory_content(user_message.strip())

    if not clean_content:
        raise ValueError("Cannot create memory with empty content.")

    norm_type = MemoryType.normalize(str(memory_type))
    clean_meta = sanitize_memory_metadata(metadata or {})
    meta_json = json.dumps(clean_meta)

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO memories (
            user_message, assistant_response, content, memory_type,
            importance, confidence, created_at, updated_at, source,
            project, task_id, status, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """,
        (
            user_message or clean_content,
            assistant_response or "",
            clean_content,
            norm_type,
            importance,
            confidence,
            source,
            project or "CyberJarvis",
            task_id,
            status.lower().strip() if status else "active",
            meta_json,
        ),
    )
    new_id = cursor.lastrowid
    conn.commit()

    # Retrieve created record
    cursor.execute("SELECT * FROM memories WHERE id = ?", (new_id,))
    row = cursor.fetchone()
    cols = [d[0] for d in cursor.description]
    conn.close()

    record = MemoryRecord.from_row(row, cols)
    logger.info(f"Created memory ID {record.id} [{record.memory_type}]: '{record.content[:60]}...'")
    return record


def get_memory_by_id(memory_id: int, db_path: Optional[str] = None) -> Optional[MemoryRecord]:
    """Retrieve single MemoryRecord by ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    cols = [d[0] for d in cursor.description]
    conn.close()
    return MemoryRecord.from_row(row, cols)


def update_memory(
    memory_id: int,
    content: Optional[str] = None,
    importance: Optional[int] = None,
    confidence: Optional[float] = None,
    status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    task_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[MemoryRecord]:
    """Update fields of an existing memory record."""
    current = get_memory_by_id(memory_id, db_path=db_path)
    if not current:
        return None

    updates = []
    params = []

    if content is not None:
        sanitized = sanitize_memory_content(content.strip())
        updates.append("content = ?")
        params.append(sanitized)
        updates.append("user_message = ?")
        params.append(sanitized)

    if importance is not None:
        updates.append("importance = ?")
        params.append(max(1, min(5, int(importance))))

    if confidence is not None:
        updates.append("confidence = ?")
        params.append(max(0.0, min(1.0, float(confidence))))

    if status is not None:
        updates.append("status = ?")
        params.append(status.lower().strip())

    if project is not None:
        updates.append("project = ?")
        params.append(project)

    if task_id is not None:
        updates.append("task_id = ?")
        params.append(task_id)

    if metadata is not None:
        merged_meta = dict(current.metadata)
        merged_meta.update(sanitize_memory_metadata(metadata))
        updates.append("metadata = ?")
        params.append(json.dumps(merged_meta))

    if not updates:
        return current

    updates.append("updated_at = CURRENT_TIMESTAMP")
    params.append(memory_id)

    conn = get_connection(db_path)
    cursor = conn.cursor()
    sql = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
    cursor.execute(sql, tuple(params))
    conn.commit()
    conn.close()

    updated = get_memory_by_id(memory_id, db_path=db_path)
    logger.info(f"Updated memory ID {memory_id}")
    return updated


class SupersedeResult(tuple):
    """2-tuple (old_record, new_record) where attributes on result delegate to new_record."""
    def __new__(cls, old_record, new_record):
        return super().__new__(cls, (old_record, new_record))

    @property
    def old_record(self):
        return self[0]

    @property
    def new_record(self):
        return self[1]

    def __getattr__(self, name):
        return getattr(self[1], name)


def supersede_memory(
    old_id: Optional[int] = None,
    new_content: str = "",
    memory_type: Optional[str] = None,
    importance: Optional[int] = None,
    confidence: Optional[float] = None,
    source: str = "user",
    project: Optional[str] = None,
    task_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
    old_memory_id: Optional[int] = None,
    new_importance: Optional[int] = None,
) -> SupersedeResult:
    """
    Mark old memory as superseded and insert new updated memory.
    Links the new memory to the superseded record.
    Returns SupersedeResult(old_record, new_record).
    """
    target_old_id = old_id if old_id is not None else old_memory_id
    if target_old_id is None:
        raise ValueError("Must provide old_id or old_memory_id to supersede.")

    target_importance = importance if importance is not None else new_importance

    old_record = get_memory_by_id(target_old_id, db_path=db_path)
    old_type = old_record.memory_type if old_record else MemoryType.SESSION.value
    old_proj = old_record.project if old_record else "CyberJarvis"
    old_imp = old_record.importance if old_record else 3
    old_conf = old_record.confidence if old_record else 1.0

    meta = metadata or {}
    meta["supersedes_id"] = target_old_id

    # 1. Create new memory
    new_record = create_memory(
        content=new_content,
        memory_type=memory_type or old_type,
        importance=target_importance if target_importance is not None else old_imp,
        confidence=confidence if confidence is not None else old_conf,
        source=source,
        project=project or old_proj,
        task_id=task_id,
        metadata=meta,
        db_path=db_path,
    )

    # 2. Mark old memory as superseded
    if old_record:
        old_meta = dict(old_record.metadata)
        old_meta["superseded_by"] = new_record.id
        update_memory(
            target_old_id,
            status=MemoryStatus.SUPERSEDED.value,
            metadata=old_meta,
            db_path=db_path,
        )
        old_record = get_memory_by_id(target_old_id, db_path=db_path)

    logger.info(f"Superseded memory ID {target_old_id} -> New memory ID {new_record.id}")
    return SupersedeResult(old_record, new_record)


def archive_memory(memory_id: int, db_path: Optional[str] = None) -> bool:
    """Archive an existing memory record."""
    updated = update_memory(memory_id, status=MemoryStatus.ARCHIVED.value, db_path=db_path)
    return updated is not None


def delete_memory(
    memory_id: int,
    confirmed: bool = False,
    hard_delete: bool = False,
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Delete a memory record with confirmation guard for important memories.
    Rule: Important memories (importance >= 4) must NOT be silently deleted.
    """
    record = get_memory_by_id(memory_id, db_path=db_path)
    if not record:
        return False, f"Memory with ID {memory_id} does not exist."

    # Guard: Deletion of important memories requires confirmation
    if record.importance >= 4 and not confirmed:
        logger.warning(f"Prevented silent deletion of important memory ID {memory_id} (importance={record.importance})")
        return (
            False,
            f"Memory '{record.content[:60]}' has high importance ({record.importance}/5). Deletion requires confirmation.",
        )

    conn = get_connection(db_path)
    cursor = conn.cursor()
    if hard_delete:
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    else:
        cursor.execute(
            "UPDATE memories SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (MemoryStatus.DELETED.value, memory_id),
        )
    conn.commit()
    conn.close()

    logger.info(f"Deleted memory ID {memory_id} (confirmed={confirmed}, hard={hard_delete})")
    return True, f"Memory ID {memory_id} successfully deleted."


def merge_duplicate_memories(
    memory_type: Optional[str] = None,
    project: Optional[str] = "CyberJarvis",
    db_path: Optional[str] = None,
) -> int:
    """
    Identify and merge duplicate or near-identical active memories.
    Keeps the highest importance/confidence record, merges metadata, and marks duplicates superseded.
    Returns the number of merged duplicates.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM memories WHERE status = 'active'"
    params: List[Any] = []
    if memory_type:
        query += " AND memory_type = ?"
        params.append(MemoryType.normalize(memory_type))
    if project:
        query += " AND project = ?"
        params.append(project)

    query += " ORDER BY id ASC"
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    conn.close()

    records = [MemoryRecord.from_row(r, cols) for r in rows]
    if not records:
        return 0

    def _normalize_key(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    seen: Dict[str, MemoryRecord] = {}
    merged_count = 0

    for rec in records:
        key = f"{rec.memory_type}:{_normalize_key(rec.content)}"
        if key in seen:
            primary = seen[key]
            # If the current record has higher importance or confidence, update primary
            new_imp = max(primary.importance, rec.importance)
            new_conf = max(primary.confidence, rec.confidence)
            merged_meta = dict(primary.metadata)
            merged_meta.update(rec.metadata)
            merged_meta.setdefault("merged_ids", []).append(rec.id)

            update_memory(
                primary.id,
                importance=new_imp,
                confidence=new_conf,
                metadata=merged_meta,
                db_path=db_path,
            )

            # Mark duplicate as superseded
            dup_meta = dict(rec.metadata)
            dup_meta["merged_into"] = primary.id
            update_memory(
                rec.id,
                status=MemoryStatus.SUPERSEDED.value,
                metadata=dup_meta,
                db_path=db_path,
            )
            merged_count += 1
        else:
            seen[key] = rec

    logger.info(f"Duplicate merge completed. Merged {merged_count} memories.")
    return merged_count
