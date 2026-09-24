"""
Deterministic Memory Extraction Engine for CyberJarvis Phase 7: Advanced Memory 2.0.
Extracts structured, typed memories directly from natural language statements:
- USER_PROFILE: Identity, career goals, personal preferences
- PROJECT: Core stack details, milestone status, architecture properties
- DECISION: Explicit architectural and implementation choices
- ISSUE: Latency metrics, known bugs, performance bottlenecks
- TASK: Active ongoing tasks and objectives
"""
import re
from typing import Optional, Tuple
from memory.models import MemoryRecord, MemoryType, Confidence
from memory.sanitizer import sanitize_memory_content
from observability.logging import get_logger

logger = get_logger("CyberJarvis.MemoryExtraction")


def extract_memory_from_text(text: str, source: str = "user") -> Optional[MemoryRecord]:
    """
    Deterministic rule-based memory extraction without LLM overhead.
    Returns MemoryRecord if pattern matches, else None.
    """
    if not text:
        return None

    clean = text.strip()
    clean_lower = clean.lower()

    # Determine confidence based on source (Requirement 7)
    base_confidence = Confidence.HIGH if source == "user" else (Confidence.TOOL if "tool" in source else Confidence.INFERRED)

    # 1. USER_PROFILE: Explicit Name Statements
    name_m = re.match(r"^(?:my name is|i am|i'm)\s+([A-Za-z0-9_\-]+(?:\s+[A-Za-z0-9_\-]+)?)\.?$", clean, re.IGNORECASE)
    if name_m and "goal" not in clean_lower and "building" not in clean_lower and "working" not in clean_lower:
        name_val = name_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.USER_PROFILE.value,
            content=f"My name is {name_val}",
            importance=5,
            confidence=base_confidence,
            source=source,
            metadata={"key": "user_name", "name": name_val},
        )

    # 2. USER_PROFILE: Explicit Goal Statements
    goal_m = re.match(r"^(?:my goal is|i want to become|i plan to)\s+(.+?)\.?$", clean, re.IGNORECASE)
    if goal_m:
        goal_val = goal_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.USER_PROFILE.value,
            content=f"My goal is {goal_val}",
            importance=5,
            confidence=base_confidence,
            source=source,
            metadata={"key": "user_goal", "goal": goal_val},
        )

    # 3. USER_PROFILE: General "remember this" or preference
    rem_m = re.match(r"^(?:remember this[:\s]+|remember that\s+)(.+)$", clean, re.IGNORECASE)
    if rem_m:
        extracted = rem_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.USER_PROFILE.value,
            content=extracted,
            importance=4,
            confidence=base_confidence,
            source=source,
            metadata={"key": "user_note", "explicit_remember": True},
        )

    # 4. DECISION: "We replaced X with Y", "We decided to X"
    replace_m = re.match(r"^(?:we replaced\s+(.+?)\s+with\s+(.+?))\.?$", clean, re.IGNORECASE)
    if replace_m:
        old_comp = replace_m.group(1).strip()
        new_comp = replace_m.group(2).strip()
        return MemoryRecord(
            memory_type=MemoryType.DECISION.value,
            content=f"We replaced {old_comp} with {new_comp}",
            importance=5,
            confidence=base_confidence,
            source=source,
            metadata={"key": "decision", "replaced": old_comp, "replacement": new_comp},
        )

    dec_m = re.match(r"^(?:we decided to|decision:\s*|we chose to)\s*(.+?)\.?$", clean, re.IGNORECASE)
    if dec_m:
        dec_val = dec_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.DECISION.value,
            content=f"Decision: {dec_val}",
            importance=5,
            confidence=base_confidence,
            source=source,
            metadata={"key": "decision"},
        )

    # 5. ISSUE / PERFORMANCE: "STT latency was 17–27 seconds"
    lat_m = re.match(r"^([a-zA-Z0-9_\-\s]+?)\s+(?:latency|response time|delay)\s+(?:was|is)\s+([0-9\.\–\-\s]+(?:s|seconds|ms))\.?$", clean, re.IGNORECASE)
    if lat_m:
        metric = lat_m.group(1).strip()
        val = lat_m.group(2).strip()
        return MemoryRecord(
            memory_type=MemoryType.ISSUE.value,
            content=f"{metric} latency was {val}",
            importance=4,
            confidence=Confidence.TOOL if source != "user" else base_confidence,
            source=source,
            metadata={"key": "performance_issue", "metric": metric, "value": val},
        )

    # 6. PROJECT / STATUS: "Phase 3 is complete"
    phase_m = re.match(r"^phase\s+(\d+)\s+is\s+(complete|completed|in progress|verified|active)\.?$", clean, re.IGNORECASE)
    if phase_m:
        p_num = phase_m.group(1).strip()
        p_st = phase_m.group(2).strip()
        return MemoryRecord(
            memory_type=MemoryType.PROJECT.value,
            content=f"Phase {p_num} is {p_st}",
            importance=5,
            confidence=base_confidence,
            source=source,
            metadata={"key": "project_phase", "phase": int(p_num), "status": p_st},
        )

    # 7. PROJECT: "CyberJarvis uses faster-whisper"
    use_m = re.match(r"^(?:cyberjarvis|the project)\s+uses\s+(.+?)\.?$", clean, re.IGNORECASE)
    if use_m:
        comp = use_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.PROJECT.value,
            content=f"CyberJarvis uses {comp}",
            importance=4,
            confidence=base_confidence,
            source=source,
            metadata={"key": "project_stack", "component": comp},
        )

    # 8. TASK: "We are working on...", "Current task is..."
    task_m = re.match(r"^(?:we are working on|working on|current task is|task:\s*)(.+?)\.?$", clean, re.IGNORECASE)
    if task_m:
        task_desc = task_m.group(1).strip()
        return MemoryRecord(
            memory_type=MemoryType.TASK.value,
            content=f"Active Task: {task_desc}",
            importance=4,
            confidence=base_confidence,
            source=source,
            metadata={"key": "active_task", "task": task_desc},
        )

    return None


def auto_extract_and_save_memory(
    text: str, source: str = "user", db_path: Optional[str] = None
) -> Optional[MemoryRecord]:
    """Extract memory from text and persist it directly if matched."""
    record = extract_memory_from_text(text, source=source)
    if not record:
        return None

    from memory.lifecycle import create_memory
    from memory.retrieval import save_fact

    # Persist as durable memory
    created = create_memory(
        content=record.content,
        memory_type=record.memory_type,
        importance=record.importance,
        confidence=record.confidence,
        source=source,
        metadata=record.metadata,
        db_path=db_path,
    )

    # If it is a user identity/goal fact, sync to legacy facts table
    if record.memory_type == MemoryType.USER_PROFILE.value and record.importance >= 5:
        save_fact(record.content, db_path=db_path)

    logger.info(f"Auto-extracted and persisted memory [{created.memory_type}]: '{created.content}'")
    return created
