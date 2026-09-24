"""
Unit and integration tests for CyberJarvis Phase 7: Advanced Memory 2.0.
Verifies all 20 requirements:
1. Memory types (USER_PROFILE, PROJECT, TASK, DECISION, ISSUE, SESSION, TOOL_RESULT)
2. Project memory
3. Task memory
4. Decision memory
5. Issue memory
6. Confidence handling
7. Importance handling
8. Recency scoring
9. Relevance scoring
10. Typed retrieval
11. Project scoping
12. Task scoping
13. Duplicate merging
14. Superseding
15. Archival
16. Deletion confirmation guard (importance >= 4)
17. Secret sanitization
18. Context builder limit (<= 2500 chars)
19. Continuation queries
20. Generic information memory isolation
"""
import os
import tempfile
import pytest
from pathlib import Path

from memory.models import MemoryRecord, MemoryType, MemoryStatus, Confidence
from memory.sanitizer import sanitize_memory_content, sanitize_memory_metadata
from memory.store import init_db, get_connection
from memory.lifecycle import (
    create_memory,
    get_memory_by_id,
    update_memory,
    supersede_memory,
    archive_memory,
    delete_memory,
    merge_duplicate_memories,
)
from memory.retrieval import (
    retrieve_memories,
    save_memory,
    get_recent_memories,
    save_fact,
    get_facts,
    search_memories,
    search_and_rank_memories,
)
from memory.ranking import (
    calculate_relevance,
    calculate_recency,
    score_memory,
    rank_memory_records,
)
from memory.extraction import extract_memory_from_text, auto_extract_and_save_memory
from memory.context_builder import MemoryContextBuilder
from agent.router import ExecutionRouter, RouteType
from agent.fast_path import FastPathHandler


@pytest.fixture
def temp_db():
    """Create an isolated temporary SQLite database for test execution."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


class TestAdvancedMemory20:

    # 1. Memory Types
    def test_memory_types_support(self, temp_db):
        for m_type in [
            MemoryType.USER_PROFILE,
            MemoryType.PROJECT,
            MemoryType.TASK,
            MemoryType.DECISION,
            MemoryType.ISSUE,
            MemoryType.SESSION,
            MemoryType.TOOL_RESULT,
        ]:
            rec = create_memory(
                content=f"Test content for {m_type.value}",
                memory_type=m_type.value,
                db_path=temp_db,
            )
            assert rec.id is not None
            assert rec.memory_type == m_type.value
            fetched = get_memory_by_id(rec.id, db_path=temp_db)
            assert fetched is not None
            assert fetched.memory_type == m_type.value

    # 2. Project Memory
    def test_project_memory_persistence(self, temp_db):
        rec = create_memory(
            content="CyberJarvis is an autonomous local voice assistant",
            memory_type=MemoryType.PROJECT.value,
            project="CyberJarvis",
            importance=5,
            confidence=Confidence.EXPLICIT_USER,
            db_path=temp_db,
        )
        assert rec.id is not None
        results = retrieve_memories(
            query="autonomous voice assistant",
            memory_type=MemoryType.PROJECT.value,
            project="CyberJarvis",
            db_path=temp_db,
        )
        assert len(results) > 0
        assert any("autonomous local voice assistant" in r.content for r in results)

    # 3. Task Memory
    def test_task_memory_persistence_and_scoping(self, temp_db):
        rec = create_memory(
            content="Refactor audio capture stream for ALSA buffer alignment",
            memory_type=MemoryType.TASK.value,
            task_id="task-capture-101",
            status=MemoryStatus.ACTIVE.value,
            db_path=temp_db,
        )
        assert rec.task_id == "task-capture-101"
        scoped = retrieve_memories(task_id="task-capture-101", db_path=temp_db)
        assert len(scoped) == 1
        assert scoped[0].content == "Refactor audio capture stream for ALSA buffer alignment"

    # 4. Decision Memory
    def test_decision_memory_retrieval(self, temp_db):
        create_memory(
            content="Decided to use faster-whisper INT8 instead of Google Speech API",
            memory_type=MemoryType.DECISION.value,
            importance=5,
            db_path=temp_db,
        )
        decisions = retrieve_memories(
            query="faster-whisper Google STT",
            memory_type=MemoryType.DECISION.value,
            db_path=temp_db,
        )
        assert len(decisions) >= 1
        assert "faster-whisper" in decisions[0].content

    # 5. Issue Memory
    def test_issue_memory_retrieval(self, temp_db):
        create_memory(
            content="Audio capture ALSA buffer underrun observed during heavy CPU load",
            memory_type=MemoryType.ISSUE.value,
            importance=4,
            db_path=temp_db,
        )
        issues = retrieve_memories(
            query="ALSA buffer underrun",
            memory_type=MemoryType.ISSUE.value,
            db_path=temp_db,
        )
        assert len(issues) >= 1
        assert "ALSA buffer underrun" in issues[0].content

    # 6. Confidence Scoring
    def test_confidence_scoring(self, temp_db):
        m1 = create_memory(
            content="User said: I work as a Cloud Architect",
            memory_type=MemoryType.USER_PROFILE.value,
            confidence=Confidence.EXPLICIT_USER,  # 1.0
            importance=3,
            db_path=temp_db,
        )
        m2 = create_memory(
            content="LLM assumption: User may prefer AWS over GCP",
            memory_type=MemoryType.USER_PROFILE.value,
            confidence=Confidence.INFERRED,  # 0.5
            importance=3,
            db_path=temp_db,
        )
        assert m1.confidence == 1.0
        assert m2.confidence == 0.5
        ranked = retrieve_memories(
            query="Cloud Architect AWS",
            memory_type=MemoryType.USER_PROFILE.value,
            db_path=temp_db,
        )
        # Higher confidence should be favored
        assert ranked[0].confidence >= ranked[-1].confidence

    # 7. Importance Ranking
    def test_importance_ranking(self, temp_db):
        low = create_memory(content="User mentioned liking tea", importance=1, db_path=temp_db)
        high = create_memory(content="User primary programming language is Python", importance=5, db_path=temp_db)
        ranked = retrieve_memories(query="user", db_path=temp_db)
        high_idx = next(i for i, r in enumerate(ranked) if r.id == high.id)
        low_idx = next(i for i, r in enumerate(ranked) if r.id == low.id)
        assert high_idx < low_idx

    # 8. Recency Scoring
    def test_recency_scoring(self):
        now_ts = "2026-09-24T18:00:00"
        recent_ts = "2026-09-24T17:00:00"
        old_ts = "2025-01-01T00:00:00"
        score_recent = calculate_recency(recent_ts)
        score_old = calculate_recency(old_ts)
        assert score_recent > score_old
        assert 0.0 <= score_recent <= 1.0

    # 9. Relevance Calculation
    def test_relevance_calculation(self):
        rel1 = calculate_relevance("docker kubernetes deployment", "Deploying docker container on kubernetes cluster")
        rel2 = calculate_relevance("docker kubernetes deployment", "Making a cup of coffee")
        assert rel1 > rel2
        assert rel1 > 0.5
        assert rel2 == 0.0

    # 10. Typed Retrieval
    def test_typed_retrieval(self, temp_db):
        create_memory(content="Task 1", memory_type=MemoryType.TASK.value, db_path=temp_db)
        create_memory(content="Decision 1", memory_type=MemoryType.DECISION.value, db_path=temp_db)
        create_memory(content="Issue 1", memory_type=MemoryType.ISSUE.value, db_path=temp_db)

        tasks = retrieve_memories(memory_type=MemoryType.TASK.value, db_path=temp_db)
        assert all(t.memory_type == MemoryType.TASK.value for t in tasks)
        assert any(t.content == "Task 1" for t in tasks)

        multi = retrieve_memories(
            memory_type=[MemoryType.TASK.value, MemoryType.DECISION.value],
            db_path=temp_db,
        )
        assert all(m.memory_type in (MemoryType.TASK.value, MemoryType.DECISION.value) for m in multi)
        assert any(m.content == "Decision 1" for m in multi)

    # 11. Project Scoping
    def test_project_scoping(self, temp_db):
        create_memory(content="CyberJarvis fact", project="CyberJarvis", db_path=temp_db)
        create_memory(content="OtherProject fact", project="OtherProject", db_path=temp_db)

        jarvis_mem = retrieve_memories(project="CyberJarvis", db_path=temp_db)
        assert all(m.project == "CyberJarvis" for m in jarvis_mem)
        assert any(m.content == "CyberJarvis fact" for m in jarvis_mem)
        assert not any(m.content == "OtherProject fact" for m in jarvis_mem)

    # 12. Task Scoping
    def test_task_scoping(self, temp_db):
        create_memory(content="Step 1 for task 42", task_id="task-42", db_path=temp_db)
        create_memory(content="Step 1 for task 99", task_id="task-99", db_path=temp_db)

        res = retrieve_memories(task_id="task-42", db_path=temp_db)
        assert len(res) == 1
        assert res[0].content == "Step 1 for task 42"

    # 13. Duplicate Merging
    def test_duplicate_merging(self, temp_db):
        m1 = create_memory(
            content="User prefers dark mode UI theme",
            importance=2,
            confidence=0.8,
            metadata={"source": "cli"},
            db_path=temp_db,
        )
        m2 = create_memory(
            content="User prefers dark mode UI theme",
            importance=4,
            confidence=1.0,
            metadata={"extra": "verified"},
            db_path=temp_db,
        )
        merged_count = merge_duplicate_memories(db_path=temp_db)
        assert merged_count >= 1

        # The duplicate memory must be superseded and primary kept
        active = retrieve_memories(query="dark mode UI theme", db_path=temp_db)
        dark_memories = [m for m in active if "dark mode UI theme" in m.content]
        assert len(dark_memories) == 1
        assert dark_memories[0].importance == 4
        assert dark_memories[0].confidence == 1.0
        assert dark_memories[0].metadata.get("extra") == "verified"

    # 14. Superseding Memories
    def test_superseding_memories(self, temp_db):
        old_mem = create_memory(
            content="Backend STT uses Google Cloud Speech API",
            memory_type=MemoryType.DECISION.value,
            importance=3,
            db_path=temp_db,
        )
        new_mem = supersede_memory(
            old_memory_id=old_mem.id,
            new_content="Backend STT uses local faster-whisper offline",
            new_importance=4,
            db_path=temp_db,
        )
        assert new_mem is not None
        assert new_mem.content == "Backend STT uses local faster-whisper offline"
        assert new_mem.status == MemoryStatus.ACTIVE.value

        old_recheck = get_memory_by_id(old_mem.id, db_path=temp_db)
        assert old_recheck.status == MemoryStatus.SUPERSEDED.value

        # Only active memories are returned in normal retrieval
        active_list = retrieve_memories(query="STT", db_path=temp_db)
        assert any("local faster-whisper" in m.content for m in active_list)
        assert not any("Google Cloud Speech API" in m.content for m in active_list)

    # 15. Archival
    def test_archival(self, temp_db):
        mem = create_memory(content="Unique temporary milestone note", db_path=temp_db)
        archived = archive_memory(mem.id, db_path=temp_db)
        assert archived is True

        check = get_memory_by_id(mem.id, db_path=temp_db)
        assert check.status == MemoryStatus.ARCHIVED.value

        active_list = retrieve_memories(query="Unique temporary milestone note", db_path=temp_db)
        assert not any(m.id == mem.id for m in active_list)

    # 16. Deletion Confirmation Guard
    def test_deletion_confirmation_guard(self, temp_db):
        # Low importance memory (importance < 4) can be deleted without confirmation
        low_mem = create_memory(content="Transient note", importance=2, db_path=temp_db)
        ok, msg = delete_memory(low_mem.id, confirmed=False, db_path=temp_db)
        assert ok is True

        # High importance memory (importance >= 4) CANNOT be deleted without confirmation
        high_mem = create_memory(content="User full legal name", importance=5, db_path=temp_db)
        ok, msg = delete_memory(high_mem.id, confirmed=False, db_path=temp_db)
        assert ok is False
        assert "confirmation" in msg.lower()

        # With confirmation, deletion succeeds
        ok, msg = delete_memory(high_mem.id, confirmed=True, db_path=temp_db)
        assert ok is True
        assert get_memory_by_id(high_mem.id, db_path=temp_db).status == MemoryStatus.DELETED.value

    # 17. Secret Protection / Sanitization
    def test_secret_sanitization(self, temp_db):
        dirty_inputs = [
            "My secret password is Password123! and token is ghp_1234567890abcdef1234567890abcdef1234",
            "Authorization: Bearer secret_jwt_token_here_with_enough_length_to_trigger",
            "AWS credentials: AKIAIOSFODNN7EXAMPLE and secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "OpenAI API key: sk-live1234567890abcdef1234567890abcdef12",
        ]
        for text in dirty_inputs:
            sanitized = sanitize_memory_content(text)
            assert "Password123!" not in sanitized
            assert "ghp_1234567890abcdef1234567890abcdef1234" not in sanitized
            assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
            assert "sk-live" not in sanitized

            # Save to memory and verify persisted row is clean
            rec = create_memory(content=text, db_path=temp_db)
            fetched = get_memory_by_id(rec.id, db_path=temp_db)
            assert "Password123!" not in fetched.content
            assert "ghp_1234567890abcdef1234567890abcdef1234" not in fetched.content
            assert "AKIAIOSFODNN7EXAMPLE" not in fetched.content

    # 18. MemoryContextBuilder Context Limit <= 2500 Characters
    def test_context_bounded_2500_chars(self, temp_db):
        # Seed 30 memories to test bounding
        for i in range(30):
            create_memory(
                content=f"Important architectural fact #{i}: CyberJarvis operates on deterministic execution matrices with bounded memory buffers.",
                memory_type=MemoryType.PROJECT.value,
                importance=4,
                db_path=temp_db,
            )

        builder = MemoryContextBuilder(max_chars=2500, db_path=temp_db)
        context = builder.build_context("architectural facts", task_profile="PROJECT_ARCHITECTURE", limit=25)
        assert len(context) <= 2500
        assert len(context) > 0

    # 19. Deterministic Continuation Queries Routing and Handlers
    def test_continuation_queries(self, temp_db):
        router = ExecutionRouter()
        queries = [
            "what were we working on",
            "continue my previous task",
            "what did we decide",
            "what is the current CyberJarvis status",
            "what do you remember about me",
        ]
        for q in queries:
            tier = router.classify(q)
            assert tier == RouteType.FAST, f"Query '{q}' did not route to FAST (got {tier})"

        # Verify fast path execution returns deterministic grounded memory responses
        fast = FastPathHandler()
        resp = fast.try_handle("what were we working on")
        assert resp is not None
        assert "working" in resp.answer.lower() or "cyberjarvis" in resp.answer.lower()

        resp = fast.try_handle("what did we decide about stt")
        assert resp is not None
        assert "stt" in resp.answer.lower() or "decid" in resp.answer.lower()

        resp = fast.try_handle("what is the current CyberJarvis project status")
        assert resp is not None
        assert "status" in resp.answer.lower() or "phase" in resp.answer.lower()

    # 20. Generic Information Memory Isolation
    def test_generic_information_memory_isolation(self, temp_db):
        # Create user profile facts
        create_memory(
            content="User goal is to become a principal DevOps architect",
            memory_type=MemoryType.USER_PROFILE.value,
            importance=5,
            db_path=temp_db,
        )

        builder = MemoryContextBuilder(db_path=temp_db)

        # Generic query -> MUST NOT inject memory
        generic_context = builder.build_information_context("How does Python work?")
        assert generic_context == "", "Generic query received memory context!"

        # Personalized query -> Allowed to inject USER_PROFILE memory
        personalized_context = builder.build_information_context("How should I learn Python based on my goals?")
        assert personalized_context != "", "Personalized query was denied memory context!"
        assert "DevOps" in personalized_context

    # 21. Automatic Memory Extraction
    def test_automatic_memory_extraction(self, temp_db):
        examples = [
            ("My goal is to become a DevOps engineer", MemoryType.USER_PROFILE.value),
            ("CyberJarvis uses faster-whisper", MemoryType.PROJECT.value),
            ("We decided to replace Google STT with local Whisper", MemoryType.DECISION.value),
            ("STT latency was 17–27 seconds", MemoryType.ISSUE.value),
            ("Phase 3 is complete", MemoryType.PROJECT.value),
        ]
        for text, expected_type in examples:
            rec = extract_memory_from_text(text)
            assert rec is not None, f"Failed to extract memory from: {text}"
            assert rec.memory_type == expected_type, f"Extracted type {rec.memory_type} != {expected_type} for '{text}'"

            # Auto-save
            saved = auto_extract_and_save_memory(text, db_path=temp_db)
            assert saved is not None
            assert saved.memory_type == expected_type

    # 22. Backward Compatibility with Legacy Memory APIs
    def test_legacy_backward_compatibility(self, temp_db):
        # save_memory & get_recent_memories
        mid = save_memory("User query legacy", "Assistant answer legacy", memory_type="general", db_path=temp_db)
        assert mid > 0
        recent = get_recent_memories(limit=20, db_path=temp_db)
        found = next((r for r in recent if r[0] == "User query legacy"), None)
        assert found is not None
        assert found[1] == "Assistant answer legacy"

        # save_fact & get_facts
        fid = save_fact("My name is John Doe", db_path=temp_db)
        assert fid > 0
        facts = get_facts(limit=5, db_path=temp_db)
        assert "My name is John Doe" in facts

        # search_and_rank_memories
        ranked = search_and_rank_memories("legacy query", limit=5, db_path=temp_db)
        assert isinstance(ranked, list)
