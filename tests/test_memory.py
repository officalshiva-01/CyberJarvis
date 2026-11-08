import tempfile
import unittest
from pathlib import Path

from memory import (
    init_db,
    get_facts,
    get_recent_memories,
    rank_memories,
    save_fact,
)


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_memory.db")
        init_db(db_path=self.db_path)
        save_fact("Shivaji is developing CyberJarvis", db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_facts_retrieval(self):
        facts = get_facts(db_path=self.db_path)
        self.assertIsInstance(facts, list)
        self.assertGreater(len(facts), 0)

        has_name_or_project = any(
            "Shivaji" in fact or "CyberJarvis" in fact
            for fact in facts
        )
        self.assertTrue(has_name_or_project)

    def test_recent_memories(self):
        memories = get_recent_memories(limit=5, db_path=self.db_path)
        self.assertIsInstance(memories, list)
        self.assertGreater(len(memories), 0)
        self.assertGreaterEqual(len(memories[0]), 4)

    def test_memory_ranking(self):
        mock_memories = [
            ("low importance", "resp1", "2026-09-01 10:00:00", "conversation", 1),
            ("high importance old", "resp2", "2026-08-01 10:00:00", "conversation", 5),
            ("high importance new", "resp3", "2026-09-02 10:00:00", "conversation", 5),
        ]

        ranked = rank_memories(mock_memories)
        self.assertEqual(ranked[0][0], "high importance new")
        self.assertEqual(ranked[1][0], "high importance old")
        self.assertEqual(ranked[2][0], "low importance")


if __name__ == "__main__":
    unittest.main()
