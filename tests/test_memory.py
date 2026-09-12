import unittest
from memory import init_db, get_facts, get_recent_memories, search_and_rank_memories, rank_memories


class TestMemory(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_facts_retrieval(self):
        facts = get_facts()
        self.assertIsInstance(facts, list)
        self.assertGreater(len(facts), 0)
        # Check that user's known fact exists
        has_name_or_project = any("Shivaji" in f or "CyberJarvis" in f for f in facts)
        self.assertTrue(has_name_or_project)

    def test_recent_memories(self):
        memories = get_recent_memories(limit=5)
        self.assertIsInstance(memories, list)
        self.assertGreater(len(memories), 0)
        first = memories[0]
        # Tuple format: user_message, assistant_response, created_at, memory_type, importance
        self.assertGreaterEqual(len(first), 4)

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
