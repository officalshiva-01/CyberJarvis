from typing import List, Tuple, Any


def rank_memories(memories: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """
    Rank memories using importance (index 4) and recency (index 2).
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
