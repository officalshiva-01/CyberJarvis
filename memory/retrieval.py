from typing import List, Tuple, Optional, Any
from memory.store import get_connection
from memory.ranking import rank_memories


def save_memory(
    user_message: str,
    assistant_response: str,
    memory_type: str = "conversation",
    importance: int = 3,
):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO memories (
            user_message,
            assistant_response,
            created_at,
            memory_type,
            importance
        )
        VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
        """,
        (user_message, assistant_response, memory_type, importance),
    )
    connection.commit()
    connection.close()


def get_recent_memories(limit: int = 10) -> List[Tuple[str, str, str, str, int]]:
    connection = get_connection()
    cursor = connection.execute(
        """
        SELECT user_message, assistant_response, created_at, memory_type, importance
        FROM memories
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    memories = cursor.fetchall()
    connection.close()
    return memories


def get_memories_by_type(memory_type: str, limit: int = 10) -> List[Tuple[str, str, str, str, int]]:
    connection = get_connection()
    cursor = connection.execute(
        """
        SELECT user_message, assistant_response, created_at, memory_type, importance
        FROM memories
        WHERE memory_type = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (memory_type, limit),
    )
    memories = cursor.fetchall()
    connection.close()
    return memories


def search_memories(
    query: str, limit: int = 10, memory_type: Optional[str] = None
) -> List[Tuple[str, str, str, str, int]]:
    query = query.strip()
    if not query:
        return []

    connection = get_connection()
    if memory_type:
        cursor = connection.execute(
            """
            SELECT user_message, assistant_response, created_at,
                   memory_type, importance
            FROM memories
            WHERE (user_message LIKE ?
               OR assistant_response LIKE ?)
              AND memory_type = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", memory_type, limit),
        )
    else:
        cursor = connection.execute(
            """
            SELECT user_message, assistant_response, created_at,
                   memory_type, importance
            FROM memories
            WHERE user_message LIKE ?
               OR assistant_response LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit),
        )

    memories = cursor.fetchall()
    connection.close()
    return memories


def search_and_rank_memories(
    query: str, limit: int = 10, memory_type: Optional[str] = None
) -> List[Tuple[str, str, str, str, int]]:
    memories = search_memories(query, limit=limit, memory_type=memory_type)
    return rank_memories(memories)


def save_fact(fact: str):
    fact = fact.strip()
    if not fact:
        return

    connection = get_connection()
    connection.execute(
        """
        INSERT OR IGNORE INTO facts (fact)
        VALUES (?)
        """,
        (fact,),
    )
    connection.commit()
    connection.close()


def get_facts(limit: int = 20) -> List[str]:
    connection = get_connection()
    cursor = connection.execute(
        """
        SELECT fact
        FROM facts
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    facts = [row[0] for row in cursor.fetchall()]
    connection.close()
    return facts
