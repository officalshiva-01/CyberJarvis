import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "cyberjarvis.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            memory_type TEXT NOT NULL DEFAULT 'conversation',
            importance INTEGER NOT NULL DEFAULT 3
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL UNIQUE
        )
    """)

    # Migrate existing databases that do not yet have created_at.
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(memories)"
        )
    ]

    if "created_at" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN created_at TEXT
            """
        )

        connection.execute(
            """
            UPDATE memories
            SET created_at = CURRENT_TIMESTAMP
            WHERE created_at IS NULL
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_created_at
            ON memories(created_at)
            """
        )

    # Migrate existing databases that do not yet have memory_type.
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(memories)"
        )
    ]

    if "memory_type" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN memory_type TEXT NOT NULL
            DEFAULT 'conversation'
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
            """
        )

    # Migrate existing databases that do not yet have importance.
    columns = [
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(memories)"
        )
    ]

    if "importance" not in columns:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN importance INTEGER NOT NULL
            DEFAULT 3
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(importance)
            """
        )

    connection.commit()
    connection.close()


def save_memory(user_message, assistant_response, memory_type="conversation"):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO memories (
            user_message,
            assistant_response,
            created_at,
            memory_type
        )
        VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        """,
        (user_message, assistant_response, memory_type),
    )

    connection.commit()
    connection.close()


def get_recent_memories(limit=10):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT user_message, assistant_response, created_at, memory_type
        FROM memories
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    memories = cursor.fetchall()
    connection.close()

    return memories


def get_memories_by_type(memory_type, limit=10):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT user_message, assistant_response, created_at, memory_type
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


def search_memories(query, limit=10, memory_type=None):
    """
    Search stored memories by text content.

    Searches both user messages and assistant responses.
    Optionally filters results by memory type.
    Returns newest matching memories first.
    """
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


def rank_memories(memories):
    """
    Rank memories using importance and recency.

    Higher importance is preferred.
    Newer memories are preferred when importance is equal.
    """
    return sorted(
        memories,
        key=lambda memory: (
            memory[4],
            memory[2],
        ),
        reverse=True,
    )


def search_and_rank_memories(query, limit=10, memory_type=None):
    """
    Search memories and return the most relevant matches first.

    Relevance is determined by memory importance and recency.
    An optional memory type can restrict the search.
    """
    memories = search_memories(
        query,
        limit=limit,
        memory_type=memory_type,
    )

    return rank_memories(memories)


def save_fact(fact):
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


def get_facts(limit=20):
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


if __name__ == "__main__":
    init_db()
    print(f"Memory database initialized at: {DB_PATH}")
