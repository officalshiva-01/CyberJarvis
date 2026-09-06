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
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

    connection.commit()
    connection.close()


def save_memory(user_message, assistant_response):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO memories (
            user_message,
            assistant_response,
            created_at
        )
        VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
        (user_message, assistant_response),
    )

    connection.commit()
    connection.close()


def get_recent_memories(limit=10):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT user_message, assistant_response, created_at
        FROM memories
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    memories = cursor.fetchall()
    connection.close()

    return memories


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
