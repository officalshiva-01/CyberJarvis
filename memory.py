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
            assistant_response TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact TEXT NOT NULL UNIQUE
        )
    """)

    connection.commit()
    connection.close()


def save_memory(user_message, assistant_response):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO memories (user_message, assistant_response)
        VALUES (?, ?)
        """,
        (user_message, assistant_response),
    )

    connection.commit()
    connection.close()


def get_recent_memories(limit=10):
    connection = get_connection()

    cursor = connection.execute(
        """
        SELECT user_message, assistant_response
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
