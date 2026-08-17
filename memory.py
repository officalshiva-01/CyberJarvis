import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "cyberjarvis.db"


def init_db():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_memory(user_message, assistant_response):
    connection = sqlite3.connect(DB_PATH)

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
    connection = sqlite3.connect(DB_PATH)

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


if __name__ == "__main__":
    init_db()
    print(f"Memory database created at: {DB_PATH}")
