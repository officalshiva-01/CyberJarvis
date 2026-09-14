import sqlite3
from pathlib import Path
from config import DB_PATH
from observability.logging import get_logger

logger = get_logger("CyberJarvis.Memory")


import os

def get_connection() -> sqlite3.Connection:
    db_path = os.getenv("CYBERJARVIS_DB_PATH", str(DB_PATH))
    return sqlite3.connect(db_path)


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

    columns = [
        row[1]
        for row in connection.execute("PRAGMA table_info(memories)")
    ]

    if "created_at" not in columns:
        connection.execute("ALTER TABLE memories ADD COLUMN created_at TEXT")
        connection.execute("UPDATE memories SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")

    if "memory_type" not in columns:
        connection.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'conversation'")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")

    if "importance" not in columns:
        connection.execute("ALTER TABLE memories ADD COLUMN importance INTEGER NOT NULL DEFAULT 3")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")

    connection.commit()
    connection.close()
    logger.debug(f"Database initialized at {DB_PATH}")
