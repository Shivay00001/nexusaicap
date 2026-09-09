"""Conversation Memory — SQLite-backed persistent conversation history."""
from __future__ import annotations
import json, time, sqlite3
from pathlib import Path
from dataclasses import dataclass
from nexusai.core.config import MEMORY_DIR

@dataclass
class MemoryEntry:
    role: str; content: str; timestamp: float; session_id: str; metadata: dict = None

class ConversationMemory:
    """Persistent conversation memory using SQLite."""
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(MEMORY_DIR / "conversations.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, timestamp REAL NOT NULL,
            metadata TEXT DEFAULT '{}')""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, title TEXT DEFAULT '')""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
        conn.commit(); conn.close()

    def add(self, role: str, content: str, session_id: str = "default", metadata: dict = None):
        conn = sqlite3.connect(self.db_path); ts = time.time()
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, ts, json.dumps(metadata or {})))
        conn.execute("INSERT OR REPLACE INTO sessions (id, created_at, updated_at) VALUES (?, COALESCE((SELECT created_at FROM sessions WHERE id = ?), ?), ?)",
            (session_id, session_id, ts, ts))
        conn.commit(); conn.close()

    def get_history(self, session_id: str = "default", limit: int = 50) -> list[dict]:
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit)).fetchall()
        conn.close()
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    def list_sessions(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, created_at, updated_at, title FROM sessions ORDER BY updated_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def clear(self, session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        if session_id:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        else:
            conn.execute("DELETE FROM messages"); conn.execute("DELETE FROM sessions")
        conn.commit(); conn.close()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT role, content, session_id, timestamp FROM messages WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?", (f"%{query}%", limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
