"""Small local demo support queue; no messages are sent to Yandex."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[1] / "data/support.sqlite3"


@contextmanager
def connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY, request_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL, question TEXT NOT NULL,
            result_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new' CHECK(status IN ('new', 'closed'))
        )""")
        with conn:
            yield conn
    finally:
        conn.close()


def create_ticket(request_key, question, result, evidence):
    if not request_key or not question.strip():
        raise ValueError("Обращение должно содержать вопрос и идентификатор.")
    with connection() as conn:
        conn.execute("""INSERT INTO tickets(request_key, created_at, question, result_json, evidence_json)
                        VALUES (?, ?, ?, ?, ?) ON CONFLICT(request_key) DO NOTHING""",
                     (request_key, datetime.now(timezone.utc).isoformat(), question,
                      json.dumps(result, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False)))
        return conn.execute("SELECT id FROM tickets WHERE request_key=?", (request_key,)).fetchone()["id"]


def list_tickets():
    with connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 100")]


def close_ticket(ticket_id):
    with connection() as conn:
        conn.execute("UPDATE tickets SET status='closed' WHERE id=?", (ticket_id,))
