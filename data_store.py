from __future__ import annotations

import os
import sqlite3
from typing import Any

from barrier_seed_data import BARRIER_SEED_ROWS
from settings import settings


def _project_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(__file__), value))


def _db_path() -> str:
    return _project_path(settings.DATABASE_PATH)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_storage() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reaction_options (
                code TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                message_id INTEGER NOT NULL,
                stage TEXT,
                scenario TEXT,
                text TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_reactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                reaction_code TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chat_id, user_id, message_id, reaction_code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS barrier_rows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sheet_name TEXT NOT NULL,
                row_num INTEGER NOT NULL,
                barrier_name TEXT NOT NULL,
                tool_name TEXT NOT NULL UNIQUE,
                k_text TEXT,
                l_text TEXT,
                m_text TEXT,
                prompt_text TEXT NOT NULL
            )
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM reaction_options").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO reaction_options (code, label, sort_order) VALUES (?, ?, ?)",
                [
                    ("like", "👍", 10),
                    ("dislike", "👎", 20),
                    ("insight", "💡", 30),
                ],
            )


def _tool_name(row_num: int) -> str:
    return f"barrier_1_{row_num}"


def _make_prompt(barrier_name: str, k: str, l: str, m: str) -> str:
    parts: list[str] = []
    if k:
        parts.append(f"K:\n{k}")
    if l:
        parts.append(f"L:\n{l}")
    if m:
        parts.append(f"M:\n{m}")
    body = "\n\n".join(parts).strip()
    title = barrier_name if barrier_name else "Без названия"
    return f"Барьер: {title}\n\n{body}".strip()


def seed_barriers() -> int:
    rows: list[tuple[str, int, str, str, str, str, str, str]] = []
    for row in BARRIER_SEED_ROWS:
        row_num = int(row["row_num"])
        barrier_name = str(row.get("barrier_name", "")).strip()
        k = str(row.get("k_text", "")).strip()
        l = str(row.get("l_text", "")).strip()
        m = str(row.get("m_text", "")).strip()
        tool_name = _tool_name(row_num)
        prompt_text = _make_prompt(barrier_name, k, l, m)
        rows.append(
            (
                str(row.get("sheet_name", "Барьеры_для старшеклассников")).strip(),
                row_num,
                barrier_name,
                tool_name,
                k,
                l,
                m,
                prompt_text,
            )
        )
    with _connect() as conn:
        conn.execute("DELETE FROM barrier_rows")
        conn.executemany(
            """
            INSERT INTO barrier_rows
            (sheet_name, row_num, barrier_name, tool_name, k_text, l_text, m_text, prompt_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_reaction_options() -> list[dict[str, str]]:
    init_storage()
    with _connect() as conn:
        data = conn.execute(
            "SELECT code, label FROM reaction_options ORDER BY sort_order, code"
        ).fetchall()
    return [{"code": str(r["code"]), "label": str(r["label"])} for r in data]


def save_bot_message(
    chat_id: int,
    user_id: int | None,
    message_id: int,
    text: str,
    stage: str,
    scenario: str,
) -> None:
    init_storage()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO bot_messages (chat_id, user_id, message_id, stage, scenario, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chat_id, user_id, message_id, stage, scenario, text),
        )


def save_reaction(chat_id: int, user_id: int, message_id: int, reaction_code: str) -> bool:
    init_storage()
    with _connect() as conn:
        valid = conn.execute(
            "SELECT 1 FROM reaction_options WHERE code = ?",
            (reaction_code,),
        ).fetchone()
        if valid is None:
            return False
        try:
            conn.execute(
                """
                INSERT INTO message_reactions (chat_id, user_id, message_id, reaction_code)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, user_id, message_id, reaction_code),
            )
            return True
        except sqlite3.IntegrityError:
            return True


def get_barrier_tools(limit: int) -> list[dict[str, Any]]:
    init_storage()
    with _connect() as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM barrier_rows").fetchone()[0]
        if row_count == 0:
            seed_barriers()
        rows = conn.execute(
            """
            SELECT tool_name, barrier_name, prompt_text
            FROM barrier_rows
            ORDER BY row_num
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    tools: list[dict[str, Any]] = []
    for r in rows:
        name = str(r["tool_name"])
        barrier_name = (r["barrier_name"] or "").strip()
        description = barrier_name if barrier_name else str(r["prompt_text"])[:180]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description[:300],
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }
        )
    return tools


def get_barrier_prompt_by_tool(tool_name: str) -> str:
    init_storage()
    with _connect() as conn:
        row = conn.execute(
            "SELECT prompt_text FROM barrier_rows WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()
    if row is None:
        return ""
    return str(row["prompt_text"] or "").strip()
