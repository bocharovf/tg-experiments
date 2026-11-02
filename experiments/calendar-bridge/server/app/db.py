"""Хранилище состояния синхронизации (SQLite)."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from .matcher import Existing, norm_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    uid        TEXT PRIMARY KEY,
    gid        TEXT,
    title      TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    all_day    INTEGER NOT NULL DEFAULT 0,
    miss       INTEGER NOT NULL DEFAULT 0,
    state      TEXT NOT NULL DEFAULT 'active',
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    last_seen_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_state_start ON events(state, start_ms);

CREATE TABLE IF NOT EXISTS sync_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms      INTEGER NOT NULL,
    ip_hash    TEXT,
    sent_ms    INTEGER,
    received   INTEGER NOT NULL,
    created    INTEGER NOT NULL,
    moved      INTEGER NOT NULL,
    retitled   INTEGER NOT NULL,
    extended   INTEGER NOT NULL,
    unchanged  INTEGER NOT NULL,
    deleted    INTEGER NOT NULL,
    suppressed INTEGER NOT NULL DEFAULT 0
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        if not existed:
            os.chmod(path, 0o600)

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(events)")}
        if "all_day" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN all_day INTEGER NOT NULL DEFAULT 0")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- события ---------------------------------------------------------
    def active(self) -> list[Existing]:
        rows = self.conn.execute(
            "SELECT uid, gid, title, start_ms, end_ms, miss, all_day FROM events WHERE state='active'"
        ).fetchall()
        return [
            Existing(
                uid=r["uid"], gid=r["gid"], title=r["title"],
                start_ms=r["start_ms"], end_ms=r["end_ms"], miss=r["miss"],
                all_day=bool(r["all_day"]),
            )
            for r in rows
        ]

    def add(self, uid: str, gid: str, title: str, start_ms: int, end_ms: int, all_day: bool = False) -> None:
        ts = now_ms()
        self.conn.execute(
            "INSERT INTO events(uid, gid, title, title_norm, start_ms, end_ms, all_day, miss, state,"
            " created_ms, updated_ms, last_seen_ms) VALUES(?,?,?,?,?,?,?,0,'active',?,?,?)",
            (uid, gid, title, norm_title(title), start_ms, end_ms, int(all_day), ts, ts, ts),
        )
        self.conn.commit()

    def touch(self, uid: str, title: str, start_ms: int, end_ms: int, all_day: bool = False) -> None:
        ts = now_ms()
        self.conn.execute(
            "UPDATE events SET title=?, title_norm=?, start_ms=?, end_ms=?, all_day=?, miss=0,"
            " updated_ms=?, last_seen_ms=? WHERE uid=?",
            (title, norm_title(title), start_ms, end_ms, int(all_day), ts, ts, uid),
        )
        self.conn.commit()

    def mark_miss(self, uid: str) -> None:
        self.conn.execute("UPDATE events SET miss=miss+1, updated_ms=? WHERE uid=?", (now_ms(), uid))
        self.conn.commit()

    def mark_deleted(self, uid: str) -> None:
        self.conn.execute(
            "UPDATE events SET state='deleted', updated_ms=? WHERE uid=?", (now_ms(), uid)
        )
        self.conn.commit()

    # --- журнал синхронизаций -------------------------------------------
    def record_run(self, **kw) -> None:
        self.conn.execute(
            "INSERT INTO sync_runs(ts_ms, ip_hash, sent_ms, received, created, moved,"
            " retitled, extended, unchanged, deleted, suppressed)"
            " VALUES(:ts_ms,:ip_hash,:sent_ms,:received,:created,:moved,"
            ":retitled,:extended,:unchanged,:deleted,:suppressed)",
            kw,
        )
        self.conn.commit()

    def last_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sync_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, MIN(start_ms) AS first, MAX(start_ms) AS last"
            " FROM events WHERE state='active'"
        ).fetchone()
        return {"active_events": row["n"], "first_start_ms": row["first"], "last_start_ms": row["last"]}
