"""Phase 2 (M2A): SQLite Longitudinal Store for Bowlers, Sessions, Deliveries, Calibrations & Workload Logs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Optional
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StudioDatabase:
    """Thread-safe SQLite store (WAL mode, strictly parameterized queries) for local-first studio data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path), timeout=15.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS bowlers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    bowling_arm TEXT NOT NULL DEFAULT 'right',
                    age_band TEXT NOT NULL DEFAULT 'adult',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    bowler_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (bowler_id) REFERENCES bowlers(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS deliveries (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    pipeline_mode TEXT NOT NULL DEFAULT 'single_view',
                    video_path TEXT NOT NULL DEFAULT '',
                    proxy_path TEXT NOT NULL DEFAULT '',
                    data_dir TEXT NOT NULL,
                    start_frame INTEGER NOT NULL DEFAULT 0,
                    end_frame INTEGER,
                    status TEXT NOT NULL DEFAULT 'ready',
                    fact_sheet_json TEXT,
                    isb_summary_json TEXT,
                    pan_summary_json TEXT,
                    overrides_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS calibration_profiles (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    mcc_landmarks_json TEXT NOT NULL DEFAULT '{}',
                    calibration_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS workload_logs (
                    id TEXT PRIMARY KEY,
                    bowler_id TEXT NOT NULL,
                    log_date TEXT NOT NULL,
                    deliveries_count INTEGER NOT NULL,
                    session_type TEXT NOT NULL DEFAULT 'net',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (bowler_id) REFERENCES bowlers(id) ON DELETE CASCADE
                );
                """
            )

    def create_bowler(self, name: str, bowling_arm: str = "right", age_band: str = "adult", bowler_id: Optional[str] = None) -> Dict[str, Any]:
        bid = bowler_id or f"bwl_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO bowlers (id, name, bowling_arm, age_band, created_at) VALUES (?, ?, ?, ?, ?)",
                (bid, name, bowling_arm, age_band, now),
            )
        return {"id": bid, "name": name, "bowling_arm": bowling_arm, "age_band": age_band, "created_at": now}

    def list_bowlers(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute("SELECT * FROM bowlers ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def get_bowler(self, bowler_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute("SELECT * FROM bowlers WHERE id = ?", (bowler_id,)).fetchone()
            return dict(r) if r else None

    def create_session(self, bowler_id: str, title: str, notes: str = "", session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or f"ses_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO sessions (id, bowler_id, title, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (sid, bowler_id, title, notes, now),
            )
        return {"id": sid, "bowler_id": bowler_id, "title": title, "notes": notes, "created_at": now}

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT s.*, b.name AS bowler_name, b.bowling_arm, b.age_band,
                       (SELECT COUNT(*) FROM deliveries d WHERE d.session_id = s.id) AS delivery_count
                FROM sessions s
                JOIN bowlers b ON b.id = s.bowler_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute(
                """
                SELECT s.*, b.name AS bowler_name, b.bowling_arm, b.age_band
                FROM sessions s
                JOIN bowlers b ON b.id = s.bowler_id
                WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
            if not r:
                return None
            out = dict(r)
            dels = con.execute(
                """
                SELECT id, session_id, label, pipeline_mode, status, start_frame, end_frame, created_at, updated_at
                FROM deliveries WHERE session_id = ? ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
            out["deliveries"] = [dict(d) for d in dels]
            return out

    def upsert_delivery(
        self,
        delivery_id: str,
        session_id: str,
        label: str,
        data_dir: str | Path,
        video_path: str = "",
        proxy_path: str = "",
        pipeline_mode: str = "single_view",
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        status: str = "ready",
        fact_sheet_json: Optional[str] = None,
        isb_summary: Optional[Dict[str, Any]] = None,
        pan_summary: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now()
        with self._conn() as con:
            existing = con.execute("SELECT created_at FROM deliveries WHERE id = ?", (delivery_id,)).fetchone()
            created = existing["created_at"] if existing else now
            con.execute(
                """
                INSERT OR REPLACE INTO deliveries (
                    id, session_id, label, pipeline_mode, video_path, proxy_path, data_dir,
                    start_frame, end_frame, status, fact_sheet_json, isb_summary_json,
                    pan_summary_json, overrides_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    session_id,
                    label,
                    pipeline_mode,
                    str(video_path),
                    str(proxy_path),
                    str(data_dir),
                    int(start_frame),
                    int(end_frame) if end_frame is not None else None,
                    status,
                    fact_sheet_json,
                    json.dumps(isb_summary) if isb_summary is not None else None,
                    json.dumps(pan_summary) if pan_summary is not None else None,
                    json.dumps(overrides or {}),
                    created,
                    now,
                ),
            )
        return self.get_delivery(delivery_id) or {"id": delivery_id}

    def get_delivery(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute("SELECT * FROM deliveries WHERE id = ?", (delivery_id,)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["overrides"] = json.loads(d.pop("overrides_json") or "{}")
            d["isb_summary"] = json.loads(d.pop("isb_summary_json")) if d.get("isb_summary_json") else None
            d["pan_summary"] = json.loads(d.pop("pan_summary_json")) if d.get("pan_summary_json") else None
            return d

    def list_session_deliveries(self, session_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM deliveries WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["overrides"] = json.loads(d.pop("overrides_json") or "{}")
                d["isb_summary"] = json.loads(d.pop("isb_summary_json")) if d.get("isb_summary_json") else None
                d["pan_summary"] = json.loads(d.pop("pan_summary_json")) if d.get("pan_summary_json") else None
                out.append(d)
            return out

    def save_calibration_profile(
        self,
        session_id: str,
        name: str,
        method: str,
        mcc_landmarks_px: Dict[str, Any],
        calibration_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        cid = f"cal_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO calibration_profiles (id, session_id, name, method, mcc_landmarks_json, calibration_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, session_id, name, method, json.dumps(mcc_landmarks_px), json.dumps(calibration_dict), now),
            )
        return {"id": cid, "session_id": session_id, "name": name, "method": method, "created_at": now}

    def add_workload_log(
        self,
        bowler_id: str,
        log_date: str,
        deliveries_count: int,
        session_type: str = "net",
        notes: str = "",
    ) -> Dict[str, Any]:
        wid = f"wkl_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO workload_logs (id, bowler_id, log_date, deliveries_count, session_type, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (wid, bowler_id, log_date, int(deliveries_count), session_type, notes, now),
            )
        return {
            "id": wid,
            "bowler_id": bowler_id,
            "log_date": log_date,
            "deliveries_count": int(deliveries_count),
            "session_type": session_type,
            "notes": notes,
            "created_at": now,
        }

    def list_workload_logs(self, bowler_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM workload_logs WHERE bowler_id = ? ORDER BY log_date DESC, created_at DESC",
                (bowler_id,),
            ).fetchall()
            return [dict(r) for r in rows]
