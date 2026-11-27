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

                CREATE TABLE IF NOT EXISTS coach_notes (
                    id TEXT PRIMARY KEY,
                    delivery_id TEXT NOT NULL,
                    frame_idx INTEGER,
                    timestamp_s REAL,
                    ball_tag TEXT NOT NULL DEFAULT '',
                    note_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (delivery_id) REFERENCES deliveries(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS delivery_annotations (
                    delivery_id TEXT PRIMARY KEY,
                    annotations_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (delivery_id) REFERENCES deliveries(id) ON DELETE CASCADE
                );
                """
            )
            # Safe idempotent migration for pre-v3 databases missing `timestamp_s` on `coach_notes`
            cols = {r[1] for r in con.execute("PRAGMA table_info(coach_notes)").fetchall()}
            if "timestamp_s" not in cols:
                con.execute("ALTER TABLE coach_notes ADD COLUMN timestamp_s REAL")

            # Safe idempotent Studio v5 migration on `workload_logs` (RPE, spell length, soreness & pain flags)
            wkl_cols = {r[1] for r in con.execute("PRAGMA table_info(workload_logs)").fetchall()}
            for col_name, col_def in [
                ("rpe", "INTEGER NOT NULL DEFAULT 6"),
                ("max_spell_balls", "INTEGER"),
                ("soreness_lumbar", "INTEGER"),
                ("fatigue_level", "INTEGER"),
                ("pain_when_bowling", "INTEGER NOT NULL DEFAULT 0"),
            ]:
                if col_name not in wkl_cols:
                    con.execute(f"ALTER TABLE workload_logs ADD COLUMN {col_name} {col_def}")

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

    def list_bowlers_with_stats(self) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT b.*,
                       (
                           SELECT COUNT(*)
                           FROM deliveries d
                           JOIN sessions s ON s.id = d.session_id
                           WHERE s.bowler_id = b.id
                       ) AS delivery_count,
                       (
                           SELECT MAX(d.created_at)
                           FROM deliveries d
                           JOIN sessions s ON s.id = d.session_id
                           WHERE s.bowler_id = b.id
                       ) AS last_delivery_at
                FROM bowlers b
                ORDER BY b.created_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_bowler(self, bowler_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute("SELECT * FROM bowlers WHERE id = ?", (bowler_id,)).fetchone()
            return dict(r) if r else None

    def get_or_create_dated_session(self, bowler_id: str, session_date: Optional[str] = None) -> Dict[str, Any]:
        """Auto-group bowler uploads by date under the hood so consumer flows never expose session IDs."""
        day_str = session_date or datetime.now(timezone.utc).date().isoformat()
        title = f"Net Analysis — {day_str}"
        with self._conn() as con:
            r = con.execute(
                """
                SELECT * FROM sessions
                WHERE bowler_id = ? AND id != 'ses_demo' AND title = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (bowler_id, title),
            ).fetchone()
            if r:
                return dict(r)
        return self.create_session(bowler_id=bowler_id, title=title, notes="Auto-created bowler session")

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

    def list_all_deliveries(self, bowler_id: Optional[str] = None, limit: int = 60, ascending: bool = False) -> List[Dict[str, Any]]:
        order_dir = "ASC" if ascending else "DESC"
        with self._conn() as con:
            if bowler_id:
                rows = con.execute(
                    f"""
                    SELECT d.*, s.bowler_id, s.title AS session_title,
                           b.name AS bowler_name, b.bowling_arm, b.age_band
                    FROM deliveries d
                    JOIN sessions s ON s.id = d.session_id
                    JOIN bowlers b ON b.id = s.bowler_id
                    WHERE s.bowler_id = ?
                    ORDER BY d.created_at {order_dir}
                    LIMIT ?
                    """,
                    (bowler_id, int(limit)),
                ).fetchall()
            else:
                rows = con.execute(
                    f"""
                    SELECT d.*, s.bowler_id, s.title AS session_title,
                           b.name AS bowler_name, b.bowling_arm, b.age_band
                    FROM deliveries d
                    JOIN sessions s ON s.id = d.session_id
                    JOIN bowlers b ON b.id = s.bowler_id
                    ORDER BY d.created_at {order_dir}
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["overrides"] = json.loads(d.pop("overrides_json") or "{}")
                d["isb_summary"] = json.loads(d.pop("isb_summary_json")) if d.get("isb_summary_json") else None
                d["pan_summary"] = json.loads(d.pop("pan_summary_json")) if d.get("pan_summary_json") else None
                out.append(d)
            return out

    def get_delivery_with_bowler(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute(
                """
                SELECT d.*, s.bowler_id, s.title AS session_title,
                       b.name AS bowler_name, b.bowling_arm, b.age_band
                FROM deliveries d
                JOIN sessions s ON s.id = d.session_id
                JOIN bowlers b ON b.id = s.bowler_id
                WHERE d.id = ?
                """,
                (delivery_id,),
            ).fetchone()
            if not r:
                return None
            d = dict(r)
            d["overrides"] = json.loads(d.pop("overrides_json") or "{}")
            d["isb_summary"] = json.loads(d.pop("isb_summary_json")) if d.get("isb_summary_json") else None
            d["pan_summary"] = json.loads(d.pop("pan_summary_json")) if d.get("pan_summary_json") else None
            return d

    def rename_delivery(self, delivery_id: str, label: str, fact_sheet_json: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now = _utc_now()
        with self._conn() as con:
            if fact_sheet_json is not None:
                cur = con.execute(
                    "UPDATE deliveries SET label = ?, fact_sheet_json = ?, updated_at = ? WHERE id = ?",
                    (label, fact_sheet_json, now, delivery_id),
                )
            else:
                cur = con.execute(
                    "UPDATE deliveries SET label = ?, updated_at = ? WHERE id = ?",
                    (label, now, delivery_id),
                )
            if cur.rowcount == 0:
                return None
        return self.get_delivery_with_bowler(delivery_id)

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
        rpe: int = 6,
        max_spell_balls: Optional[int] = None,
        soreness_lumbar: Optional[int] = None,
        fatigue_level: Optional[int] = None,
        pain_when_bowling: bool = False,
    ) -> Dict[str, Any]:
        wid = f"wkl_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        rpe_use = int(max(1, min(10, int(rpe))))
        spell_use = int(max_spell_balls) if max_spell_balls is not None else None
        sore_use = int(max(0, min(10, int(soreness_lumbar)))) if soreness_lumbar is not None else None
        fat_use = int(max(1, min(10, int(fatigue_level)))) if fatigue_level is not None else None
        pain_int = 1 if pain_when_bowling else 0
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO workload_logs (
                    id, bowler_id, log_date, deliveries_count, session_type, notes,
                    rpe, max_spell_balls, soreness_lumbar, fatigue_level, pain_when_bowling, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wid,
                    bowler_id,
                    log_date,
                    int(deliveries_count),
                    session_type,
                    notes,
                    rpe_use,
                    spell_use,
                    sore_use,
                    fat_use,
                    pain_int,
                    now,
                ),
            )
        return {
            "id": wid,
            "bowler_id": bowler_id,
            "log_date": log_date,
            "deliveries_count": int(deliveries_count),
            "session_type": session_type,
            "notes": notes,
            "rpe": rpe_use,
            "max_spell_balls": spell_use,
            "soreness_lumbar": sore_use,
            "fatigue_level": fat_use,
            "pain_when_bowling": bool(pain_int),
            "created_at": now,
        }

    def list_workload_logs(self, bowler_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM workload_logs WHERE bowler_id = ? ORDER BY log_date DESC, created_at DESC",
                (bowler_id,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["pain_when_bowling"] = bool(d.get("pain_when_bowling") or 0)
                out.append(d)
            return out

    def delete_workload_log(self, bowler_id: str, log_id: str) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM workload_logs WHERE id = ? AND bowler_id = ?", (log_id, bowler_id))
            return cur.rowcount > 0

    def delete_delivery(self, delivery_id: str) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM deliveries WHERE id = ?", (delivery_id,))
            return cur.rowcount > 0

    def list_notes(self, delivery_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM coach_notes WHERE delivery_id = ? ORDER BY created_at DESC",
                (delivery_id,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                out.append(
                    {
                        "id": d["id"],
                        "delivery_id": d["delivery_id"],
                        "frame_idx": d.get("frame_idx"),
                        "timestamp_s": d.get("timestamp_s"),
                        "tag": d.get("ball_tag", ""),
                        "body": d.get("note_text", ""),
                        "created_at": d["created_at"],
                    }
                )
            return out

    def add_note(
        self,
        delivery_id: str,
        note_text: str,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
        tag: str = "",
        ball_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        nid = f"note_{uuid.uuid4().hex[:10]}"
        now = _utc_now()
        use_tag = ball_tag if ball_tag is not None else tag
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO coach_notes (id, delivery_id, frame_idx, timestamp_s, ball_tag, note_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    nid,
                    delivery_id,
                    int(frame_idx) if frame_idx is not None else None,
                    float(timestamp_s) if timestamp_s is not None else None,
                    use_tag,
                    note_text,
                    now,
                ),
            )
        return {
            "id": nid,
            "delivery_id": delivery_id,
            "frame_idx": int(frame_idx) if frame_idx is not None else None,
            "timestamp_s": float(timestamp_s) if timestamp_s is not None else None,
            "tag": use_tag,
            "body": note_text,
            "created_at": now,
        }

    def delete_note(self, delivery_id: str, note_id: str) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM coach_notes WHERE id = ? AND delivery_id = ?", (note_id, delivery_id))
            return cur.rowcount > 0

    def get_annotations(self, delivery_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            r = con.execute(
                "SELECT annotations_json FROM delivery_annotations WHERE delivery_id = ?",
                (delivery_id,),
            ).fetchone()
            if not r or not r["annotations_json"]:
                return []
            parsed = json.loads(r["annotations_json"])
            return parsed if isinstance(parsed, list) else []

    def save_annotations(self, delivery_id: str, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now = _utc_now()
        payload = list(annotations or [])
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO delivery_annotations (delivery_id, annotations_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(delivery_id) DO UPDATE SET
                    annotations_json = excluded.annotations_json,
                    updated_at = excluded.updated_at
                """,
                (delivery_id, json.dumps(payload), now),
            )
        return payload
