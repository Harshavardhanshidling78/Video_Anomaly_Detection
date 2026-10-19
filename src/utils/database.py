import sqlite3
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from src.config import DB_PATH


class AnomalyDatabase:
    """
    Production-Grade SQLite Manager for Video Anomaly Detection.

    Schema:
        video_sessions   — one row per full video analysis run
        anomaly_events   — one row per anomalous clip/frame detected
        score_timeseries — full clip-by-clip score trace for every session

    Features:
        - WAL (Write-Ahead Logging) for concurrent read/write safety
        - Indexed columns for O(log n) query performance
        - Full search, date-range, score-range filtering
        - Pagination support (LIMIT / OFFSET)
        - Session-level analytics (per-video summary)
        - Bulk insert for score time-series
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    # ──────────────────────────────────────────────
    # CONNECTION
    # ──────────────────────────────────────────────
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
        finally:
            conn.close()

    # ──────────────────────────────────────────────
    # SCHEMA INIT
    # ──────────────────────────────────────────────
    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Table 1: one row per full video analysis session
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_sessions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    video_source    TEXT    NOT NULL,
                    spatial_mode    TEXT    NOT NULL DEFAULT 'fast',
                    threshold_used  REAL    NOT NULL,
                    total_frames    INTEGER NOT NULL DEFAULT 0,
                    total_clips     INTEGER NOT NULL DEFAULT 0,
                    anomaly_clips   INTEGER NOT NULL DEFAULT 0,
                    max_score       REAL    NOT NULL DEFAULT 0.0,
                    avg_score       REAL    NOT NULL DEFAULT 0.0,
                    anomaly_detected INTEGER NOT NULL DEFAULT 0,
                    use_ema         INTEGER NOT NULL DEFAULT 1,
                    use_adaptive    INTEGER NOT NULL DEFAULT 0,
                    notes           TEXT    DEFAULT ''
                );
            """)

            # Table 2: individual anomalous clip events (high-score alerts)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      INTEGER REFERENCES video_sessions(id),
                    timestamp       TEXT    NOT NULL,
                    video_source    TEXT    NOT NULL,
                    frame_number    INTEGER NOT NULL,
                    anomaly_score   REAL    NOT NULL,
                    classification  TEXT    NOT NULL,
                    threshold_used  REAL    NOT NULL,
                    notes           TEXT    DEFAULT ''
                );
            """)

            # Migration: Ensure session_id exists in anomaly_events if table was created previously
            cur.execute("PRAGMA table_info(anomaly_events);")
            existing_cols = {row["name"] for row in cur.fetchall()}
            if "session_id" not in existing_cols:
                cur.execute("ALTER TABLE anomaly_events ADD COLUMN session_id INTEGER REFERENCES video_sessions(id);")

            # Table 3: full clip-by-clip score trace per session
            cur.execute("""
                CREATE TABLE IF NOT EXISTS score_timeseries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL REFERENCES video_sessions(id),
                    clip_index  INTEGER NOT NULL,
                    raw_score   REAL    NOT NULL,
                    ema_score   REAL    NOT NULL,
                    is_anomalous INTEGER NOT NULL DEFAULT 0
                );
            """)

            # ── Indices for fast lookups
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_ae_timestamp  ON anomaly_events(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_ae_source     ON anomaly_events(video_source);",
                "CREATE INDEX IF NOT EXISTS idx_ae_score      ON anomaly_events(anomaly_score);",
                "CREATE INDEX IF NOT EXISTS idx_ae_session    ON anomaly_events(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_vs_timestamp  ON video_sessions(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_vs_source     ON video_sessions(video_source);",
                "CREATE INDEX IF NOT EXISTS idx_ts_session    ON score_timeseries(session_id);",
            ]
            for idx in indices:
                cur.execute(idx)

            conn.commit()

    # ──────────────────────────────────────────────
    # WRITE — SESSION
    # ──────────────────────────────────────────────
    def log_session(
        self,
        video_source: str,
        spatial_mode: str,
        threshold: float,
        total_frames: int,
        total_clips: int,
        anomaly_clips: int,
        max_score: float,
        avg_score: float,
        anomaly_detected: bool,
        use_ema: bool = True,
        use_adaptive: bool = False,
        notes: str = ""
    ) -> int:
        """
        Inserts a complete video analysis session record.
        Returns the new session_id.
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO video_sessions
                (timestamp, video_source, spatial_mode, threshold_used,
                 total_frames, total_clips, anomaly_clips,
                 max_score, avg_score, anomaly_detected,
                 use_ema, use_adaptive, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now, video_source, spatial_mode, float(threshold),
                int(total_frames), int(total_clips), int(anomaly_clips),
                float(max_score), float(avg_score), int(anomaly_detected),
                int(use_ema), int(use_adaptive), notes
            ))
            conn.commit()
            return cur.lastrowid

    # ──────────────────────────────────────────────
    # WRITE — ANOMALY EVENT
    # ──────────────────────────────────────────────
    def log_event(
        self,
        video_source: str,
        frame_number: int,
        anomaly_score: float,
        classification: str,
        threshold: float,
        notes: str = "",
        session_id: Optional[int] = None
    ) -> int:
        """
        Inserts a single anomalous frame/clip event.
        Returns the new event id.
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO anomaly_events
                (session_id, timestamp, video_source, frame_number,
                 anomaly_score, classification, threshold_used, notes)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                session_id, now, video_source, int(frame_number),
                float(anomaly_score), classification, float(threshold), notes
            ))
            conn.commit()
            return cur.lastrowid

    # ──────────────────────────────────────────────
    # WRITE — SCORE TIME-SERIES (bulk)
    # ──────────────────────────────────────────────
    def log_score_timeseries(
        self,
        session_id: int,
        raw_scores: List[float],
        ema_scores: List[float],
        is_anomalous: List[bool]
    ):
        """
        Bulk-inserts the full clip-by-clip score trace for a session.
        """
        rows = [
            (session_id, i, float(raw_scores[i]), float(ema_scores[i]), int(is_anomalous[i]))
            for i in range(len(raw_scores))
        ]
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO score_timeseries (session_id, clip_index, raw_score, ema_score, is_anomalous)
                VALUES (?,?,?,?,?)
            """, rows)
            conn.commit()

    # ──────────────────────────────────────────────
    # READ — ANOMALY EVENTS (with rich filtering)
    # ──────────────────────────────────────────────
    def fetch_recent_events(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        video_source: Optional[str] = None,
        date_from: Optional[str] = None,    # "YYYY-MM-DD"
        date_to: Optional[str] = None,      # "YYYY-MM-DD"
        classification: Optional[str] = None,
        search_text: Optional[str] = None,
        order_by: str = "id DESC"
    ) -> List[Dict[str, Any]]:
        """
        Flexible query with score range, source, date range,
        full-text search on notes/source, and pagination.
        """
        clauses = []
        params: List[Any] = []

        if min_score is not None:
            clauses.append("anomaly_score >= ?")
            params.append(min_score)
        if max_score is not None:
            clauses.append("anomaly_score <= ?")
            params.append(max_score)
        if video_source:
            clauses.append("video_source LIKE ?")
            params.append(f"%{video_source}%")
        if date_from:
            clauses.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("timestamp <= ?")
            params.append(date_to + " 23:59:59")
        if classification:
            clauses.append("classification = ?")
            params.append(classification)
        if search_text:
            clauses.append("(notes LIKE ? OR video_source LIKE ?)")
            params.extend([f"%{search_text}%", f"%{search_text}%"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM anomaly_events {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def count_events(
        self,
        min_score: Optional[float] = None,
        video_source: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search_text: Optional[str] = None
    ) -> int:
        """Returns total count matching the same filters (for pagination)."""
        clauses, params = [], []
        if min_score is not None:
            clauses.append("anomaly_score >= ?"); params.append(min_score)
        if video_source:
            clauses.append("video_source LIKE ?"); params.append(f"%{video_source}%")
        if date_from:
            clauses.append("timestamp >= ?"); params.append(date_from)
        if date_to:
            clauses.append("timestamp <= ?"); params.append(date_to + " 23:59:59")
        if search_text:
            clauses.append("(notes LIKE ? OR video_source LIKE ?)"); params.extend([f"%{search_text}%"] * 2)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM anomaly_events {where}", params)
            return cur.fetchone()[0]

    # ──────────────────────────────────────────────
    # READ — SESSIONS
    # ──────────────────────────────────────────────
    def fetch_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        anomaly_only: bool = False,
        video_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches video analysis session records with optional filters."""
        clauses, params = [], []
        if anomaly_only:
            clauses.append("anomaly_detected = 1")
        if video_source:
            clauses.append("video_source LIKE ?"); params.append(f"%{video_source}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM video_sessions {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params
            )
            return [dict(r) for r in cur.fetchall()]

    def get_session_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single session by its ID."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM video_sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ──────────────────────────────────────────────
    # READ — SCORE TIME-SERIES
    # ──────────────────────────────────────────────
    def fetch_timeseries(self, session_id: int) -> List[Dict[str, Any]]:
        """Returns the full clip-by-clip score trace for a session."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM score_timeseries WHERE session_id = ? ORDER BY clip_index ASC",
                (session_id,)
            )
            return [dict(r) for r in cur.fetchall()]

    # ──────────────────────────────────────────────
    # READ — ANALYTICS
    # ──────────────────────────────────────────────
    def get_incident_analytics(self) -> Dict[str, Any]:
        """
        Aggregated statistics across all anomaly_events.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) as c FROM anomaly_events;")
            total_events = cur.fetchone()["c"]

            cur.execute("SELECT MAX(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone(); max_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("SELECT AVG(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone(); avg_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("SELECT MIN(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone(); min_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("""
                SELECT video_source, COUNT(*) as count
                FROM anomaly_events GROUP BY video_source
                ORDER BY count DESC LIMIT 5;
            """)
            top_sources = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT strftime('%Y-%m-%d', timestamp) as day, COUNT(*) as count
                FROM anomaly_events
                GROUP BY day ORDER BY day DESC LIMIT 30;
            """)
            daily = [dict(r) for r in cur.fetchall()]

            return {
                "total_events":            total_events,
                "max_recorded_score":      max_score,
                "average_recorded_score":  avg_score,
                "min_recorded_score":      min_score,
                "top_incident_sources":    top_sources,
                "daily_counts":            daily,
            }

    def get_session_analytics(self) -> Dict[str, Any]:
        """Aggregated statistics across all video_sessions."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as c FROM video_sessions;")
            total_sessions = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM video_sessions WHERE anomaly_detected=1;")
            alert_sessions = cur.fetchone()["c"]

            cur.execute("SELECT AVG(avg_score) as v FROM video_sessions;")
            row = cur.fetchone(); avg_score = float(row["v"]) if row and row["v"] else 0.0

            cur.execute("SELECT MAX(max_score) as v FROM video_sessions;")
            row = cur.fetchone(); peak_score = float(row["v"]) if row and row["v"] else 0.0

            return {
                "total_sessions":   total_sessions,
                "alert_sessions":   alert_sessions,
                "normal_sessions":  total_sessions - alert_sessions,
                "avg_score":        avg_score,
                "peak_score":       peak_score,
            }

    # ──────────────────────────────────────────────
    # MAINTENANCE
    # ──────────────────────────────────────────────
    def clear_logs(self):
        """Deletes all records from all tables (keeps schema)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM score_timeseries;")
            conn.execute("DELETE FROM anomaly_events;")
            conn.execute("DELETE FROM video_sessions;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('anomaly_events','video_sessions','score_timeseries');")
            conn.commit()

    def get_db_info(self) -> Dict[str, Any]:
        """Returns DB file path, size, and row counts per table."""
        import os
        size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        with self._get_connection() as conn:
            cur = conn.cursor()
            counts = {}
            for table in ("video_sessions", "anomaly_events", "score_timeseries"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]
        return {
            "path":        self.db_path,
            "size_bytes":  size_bytes,
            "size_kb":     round(size_bytes / 1024, 1),
            "row_counts":  counts,
        }
