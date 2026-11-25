import os
import sys
import datetime
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from src.config import DB_PATH, DATABASE_URL

# Optional PostgreSQL driver
try:
    import psycopg2
    from psycopg2 import pool, extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class AnomalyDatabase:
    """
    Enterprise Dual-Backend Database Manager for Video Anomaly Detection.
    Supports both PostgreSQL (production) and SQLite (local/embedded development).

    Schema:
        video_sessions   — one row per full video analysis run
        anomaly_events   — one row per anomalous clip/frame detected
        score_timeseries — full clip-by-clip score trace for every session

    Features:
        - Automatic backend selection: Uses PostgreSQL when DATABASE_URL is set,
          with graceful fallback to SQLite if PostgreSQL is unavailable or unconfigured.
        - Connection pooling for high-concurrency production deployments.
        - Full search, date-range, score-range filtering with pagination.
        - Session-level analytics and aggregated KPIs.
        - Batch inserts for high-throughput score time-series telemetry.
    """

    def __init__(self, db_path: Path = DB_PATH, database_url: Optional[str] = DATABASE_URL):
        self.db_path = str(db_path)
        self.database_url = database_url
        self.is_postgres = False
        self._pg_pool = None

        if self.database_url and PSYCOPG2_AVAILABLE:
            try:
                # Initialize PostgreSQL connection pool (1-10 connections)
                self._pg_pool = pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.database_url,
                    connect_timeout=1
                )
                self.is_postgres = True
                print("[Info] AnomalyDatabase connected to PostgreSQL successfully.")
            except Exception as e:
                print(f"[Warning] Failed to connect to PostgreSQL ({e}). Falling back to SQLite.")
                self.is_postgres = False
                self._pg_pool = None
        elif self.database_url and not PSYCOPG2_AVAILABLE:
            print("[Warning] DATABASE_URL provided but psycopg2 is not installed. Falling back to SQLite.")

        self._init_db()

    # ──────────────────────────────────────────────
    # CONNECTIONS & CONTEXT MANAGERS
    # ──────────────────────────────────────────────
    @contextmanager
    def _get_connection(self):
        if self.is_postgres and self._pg_pool:
            conn = self._pg_pool.getconn()
            try:
                yield conn
            finally:
                self._pg_pool.putconn(conn)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                yield conn
            finally:
                conn.close()

    def _ph(self) -> str:
        """Returns placeholder for SQL queries (%s for PostgreSQL, ? for SQLite)."""
        return "%s" if self.is_postgres else "?"

    # ──────────────────────────────────────────────
    # SCHEMA INIT
    # ──────────────────────────────────────────────
    def _init_db(self):
        with self._get_connection() as conn:
            cur = conn.cursor()

            if self.is_postgres:
                # PostgreSQL schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS video_sessions (
                        id              SERIAL PRIMARY KEY,
                        timestamp       VARCHAR(32) NOT NULL,
                        video_source    TEXT        NOT NULL,
                        spatial_mode    VARCHAR(32) NOT NULL DEFAULT 'vit',
                        threshold_used  DOUBLE PRECISION NOT NULL,
                        total_frames    INTEGER NOT NULL DEFAULT 0,
                        total_clips     INTEGER NOT NULL DEFAULT 0,
                        anomaly_clips   INTEGER NOT NULL DEFAULT 0,
                        max_score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        avg_score       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                        anomaly_detected INTEGER NOT NULL DEFAULT 0,
                        use_ema         INTEGER NOT NULL DEFAULT 1,
                        use_adaptive    INTEGER NOT NULL DEFAULT 0,
                        notes           TEXT    DEFAULT ''
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS anomaly_events (
                        id              SERIAL PRIMARY KEY,
                        session_id      INTEGER REFERENCES video_sessions(id) ON DELETE SET NULL,
                        timestamp       VARCHAR(32) NOT NULL,
                        video_source    TEXT        NOT NULL,
                        frame_number    INTEGER     NOT NULL,
                        anomaly_score   DOUBLE PRECISION NOT NULL,
                        classification  VARCHAR(64) NOT NULL,
                        threshold_used  DOUBLE PRECISION NOT NULL,
                        notes           TEXT        DEFAULT ''
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS score_timeseries (
                        id          SERIAL PRIMARY KEY,
                        session_id  INTEGER NOT NULL REFERENCES video_sessions(id) ON DELETE CASCADE,
                        clip_index  INTEGER NOT NULL,
                        raw_score   DOUBLE PRECISION NOT NULL,
                        ema_score   DOUBLE PRECISION NOT NULL,
                        is_anomalous INTEGER NOT NULL DEFAULT 0
                    );
                """)

                # PostgreSQL Indices
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

            else:
                # SQLite schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS video_sessions (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp       TEXT    NOT NULL,
                        video_source    TEXT    NOT NULL,
                        spatial_mode    TEXT    NOT NULL DEFAULT 'vit',
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

                # Check if session_id exists in anomaly_events
                cur.execute("PRAGMA table_info(anomaly_events);")
                existing_cols = {row["name"] for row in cur.fetchall()}
                if "session_id" not in existing_cols:
                    cur.execute("ALTER TABLE anomaly_events ADD COLUMN session_id INTEGER REFERENCES video_sessions(id);")

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
        ph = self._ph()
        with self._get_connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute(f"""
                    INSERT INTO video_sessions
                    (timestamp, video_source, spatial_mode, threshold_used,
                     total_frames, total_clips, anomaly_clips,
                     max_score, avg_score, anomaly_detected,
                     use_ema, use_adaptive, notes)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                    RETURNING id;
                """, (
                    now, video_source, spatial_mode, float(threshold),
                    int(total_frames), int(total_clips), int(anomaly_clips),
                    float(max_score), float(avg_score), int(anomaly_detected),
                    int(use_ema), int(use_adaptive), notes
                ))
                new_id = cur.fetchone()[0]
            else:
                cur.execute(f"""
                    INSERT INTO video_sessions
                    (timestamp, video_source, spatial_mode, threshold_used,
                     total_frames, total_clips, anomaly_clips,
                     max_score, avg_score, anomaly_detected,
                     use_ema, use_adaptive, notes)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph});
                """, (
                    now, video_source, spatial_mode, float(threshold),
                    int(total_frames), int(total_clips), int(anomaly_clips),
                    float(max_score), float(avg_score), int(anomaly_detected),
                    int(use_ema), int(use_adaptive), notes
                ))
                new_id = cur.lastrowid
            conn.commit()
            return new_id

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
        ph = self._ph()
        with self._get_connection() as conn:
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute(f"""
                    INSERT INTO anomaly_events
                    (session_id, timestamp, video_source, frame_number,
                     anomaly_score, classification, threshold_used, notes)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
                    RETURNING id;
                """, (
                    session_id, now, video_source, int(frame_number),
                    float(anomaly_score), classification, float(threshold), notes
                ))
                new_id = cur.fetchone()[0]
            else:
                cur.execute(f"""
                    INSERT INTO anomaly_events
                    (session_id, timestamp, video_source, frame_number,
                     anomaly_score, classification, threshold_used, notes)
                    VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph});
                """, (
                    session_id, now, video_source, int(frame_number),
                    float(anomaly_score), classification, float(threshold), notes
                ))
                new_id = cur.lastrowid
            conn.commit()
            return new_id

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
        ph = self._ph()
        with self._get_connection() as conn:
            cur = conn.cursor()
            sql = f"""
                INSERT INTO score_timeseries (session_id, clip_index, raw_score, ema_score, is_anomalous)
                VALUES ({ph},{ph},{ph},{ph},{ph})
            """
            cur.executemany(sql, rows)
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
        ph = self._ph()
        clauses = []
        params: List[Any] = []

        if min_score is not None:
            clauses.append(f"anomaly_score >= {ph}")
            params.append(min_score)
        if max_score is not None:
            clauses.append(f"anomaly_score <= {ph}")
            params.append(max_score)
        if video_source:
            clauses.append(f"video_source LIKE {ph}")
            params.append(f"%{video_source}%")
        if date_from:
            clauses.append(f"timestamp >= {ph}")
            params.append(date_from)
        if date_to:
            clauses.append(f"timestamp <= {ph}")
            params.append(date_to + " 23:59:59")
        if classification:
            clauses.append(f"classification = {ph}")
            params.append(classification)
        if search_text:
            clauses.append(f"(notes LIKE {ph} OR video_source LIKE {ph})")
            params.extend([f"%{search_text}%", f"%{search_text}%"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM anomaly_events {where} ORDER BY {order_by} LIMIT {ph} OFFSET {ph}"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
            else:
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
        ph = self._ph()
        clauses, params = [], []
        if min_score is not None:
            clauses.append(f"anomaly_score >= {ph}")
            params.append(min_score)
        if video_source:
            clauses.append(f"video_source LIKE {ph}")
            params.append(f"%{video_source}%")
        if date_from:
            clauses.append(f"timestamp >= {ph}")
            params.append(date_from)
        if date_to:
            clauses.append(f"timestamp <= {ph}")
            params.append(date_to + " 23:59:59")
        if search_text:
            clauses.append(f"(notes LIKE {ph} OR video_source LIKE {ph})")
            params.extend([f"%{search_text}%", f"%{search_text}%"])

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
        ph = self._ph()
        clauses, params = [], []
        if anomaly_only:
            clauses.append("anomaly_detected = 1")
        if video_source:
            clauses.append(f"video_source LIKE {ph}")
            params.append(f"%{video_source}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([limit, offset])

        with self._get_connection() as conn:
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute(f"SELECT * FROM video_sessions {where} ORDER BY id DESC LIMIT {ph} OFFSET {ph}", params)
                return [dict(r) for r in cur.fetchall()]
            else:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM video_sessions {where} ORDER BY id DESC LIMIT {ph} OFFSET {ph}", params)
                return [dict(r) for r in cur.fetchall()]

    def get_session_by_id(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Fetches a single session by its ID."""
        ph = self._ph()
        with self._get_connection() as conn:
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute(f"SELECT * FROM video_sessions WHERE id = {ph}", (session_id,))
                row = cur.fetchone()
                return dict(row) if row else None
            else:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM video_sessions WHERE id = {ph}", (session_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    # ──────────────────────────────────────────────
    # READ — SCORE TIME-SERIES
    # ──────────────────────────────────────────────
    def fetch_timeseries(self, session_id: int) -> List[Dict[str, Any]]:
        """Returns the full clip-by-clip score trace for a session."""
        ph = self._ph()
        with self._get_connection() as conn:
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
                cur.execute(f"SELECT * FROM score_timeseries WHERE session_id = {ph} ORDER BY clip_index ASC", (session_id,))
                return [dict(r) for r in cur.fetchall()]
            else:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM score_timeseries WHERE session_id = {ph} ORDER BY clip_index ASC", (session_id,))
                return [dict(r) for r in cur.fetchall()]

    # ──────────────────────────────────────────────
    # READ — ANALYTICS
    # ──────────────────────────────────────────────
    def get_incident_analytics(self) -> Dict[str, Any]:
        """Aggregated statistics across all anomaly_events."""
        with self._get_connection() as conn:
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            else:
                cur = conn.cursor()

            cur.execute("SELECT COUNT(*) as c FROM anomaly_events;")
            row = cur.fetchone()
            total_events = row["c"] if row else 0

            cur.execute("SELECT MAX(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone()
            max_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("SELECT AVG(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone()
            avg_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("SELECT MIN(anomaly_score) as v FROM anomaly_events;")
            row = cur.fetchone()
            min_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("""
                SELECT video_source, COUNT(*) as count
                FROM anomaly_events GROUP BY video_source
                ORDER BY count DESC LIMIT 5;
            """)
            top_sources = [dict(r) for r in cur.fetchall()]

            # Date substring works across both SQLite and Postgres
            if self.is_postgres:
                cur.execute("""
                    SELECT LEFT(timestamp, 10) as day, COUNT(*) as count
                    FROM anomaly_events
                    GROUP BY LEFT(timestamp, 10) ORDER BY day DESC LIMIT 30;
                """)
            else:
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
            if self.is_postgres:
                cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            else:
                cur = conn.cursor()

            cur.execute("SELECT COUNT(*) as c FROM video_sessions;")
            row = cur.fetchone()
            total_sessions = row["c"] if row else 0

            cur.execute("SELECT COUNT(*) as c FROM video_sessions WHERE anomaly_detected=1;")
            row = cur.fetchone()
            alert_sessions = row["c"] if row else 0

            cur.execute("SELECT AVG(avg_score) as v FROM video_sessions;")
            row = cur.fetchone()
            avg_score = float(row["v"]) if row and row["v"] is not None else 0.0

            cur.execute("SELECT MAX(max_score) as v FROM video_sessions;")
            row = cur.fetchone()
            peak_score = float(row["v"]) if row and row["v"] is not None else 0.0

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
            cur = conn.cursor()
            if self.is_postgres:
                cur.execute("TRUNCATE TABLE score_timeseries, anomaly_events, video_sessions RESTART IDENTITY CASCADE;")
            else:
                cur.execute("DELETE FROM score_timeseries;")
                cur.execute("DELETE FROM anomaly_events;")
                cur.execute("DELETE FROM video_sessions;")
                cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('anomaly_events','video_sessions','score_timeseries');")
            conn.commit()

    def get_db_info(self) -> Dict[str, Any]:
        """Returns DB type, location/connection info, and row counts per table."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            counts = {}
            for table in ("video_sessions", "anomaly_events", "score_timeseries"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]

            if self.is_postgres:
                cur.execute("SELECT pg_database_size(current_database());")
                size_bytes = cur.fetchone()[0]
                target = self.database_url.split("@")[-1] if "@" in (self.database_url or "") else "PostgreSQL Server"
                return {
                    "type":        "PostgreSQL",
                    "path":        f"PostgreSQL ({target})",
                    "target":      target,
                    "size_bytes":  size_bytes,
                    "size_kb":     round(size_bytes / 1024, 1),
                    "row_counts":  counts,
                }
            else:
                size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                return {
                    "type":        "SQLite",
                    "path":        self.db_path,
                    "target":      self.db_path,
                    "size_bytes":  size_bytes,
                    "size_kb":     round(size_bytes / 1024, 1),
                    "row_counts":  counts,
                }
