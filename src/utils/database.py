import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from src.config import DB_PATH

class AnomalyDatabase:
    """
    Production-Grade SQLite Manager:
    - Thread-safe connections with timeout locks
    - WAL (Write-Ahead Logging) mode for concurrent high-speed writes
    - Indexed columns for fast query execution
    - Built-in incident analytics and audit logging
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
        finally:
            conn.close()


    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    video_source TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    anomaly_score REAL NOT NULL,
                    classification TEXT NOT NULL,
                    threshold_used REAL NOT NULL,
                    notes TEXT
                );
            """)
            # Create indices for fast query execution
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_events(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_source ON anomaly_events(video_source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_score ON anomaly_events(anomaly_score);")
            conn.commit()

    def log_event(
        self,
        video_source: str,
        frame_number: int,
        anomaly_score: float,
        classification: str,
        threshold: float,
        notes: str = ""
    ) -> int:
        """
        Inserts a new anomaly detection incident with thread safety.
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO anomaly_events 
                (timestamp, video_source, frame_number, anomaly_score, classification, threshold_used, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, video_source, frame_number, float(anomaly_score), classification, float(threshold), notes))
            conn.commit()
            return cursor.lastrowid

    def fetch_recent_events(self, limit: int = 50, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieves recent logged anomaly events with optional score filtering.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if min_score is not None:
                cursor.execute("""
                    SELECT * FROM anomaly_events 
                    WHERE anomaly_score >= ?
                    ORDER BY id DESC 
                    LIMIT ?
                """, (min_score, limit))
            else:
                cursor.execute("""
                    SELECT * FROM anomaly_events 
                    ORDER BY id DESC 
                    LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_incident_analytics(self) -> Dict[str, Any]:
        """
        Generates aggregated incident statistics and health analytics.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total_events FROM anomaly_events;")
            total_events = cursor.fetchone()["total_events"]

            cursor.execute("SELECT MAX(anomaly_score) as max_score FROM anomaly_events;")
            row_max = cursor.fetchone()
            max_score = row_max["max_score"] if row_max and row_max["max_score"] is not None else 0.0

            cursor.execute("SELECT AVG(anomaly_score) as avg_score FROM anomaly_events;")
            row_avg = cursor.fetchone()
            avg_score = row_avg["avg_score"] if row_avg and row_avg["avg_score"] is not None else 0.0

            cursor.execute("SELECT video_source, COUNT(*) as count FROM anomaly_events GROUP BY video_source ORDER BY count DESC LIMIT 5;")
            top_sources = [dict(r) for r in cursor.fetchall()]

            return {
                "total_events": total_events,
                "max_recorded_score": float(max_score),
                "average_recorded_score": float(avg_score),
                "top_incident_sources": top_sources
            }

    def clear_logs(self):
        """
        Clears all recorded anomaly history.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM anomaly_events;")
            conn.commit()
