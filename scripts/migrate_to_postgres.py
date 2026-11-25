"""
Database Migration Utility: SQLite to PostgreSQL
Migrates historical video analysis sessions, anomaly events, and score time-series
from SQLite (data/anomaly_events.db) to a target PostgreSQL database.

Usage:
    python scripts/migrate_to_postgres.py
    python scripts/migrate_to_postgres.py --pg_url "postgresql://user:pass@localhost:5432/anomaly_db"
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DB_PATH, DATABASE_URL

try:
    import psycopg2
    from psycopg2 import extras
except ImportError:
    print("[Error] psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def migrate(sqlite_path: Path, pg_url: str):
    if not sqlite_path.exists():
        print(f"[Error] Source SQLite database not found at {sqlite_path}")
        return

    print(f"Connecting to SQLite: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sq_cur = sqlite_conn.cursor()

    print(f"Connecting to PostgreSQL: {pg_url.split('@')[-1] if '@' in pg_url else 'PostgreSQL Server'}")
    try:
        pg_conn = psycopg2.connect(pg_url)
    except Exception as e:
        print(f"[Error] Failed to connect to PostgreSQL: {e}")
        return

    pg_cur = pg_conn.cursor()

    # Step 1: Initialize Schema on PostgreSQL
    from src.utils.database import AnomalyDatabase
    db_helper = AnomalyDatabase(database_url=pg_url)

    # Step 2: Migrate video_sessions
    sq_cur.execute("SELECT * FROM video_sessions ORDER BY id ASC")
    sessions = sq_cur.fetchall()
    print(f"Found {len(sessions)} video_sessions in SQLite.")

    session_id_map = {}
    for row in sessions:
        r = dict(row)
        old_id = r["id"]
        pg_cur.execute("""
            INSERT INTO video_sessions 
            (timestamp, video_source, spatial_mode, threshold_used,
             total_frames, total_clips, anomaly_clips,
             max_score, avg_score, anomaly_detected,
             use_ema, use_adaptive, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            r["timestamp"], r["video_source"], r.get("spatial_mode", "vit"),
            float(r["threshold_used"]), int(r["total_frames"]), int(r["total_clips"]),
            int(r["anomaly_clips"]), float(r["max_score"]), float(r["avg_score"]),
            int(r["anomaly_detected"]), int(r.get("use_ema", 1)), int(r.get("use_adaptive", 0)),
            r.get("notes", "")
        ))
        new_id = pg_cur.fetchone()[0]
        session_id_map[old_id] = new_id

    pg_conn.commit()
    print(f"Migrated {len(session_id_map)} video_sessions successfully.")

    # Step 3: Migrate anomaly_events
    sq_cur.execute("SELECT * FROM anomaly_events ORDER BY id ASC")
    events = sq_cur.fetchall()
    print(f"Found {len(events)} anomaly_events in SQLite.")

    event_rows = []
    for row in events:
        r = dict(row)
        old_session_id = r.get("session_id")
        mapped_session_id = session_id_map.get(old_session_id) if old_session_id else None
        event_rows.append((
            mapped_session_id,
            r["timestamp"],
            r["video_source"],
            int(r["frame_number"]),
            float(r["anomaly_score"]),
            r["classification"],
            float(r["threshold_used"]),
            r.get("notes", "")
        ))

    if event_rows:
        extras.execute_batch(pg_cur, """
            INSERT INTO anomaly_events
            (session_id, timestamp, video_source, frame_number,
             anomaly_score, classification, threshold_used, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, event_rows, page_size=500)
        pg_conn.commit()
    print(f"Migrated {len(event_rows)} anomaly_events successfully.")

    # Step 4: Migrate score_timeseries
    sq_cur.execute("SELECT * FROM score_timeseries ORDER BY id ASC")
    timeseries = sq_cur.fetchall()
    print(f"Found {len(timeseries)} score_timeseries rows in SQLite.")

    ts_rows = []
    for row in timeseries:
        r = dict(row)
        old_session_id = r.get("session_id")
        mapped_session_id = session_id_map.get(old_session_id)
        if mapped_session_id:
            ts_rows.append((
                mapped_session_id,
                int(r["clip_index"]),
                float(r["raw_score"]),
                float(r["ema_score"]),
                int(r.get("is_anomalous", 0))
            ))

    if ts_rows:
        extras.execute_batch(pg_cur, """
            INSERT INTO score_timeseries
            (session_id, clip_index, raw_score, ema_score, is_anomalous)
            VALUES (%s, %s, %s, %s, %s);
        """, ts_rows, page_size=1000)
        pg_conn.commit()
    print(f"Migrated {len(ts_rows)} score_timeseries rows successfully.")

    sqlite_conn.close()
    pg_conn.close()
    print("Migration completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL")
    parser.add_argument("--sqlite_path", type=Path, default=DB_PATH, help="Path to SQLite .db file")
    parser.add_argument("--pg_url", type=str, default=DATABASE_URL, help="PostgreSQL connection string")
    args = parser.parse_args()

    if not args.pg_url:
        print("[Error] No PostgreSQL URL provided. Set DATABASE_URL in .env or pass --pg_url.")
        sys.exit(1)

    migrate(args.sqlite_path, args.pg_url)
