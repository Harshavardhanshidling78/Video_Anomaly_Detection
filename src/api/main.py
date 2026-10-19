import os
import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
import torch

from src.config import ANOMALY_THRESHOLD, CHECKPOINT_DIR, DEVICE
from src.pipeline.detector import VideoAnomalyDetector
from src.utils.database import AnomalyDatabase

# Fix Windows Proactor ConnectionResetError [WinError 10054] on socket disconnect
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Initialize FastAPI App
app = FastAPI(
    title="Video Anomaly Detection REST API",
    description="Production REST API for automated CCTV video anomaly detection using ViT / MobileNet + LSTM/GRU",
    version="1.0.0"
)


# Global Instance Management
ckpt_path = CHECKPOINT_DIR / "best_model.pth"
model_file = str(ckpt_path) if ckpt_path.exists() else None
detector = VideoAnomalyDetector(model_path=model_file, spatial_mode="fast")
db = AnomalyDatabase()

# Pydantic Schemas
class DetectionResponse(BaseModel):
    status: str
    video_name: str
    total_frames: int
    total_clips: int
    max_anomaly_score: float
    avg_anomaly_score: float
    anomaly_detected: bool
    threshold_used: float
    scores: List[float]

class HealthResponse(BaseModel):
    status: str
    device: str
    model_checkpoint_loaded: bool
    total_logged_incidents: int

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    """
    Returns system health telemetry and database status.
    """
    analytics = db.get_incident_analytics()
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_checkpoint_loaded": model_file is not None and Path(model_file).exists(),
        "total_logged_incidents": analytics["total_events"]
    }

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_anomaly_file(
    file: UploadFile = File(...),
    threshold: float = Query(ANOMALY_THRESHOLD, ge=0.0, le=1.0),
    spatial_mode: str = Query("fast", pattern="^(fast|vit)$")
):
    """
    Upload a video file (MP4, AVI, MOV) for automated anomaly score detection.
    Returns structured JSON with frame-by-frame scores and incident flags.
    """
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov, .mkv")

    # Save uploaded file to temp directory
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        content = await file.read()
        tfile.write(content)
        tfile.close()

        detector.set_threshold(threshold)
        detector.set_spatial_mode(spatial_mode)

        results = detector.process_video(tfile.name, log_to_db=True)

        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return {
            "status": "success",
            "video_name": file.filename,
            "total_frames": results["total_frames"],
            "total_clips": results["total_clips"],
            "max_anomaly_score": results["max_score"],
            "avg_anomaly_score": results["avg_score"],
            "anomaly_detected": results["anomaly_detected"],
            "threshold_used": threshold,
            "scores": [float(s) for s in results["scores"]]
        }
    finally:
        if os.path.exists(tfile.name):
            try:
                os.remove(tfile.name)
            except Exception:
                pass

@app.get("/api/v1/incidents")
def get_incidents(
    limit: int = Query(50, ge=1, le=500),
    min_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    video_source: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    """
    Retrieves historical logged anomaly incidents from SQLite database with flexible filters.
    """
    events = db.fetch_recent_events(
        limit=limit,
        min_score=min_score,
        max_score=max_score,
        video_source=video_source,
        search_text=search
    )
    return {
        "status": "success",
        "count": len(events),
        "incidents": events
    }

@app.get("/api/v1/sessions")
def get_sessions(
    limit: int = Query(50, ge=1, le=500),
    anomaly_only: bool = Query(False),
    video_source: Optional[str] = Query(None)
):
    """
    Retrieves full video analysis session summaries from SQLite database.
    """
    sessions = db.fetch_sessions(limit=limit, anomaly_only=anomaly_only, video_source=video_source)
    return {
        "status": "success",
        "count": len(sessions),
        "sessions": sessions
    }

@app.get("/api/v1/sessions/{session_id}")
def get_session_detail(session_id: int):
    """
    Retrieves single session details and full clip-by-clip score timeseries.
    """
    session = db.get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    timeseries = db.fetch_timeseries(session_id)
    return {
        "status": "success",
        "session": session,
        "timeseries_count": len(timeseries),
        "timeseries": timeseries
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
