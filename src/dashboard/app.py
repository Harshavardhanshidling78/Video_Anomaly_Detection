import sys
import os
import tempfile
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cv2

from src.config import (
    CHECKPOINT_DIR, DATA_DIR, DEMO_DIR,
    ANOMALY_THRESHOLD, DEVICE
)
from src.pipeline.detector import VideoAnomalyDetector
from src.utils.database import AnomalyDatabase

# Page Configuration
st.set_page_config(
    page_title="Video Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .status-normal {
        color: #00d26a;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-alert {
        color: #f8312f;
        font-weight: 700;
        font-size: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector(model_path: str, spatial_mode: str, threshold: float):
    return VideoAnomalyDetector(
        model_path=model_path,
        spatial_mode=spatial_mode,
        threshold=threshold
    )

@st.cache_resource
def get_database():
    return AnomalyDatabase()

db = get_database()

# Sidebar: Controls & Configuration
st.sidebar.title("🛡️ Surveillance Control")
st.sidebar.caption(f"Hardware Device: **{DEVICE.upper()}**")

# Model configuration
st.sidebar.subheader("Model Configuration")
spatial_mode = st.sidebar.selectbox(
    "Spatial Backbone",
    options=["vit", "fast"],
    format_func=lambda x: "Vision Transformer (ViT-B/16)" if x == "vit" else "MobileNetV2 (Fast CPU)",
    index=0
)

# Checkpoint status
ckpt_path = CHECKPOINT_DIR / "best_model.pth"
has_ckpt = ckpt_path.exists()
if has_ckpt:
    st.sidebar.success("Trained Checkpoint Loaded", icon=":material/check_circle:")
else:
    st.sidebar.warning("Using Initialized Weights (Training in progress)", icon=":material/hourglass_top:")

model_file = str(ckpt_path) if has_ckpt else None

# Threshold & Filtering controls
st.sidebar.subheader("Detection Controls")
threshold = st.sidebar.slider(
    "Anomaly Decision Threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(ANOMALY_THRESHOLD),
    step=0.05,
    help="Clips with anomaly score at or above this threshold trigger an alert incident."
)

use_ema = st.sidebar.toggle(
    "Exponential Moving Average (EMA) Smoothing",
    value=True,
    help="Smooths out single-frame noise flicker across sequence clips."
)

use_adaptive = st.sidebar.toggle(
    "Dynamic Adaptive Thresholding (μ + 1.5σ)",
    value=False,
    help="Dynamically adapts alert sensitivity to the video's baseline activity."
)

# Instantiate detector
detector = get_detector(model_file, spatial_mode, threshold)
detector.set_threshold(threshold)
detector.set_spatial_mode(spatial_mode)

# Video Input Source
st.sidebar.subheader("Video Feed Selection")
feed_mode = st.sidebar.radio(
    "Select Source",
    options=["Upload Video File", "Demo Videos", "Training Dataset Samples"]
)

video_source_path = None
uploaded_video_name = ""

if feed_mode == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload surveillance clip (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_source_path = tfile.name
        uploaded_video_name = uploaded_file.name

elif feed_mode == "Demo Videos":
    demo_files = list(DEMO_DIR.glob("*.mp4"))
    if not demo_files:
        if st.sidebar.button("Generate Synthetic Demo Clips", icon=":material/video_camera_front:"):
            from scripts.generate_demo_data import create_demo_videos
            create_demo_videos()
            st.rerun()
        st.sidebar.info("Click above to generate demo test clips.")
    else:
        chosen_demo = st.sidebar.selectbox("Select Demo Video", [f.name for f in demo_files])
        video_source_path = str(DEMO_DIR / chosen_demo)
        uploaded_video_name = chosen_demo

elif feed_mode == "Training Dataset Samples":
    sample_type = st.sidebar.radio("Category", ["Normal", "Anomalous"], horizontal=True)
    sample_folder = DATA_DIR / "train" / sample_type.lower()
    if sample_folder.exists():
        sample_vids = list(sample_folder.glob("*.avi")) + list(sample_folder.glob("*.mp4"))
        if sample_vids:
            chosen_sample = st.sidebar.selectbox("Select Sample Video", [v.name for v in sample_vids[:25]])
            video_source_path = str(sample_folder / chosen_sample)
            uploaded_video_name = f"{sample_type}_{chosen_sample}"
        else:
            st.sidebar.warning(f"No {sample_type} videos found in dataset folder.")

# Main Interface Header
st.title("🛡️ Enterprise Video Anomaly Detection System")
st.caption("Automated surveillance monitoring powered by Vision Transformer (ViT-B/16) and Recurrent Temporal Modeling (LSTM)")

# Analysis Execution
if video_source_path:
    st.divider()
    run_col1, run_col2 = st.columns([1, 4])
    with run_col1:
        run_btn = st.button("🚀 Analyze Video Feed", type="primary")

    if run_btn:
        progress_bar = st.progress(0, text="Initializing video inspection pipeline...")

        def update_progress(pct, msg):
            progress_bar.progress(int(pct * 100), text=msg)

        with st.spinner("Processing video frames and running ViT + LSTM sequence model..."):
            results = detector.process_video(
                video_source_path,
                log_to_db=True,
                use_ema=use_ema,
                use_adaptive_threshold=use_adaptive,
                progress_callback=update_progress
            )

        progress_bar.empty()

        if "error" in results:
            st.error(f"Detection Error: {results['error']}")
        else:
            st.session_state["results"] = results
            st.session_state["video_name"] = uploaded_video_name
            st.success("Analysis Complete!", icon=":material/check_circle:")

    # Display Results if Available
    if "results" in st.session_state:
        res = st.session_state["results"]
        scores = res["scores"]
        raw_scores = res.get("raw_scores", scores)
        max_score = res["max_score"]
        avg_score = res["avg_score"]
        is_anom = res["anomaly_detected"]
        annotated_frames = res.get("annotated_frames", [])

        # Row 1: KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                status_label = "🚨 ANOMALY DETECTED" if is_anom else "✅ NORMAL ACTIVITY"
                st.metric("Detection Verdict", status_label)
        with m2:
            with st.container(border=True):
                st.metric("Peak Anomaly Score", f"{max_score:.3f}", delta=f"{max_score - threshold:+.2f} vs Thresh")
        with m3:
            with st.container(border=True):
                st.metric("Average Score", f"{avg_score:.3f}")
        with m4:
            with st.container(border=True):
                st.metric("Clips Evaluated", f"{res['total_clips']} clips", f"{res['total_frames']} frames")

        # Row 2: Dual Panel - Visual Telemetry & Temporal Graph
        col_video, col_chart = st.columns([1, 1])

        with col_video:
            st.subheader("📹 Real-Time Frame Telemetry")
            if annotated_frames:
                frame_idx = st.slider(
                    "Scrub Sequence Frame",
                    min_value=0,
                    max_value=len(annotated_frames) - 1,
                    value=0,
                    step=1
                )
                curr_frame = annotated_frames[frame_idx]
                curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                st.image(curr_frame_rgb, caption=f"Sequence Frame {frame_idx + 1} of {len(annotated_frames)}")
            else:
                st.info("No annotated preview frames available.")

        with col_chart:
            st.subheader("📈 Anomaly Score Over Time")
            clip_indices = list(range(len(scores)))
            fig = go.Figure()

            # Raw Scores trace
            fig.add_trace(go.Scatter(
                x=clip_indices,
                y=raw_scores,
                mode="lines",
                name="Raw Model Output",
                line=dict(color="rgba(150, 150, 150, 0.5)", width=1, dash="dot")
            ))

            # Filtered / Smoothed trace
            fig.add_trace(go.Scatter(
                x=clip_indices,
                y=scores,
                mode="lines+markers",
                name="EMA Filtered Score",
                line=dict(color="#00d26a" if not is_anom else "#f8312f", width=2.5),
                marker=dict(size=4)
            ))

            # Threshold line
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#ffcc00",
                annotation_text=f"Alert Threshold ({threshold:.2f})",
                annotation_position="top left"
            )

            fig.update_layout(
                xaxis_title="Temporal Sequence Clip (T=16)",
                yaxis_title="Anomaly Probability",
                yaxis_range=[-0.05, 1.05],
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, width="stretch")

else:
    st.info("👈 Please select or upload a video clip in the sidebar to begin analysis.", icon=":material/info:")

# Row 3: Historical Incident Audit Trail
st.divider()
st.subheader("📋 SQLite Incident Audit Trail & Analytics")

analytics = db.get_incident_analytics()
a1, a2, a3 = st.columns(3)
with a1:
    with st.container(border=True):
        st.metric("Total Logged Incidents", analytics["total_events"])
with a2:
    with st.container(border=True):
        st.metric("Max Recorded Anomaly Score", f"{analytics['max_recorded_score']:.3f}")
with a3:
    with st.container(border=True):
        st.metric("Historical Average Score", f"{analytics['average_recorded_score']:.3f}")

# Recent Incidents Table
recent_events = db.fetch_recent_events(limit=25)
if recent_events:
    df_events = pd.DataFrame(recent_events)
    st.dataframe(
        df_events[["id", "timestamp", "video_source", "frame_number", "anomaly_score", "classification", "threshold_used", "notes"]],
        width="stretch",
        hide_index=True
    )
else:
    st.info("No incident events logged in database yet.")

col_audit1, col_audit2 = st.columns([1, 4])
with col_audit1:
    if st.button("🗑️ Clear Incident Logs", help="Clears all records in anomaly_events.db"):
        db.clear_logs()
        st.success("Audit records cleared!")
        st.rerun()
