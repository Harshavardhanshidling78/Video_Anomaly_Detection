import sys
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2

from src.config import (
    CHECKPOINT_DIR, DATA_DIR, DEMO_DIR,
    ANOMALY_THRESHOLD, DEVICE
)
from src.pipeline.detector import VideoAnomalyDetector
from src.utils.database import AnomalyDatabase

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — modern dark glassmorphism theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0f1a 0%, #111320 60%, #0d1525 100%);
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 17, 32, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.10) 50%, rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 28px 36px 22px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.12), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #c4b5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 6px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.30);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #a5b4fc;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge.green  { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.3); color: #6ee7b7; }
.badge.cyan   { background: rgba(6,182,212,0.12);  border-color: rgba(6,182,212,0.3);  color: #67e8f9; }
.badge.orange { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.3); color: #fcd34d; }

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    background: rgba(17,19,40,0.85);
    border: 1px solid rgba(99,102,241,0.20);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.kpi-card:hover { border-color: rgba(99,102,241,0.45); }
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.blue::after  { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.green::after { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.red::after   { background: linear-gradient(90deg, #ef4444, #f97316); }
.kpi-card.cyan::after  { background: linear-gradient(90deg, #06b6d4, #0ea5e9); }
.kpi-label { font-size: 0.72rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-value { font-size: 1.65rem; font-weight: 800; color: #f1f5f9; margin: 4px 0 2px; line-height: 1; }
.kpi-sub   { font-size: 0.75rem; color: #64748b; }
.kpi-alert-value { color: #f87171 !important; }
.kpi-safe-value  { color: #34d399 !important; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.section-header h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c7d2fe;
    margin: 0;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 8px rgba(99,102,241,0.6);
}

/* ── Alert banner ── */
.alert-box {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.normal-box {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.30);
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.alert-icon { font-size: 1.4rem; }
.alert-text-title { font-weight: 700; font-size: 0.95rem; }
.alert-text-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }

/* ── Timeline row ── */
.timeline-clip {
    display: inline-block;
    width: 12px; height: 28px;
    border-radius: 3px;
    margin: 1px;
    cursor: pointer;
    transition: transform 0.15s;
}
.timeline-clip:hover { transform: scaleY(1.4); }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(15,17,32,0.6) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(99,102,241,0.25) !important;
    color: #a5b4fc !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(15,17,32,0.5); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.6); }

/* ── Streamlit overrides ── */
[data-testid="stMetric"] { background: transparent !important; }
div.stSpinner > div { border-top-color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _plotly_chart(fig, height=None, **kwargs):
    """Wrapper: always use width='stretch' (new API) + suppress displayModeBar."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kwargs)


def _safe_bar_colors(values, low_color="#6366f1", high_color="#8b5cf6"):
    """Return per-bar colors via interpolation — avoids colorscale crash on 1-value arrays."""
    arr = list(values)
    mn, mx = min(arr), max(arr)
    if mx == mn:
        return [high_color] * len(arr)
    # Linear interpolation between low and high hex colors (simple purple palette)
    palette = [
        "rgba(99,102,241,0.30)", "rgba(99,102,241,0.45)", "rgba(99,102,241,0.60)",
        "rgba(120,100,241,0.70)", "rgba(139,92,246,0.80)", "#8b5cf6"
    ]
    colors = []
    for v in arr:
        idx = int((v - mn) / (mx - mn) * (len(palette) - 1))
        colors.append(palette[idx])
    return colors


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 20px;">
        <div style="font-size:2.2rem;">🛡️</div>
        <div style="font-size:1.1rem; font-weight:800; color:#a5b4fc; margin-top:4px;">AnomalyShield</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Video Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    device_label = "🟢 GPU (CUDA)" if DEVICE == "cuda" else "🟡 CPU Mode"
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    has_ckpt = ckpt_path.exists()
    ckpt_label = "✅ Checkpoint Loaded" if has_ckpt else "⚠️ Initialized Weights"
    model_file = str(ckpt_path) if has_ckpt else None

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                border-radius:10px; padding:12px 14px; margin-bottom:14px; font-size:0.8rem;">
        <div style="color:#94a3b8; margin-bottom:6px; font-weight:600; font-size:0.7rem;
                    text-transform:uppercase; letter-spacing:0.6px;">System Status</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#64748b;">Hardware</span>
            <span style="color:#a5b4fc; font-weight:700;">{device_label}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#64748b;">Model</span>
            <span style="color:#{'6ee7b7' if has_ckpt else 'fcd34d'}; font-weight:600;">{ckpt_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🔧 Model Configuration")
    spatial_mode = st.selectbox(
        "Spatial Backbone",
        options=["fast", "vit"],
        format_func=lambda x: "⚡ MobileNetV2 (Fast CPU)" if x == "fast" else "🔬 ViT-B/16 (Accurate)",
        index=0,
        help="ViT gives best accuracy; MobileNetV2 runs faster on CPU."
    )

    st.markdown("##### 🎯 Detection Controls")
    threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.10, max_value=0.95,
        value=float(ANOMALY_THRESHOLD), step=0.05,
        help="Clips scoring >= this value trigger an alert."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        use_ema = st.toggle("EMA Smoothing", value=True,
                            help="Exponential Moving Average noise reduction.")
    with col_b:
        use_adaptive = st.toggle("Adaptive Thresh", value=False,
                                 help="Dynamic threshold: mu + 1.5 * sigma.")

    st.markdown("##### 📹 Video Source")
    feed_mode = st.radio(
        "Source",
        options=["📂 Upload File", "🎬 Demo Videos", "📊 Dataset Samples"],
        label_visibility="collapsed"
    )

    video_source_path = None
    uploaded_video_name = ""

    if feed_mode == "📂 Upload File":
        uploaded_file = st.file_uploader(
            "Drop or click to upload",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            video_source_path = tfile.name
            uploaded_video_name = uploaded_file.name
            st.success(f"📎 `{uploaded_file.name}` ready")

    elif feed_mode == "🎬 Demo Videos":
        demo_files = list(DEMO_DIR.glob("*.mp4"))
        if not demo_files:
            if st.button("⚙️ Generate Demo Clips", use_container_width=True):
                from scripts.generate_demo_data import create_demo_videos
                create_demo_videos()
                st.rerun()
            st.caption("No demo clips found. Click above to generate.")
        else:
            chosen_demo = st.selectbox("Select clip", [f.name for f in demo_files])
            video_source_path = str(DEMO_DIR / chosen_demo)
            uploaded_video_name = chosen_demo

    elif feed_mode == "📊 Dataset Samples":
        sample_type = st.radio("Category", ["Normal", "Anomalous"], horizontal=True)
        sample_folder = DATA_DIR / "train" / sample_type.lower()
        if sample_folder.exists():
            sample_vids = list(sample_folder.glob("*.avi")) + list(sample_folder.glob("*.mp4"))
            if sample_vids:
                chosen_sample = st.selectbox("Select video", [v.name for v in sample_vids[:30]])
                video_source_path = str(sample_folder / chosen_sample)
                uploaded_video_name = f"{sample_type}_{chosen_sample}"
            else:
                st.warning(f"No {sample_type} videos found.")

    st.divider()
    st.markdown("""
    <div style="font-size:0.68rem; color:#334155; text-align:center; padding:4px 0;">
        AnomalyShield v2.0 · ViT + LSTM<br>
        Built with Streamlit &amp; PyTorch
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT DETECTOR
# ─────────────────────────────────────────────
detector = get_detector(model_file, spatial_mode, threshold)
detector.set_threshold(threshold)
detector.set_spatial_mode(spatial_mode)


# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
    <p class="hero-title">🛡️ AnomalyShield Dashboard</p>
    <p class="hero-sub">AI-powered CCTV surveillance · Vision Transformer + LSTM Temporal Sequence Analysis</p>
    <div class="hero-badges">
        <span class="badge">ViT-B/16</span>
        <span class="badge">LSTM Temporal</span>
        <span class="badge green">{'GPU Active' if DEVICE == 'cuda' else 'CPU Mode'}</span>
        <span class="badge cyan">EMA Smoothing</span>
        <span class="badge orange">Thresh: {threshold:.2f}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_analyze, tab_incidents, tab_sessions, tab_analytics = st.tabs([
    "🔍  Analysis", "📋  Incident Log", "🎬  Sessions", "📊  Analytics"
])


# ══════════════════════════════════════════════
# TAB 1 · ANALYSIS
# ══════════════════════════════════════════════
with tab_analyze:
    if not video_source_path:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size:4rem; margin-bottom:16px;">📹</div>
            <div style="font-size:1.3rem; font-weight:700; color:#c7d2fe; margin-bottom:8px;">No Video Selected</div>
            <div style="color:#64748b; font-size:0.9rem; max-width:420px; margin:0 auto;">
                Select a video source from the <b style="color:#a5b4fc">sidebar</b> — upload a file,
                pick a demo clip, or browse your training dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("🚀  Analyze Video", type="primary", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:9px 0;">
                <span style="font-size:1.2rem;">🎬</span>
                <div>
                    <div style="font-weight:700; color:#e2e8f0; font-size:0.9rem;">{uploaded_video_name}</div>
                    <div style="color:#64748b; font-size:0.74rem;">
                        Backbone: <b style="color:#a5b4fc">{'ViT-B/16' if spatial_mode == 'vit' else 'MobileNetV2'}</b> ·
                        Threshold: <b style="color:#fcd34d">{threshold:.2f}</b> ·
                        EMA: <b style="color:#6ee7b7">{'ON' if use_ema else 'OFF'}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if run_btn:
            prog = st.progress(0, text="Starting analysis pipeline...")

            def cb(pct, msg):
                prog.progress(int(pct * 100), text=msg)

            with st.spinner(""):
                results = detector.process_video(
                    video_source_path,
                    log_to_db=True,
                    use_ema=use_ema,
                    use_adaptive_threshold=use_adaptive,
                    progress_callback=cb
                )
            prog.empty()

            if "error" in results:
                st.error(f"Detection failed: {results['error']}")
            else:
                st.session_state["results"] = results
                st.session_state["video_name"] = uploaded_video_name
                st.toast("Analysis complete!", icon="✅")

        if "results" in st.session_state:
            res          = st.session_state["results"]
            scores       = res["scores"]
            raw_scores   = res.get("raw_scores", scores)
            max_score    = res["max_score"]
            avg_score    = res["avg_score"]
            is_anom      = res["anomaly_detected"]
            ann_frames   = res.get("annotated_frames", [])
            total_clips  = res["total_clips"]
            total_frames = res["total_frames"]
            is_anom_list = res.get("is_anomalous", [False] * total_clips)
            anom_count   = sum(is_anom_list)
            video_name   = st.session_state.get("video_name", "—")

            # ── Verdict banner
            if is_anom:
                st.markdown(f"""
                <div class="alert-box">
                    <div class="alert-icon">🚨</div>
                    <div>
                        <div class="alert-text-title" style="color:#fca5a5;">ANOMALY DETECTED</div>
                        <div class="alert-text-sub">{anom_count} of {total_clips} clips exceeded threshold {threshold:.2f} · Peak: {max_score:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="normal-box">
                    <div class="alert-icon">✅</div>
                    <div>
                        <div class="alert-text-title" style="color:#6ee7b7;">NORMAL ACTIVITY</div>
                        <div class="alert-text-sub">All {total_clips} clips below threshold {threshold:.2f} · Peak: {max_score:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── KPI Cards
            avg_pct = f"{avg_score * 100:.1f}%"
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card {'red' if is_anom else 'green'}">
                    <div class="kpi-label">Detection Verdict</div>
                    <div class="kpi-value {'kpi-alert-value' if is_anom else 'kpi-safe-value'}">
                        {'🚨 ALERT' if is_anom else '✅ SAFE'}
                    </div>
                    <div class="kpi-sub">{video_name[:28]}</div>
                </div>
                <div class="kpi-card blue">
                    <div class="kpi-label">Peak Anomaly Score</div>
                    <div class="kpi-value">{max_score:.4f}</div>
                    <div class="kpi-sub">Threshold delta {max_score - threshold:+.3f}</div>
                </div>
                <div class="kpi-card cyan">
                    <div class="kpi-label">Average Score</div>
                    <div class="kpi-value">{avg_score:.4f}</div>
                    <div class="kpi-sub">{avg_pct} of max range</div>
                </div>
                <div class="kpi-card {'red' if anom_count else 'green'}">
                    <div class="kpi-label">Alert Clips</div>
                    <div class="kpi-value">{anom_count} / {total_clips}</div>
                    <div class="kpi-sub">{total_frames} total frames</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Clip Timeline Heatmap
            if scores:
                st.markdown("""
                <div class="section-header">
                    <div class="section-dot"></div>
                    <h3>Clip-Level Timeline Heatmap</h3>
                </div>
                """, unsafe_allow_html=True)

                blocks_html = ""
                for i, (s, flag) in enumerate(zip(scores, is_anom_list)):
                    intensity = int(s * 255)
                    if flag:
                        color = f"rgb(239, {max(0, 100 - intensity)}, {max(0, 68 - intensity)})"
                    else:
                        g = int(60 + s * 80)
                        color = f"rgb({max(0, int(s*80))}, {g + 60}, {max(60, 100 - int(s*60))})"
                    title = f"Clip {i+1}: score={s:.3f}"
                    blocks_html += f'<div class="timeline-clip" style="background:{color};" title="{title}"></div>'

                st.markdown(f"""
                <div style="background:rgba(8,10,24,0.6); border:1px solid rgba(99,102,241,0.18);
                            border-radius:12px; padding:14px 18px; margin-bottom:20px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:10px; font-weight:600;">
                        Normal (green) | Anomalous (red) &nbsp;·&nbsp; {total_clips} clips &nbsp;·&nbsp; hover for score
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:2px; align-items:center;">
                        {blocks_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Frame viewer + Chart columns
            col_vid, col_chart = st.columns([1, 1], gap="medium")

            with col_vid:
                st.markdown("""
                <div class="section-header">
                    <div class="section-dot" style="background:#06b6d4; box-shadow:0 0 8px rgba(6,182,212,0.6);"></div>
                    <h3>Frame Telemetry Viewer</h3>
                </div>
                """, unsafe_allow_html=True)

                if ann_frames:
                    frame_idx = st.slider(
                        "Scrub frames",
                        min_value=0, max_value=len(ann_frames) - 1,
                        value=0, step=1, label_visibility="collapsed"
                    )
                    curr_frame = ann_frames[frame_idx]
                    curr_rgb   = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
                    clip_score = scores[frame_idx] if frame_idx < len(scores) else 0.0
                    clip_flag  = is_anom_list[frame_idx] if frame_idx < len(is_anom_list) else False

                    # FIX: use width="stretch" instead of deprecated use_container_width
                    st.image(curr_rgb, width="stretch")

                    bar_color = "#ef4444" if clip_flag else "#10b981"
                    bar_pct   = int(clip_score * 100)

                    st.markdown(f"""
                    <div style="margin-top:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.74rem; color:#94a3b8; margin-bottom:4px;">
                            <span>Anomaly Score</span>
                            <span style="font-weight:700; color:{'#f87171' if clip_flag else '#34d399'};">{clip_score:.4f}</span>
                        </div>
                        <div style="background:rgba(30,32,50,0.8); border-radius:6px; height:8px; overflow:hidden;">
                            <div style="width:{bar_pct}%; height:100%; background:{bar_color};
                                        border-radius:6px; transition:width 0.3s;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px;">
                            <span style="font-size:0.72rem; color:#64748b;">Frame {frame_idx+1} of {len(ann_frames)}</span>
                            <span style="font-size:0.72rem;
                                         background:rgba({'239,68,68' if clip_flag else '16,185,129'},0.15);
                                         color:{'#fca5a5' if clip_flag else '#6ee7b7'};
                                         border:1px solid rgba({'239,68,68' if clip_flag else '16,185,129'},0.3);
                                         border-radius:10px; padding:1px 10px; font-weight:700;">
                                {'ANOMALY' if clip_flag else 'NORMAL'}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No annotated frames available.")

            with col_chart:
                st.markdown("""
                <div class="section-header">
                    <div class="section-dot" style="background:#8b5cf6; box-shadow:0 0 8px rgba(139,92,246,0.6);"></div>
                    <h3>Anomaly Score Timeline</h3>
                </div>
                """, unsafe_allow_html=True)

                clip_x = list(range(len(scores)))
                fig = go.Figure()

                # Raw score (dotted)
                fig.add_trace(go.Scatter(
                    x=clip_x, y=raw_scores,
                    mode="lines", name="Raw Output",
                    line=dict(color="rgba(148,163,184,0.35)", width=1.2, dash="dot"),
                    hovertemplate="Raw: %{y:.4f}<extra></extra>"
                ))

                # Shaded anomaly zones
                for i, flag in enumerate(is_anom_list):
                    if flag:
                        fig.add_vrect(
                            x0=i - 0.5, x1=i + 0.5,
                            fillcolor="rgba(239,68,68,0.07)",
                            line_width=0, layer="below"
                        )

                # EMA / Smoothed score
                line_color = "#ef4444" if is_anom else "#10b981"
                fig.add_trace(go.Scatter(
                    x=clip_x, y=scores,
                    mode="lines+markers",
                    name="EMA Score",
                    line=dict(color=line_color, width=2.5),
                    marker=dict(
                        size=6,
                        color=["#ef4444" if f else "#10b981" for f in is_anom_list],
                        line=dict(width=1, color="rgba(0,0,0,0.4)")
                    ),
                    fill="tozeroy",
                    fillcolor=f"rgba({'239,68,68' if is_anom else '16,185,129'},0.06)",
                    hovertemplate="Clip %{x} · Score: %{y:.4f}<extra></extra>"
                ))

                fig.add_hline(
                    y=threshold,
                    line_dash="dash", line_color="#fbbf24", line_width=1.5,
                    annotation_text=f"Threshold ({threshold:.2f})",
                    annotation_position="top left",
                    annotation_font=dict(color="#fbbf24", size=11)
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,10,24,0.0)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    xaxis=dict(title="Temporal Clip Index (T=16 frames)",
                               gridcolor="rgba(99,102,241,0.08)", zeroline=False,
                               tickfont=dict(size=10)),
                    yaxis=dict(title="Anomaly Probability", range=[-0.05, 1.05],
                               gridcolor="rgba(99,102,241,0.08)", zeroline=False,
                               tickfont=dict(size=10)),
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11)),
                    hovermode="x unified",
                )
                _plotly_chart(fig, height=340)

                # ── Mini score histogram  (FIX: use per-bar marker_color list, not colorscale on array)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                hist_colors = ["#ef4444" if s >= threshold else "#6366f1" for s in scores]
                fig2 = go.Figure()
                # Build histogram manually so we control colors per bin
                score_arr = np.array(scores)
                bins = np.linspace(0, 1, 22)
                counts, edges = np.histogram(score_arr, bins=bins)
                bar_colors_hist = ["#ef4444" if (edges[i] + edges[i+1]) / 2 >= threshold else "#6366f1"
                                   for i in range(len(counts))]
                fig2.add_trace(go.Bar(
                    x=[(edges[i] + edges[i+1]) / 2 for i in range(len(counts))],
                    y=counts,
                    width=[(edges[i+1] - edges[i]) * 0.9 for i in range(len(counts))],
                    marker=dict(color=bar_colors_hist, line=dict(width=0)),
                    opacity=0.88,
                    name="Distribution",
                    hovertemplate="Score ~%{x:.2f}<br>Count: %{y}<extra></extra>"
                ))
                fig2.add_vline(
                    x=threshold, line_dash="dash", line_color="#fbbf24",
                    annotation_text=f"Thresh {threshold:.2f}",
                    annotation_font=dict(color="#fbbf24", size=10)
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,10,24,0.0)",
                    font=dict(color="#94a3b8"),
                    margin=dict(l=10, r=10, t=24, b=10),
                    xaxis=dict(title="Score", gridcolor="rgba(99,102,241,0.08)", zeroline=False, range=[0, 1]),
                    yaxis=dict(title="# Clips", gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                    showlegend=False, bargap=0.05
                )
                _plotly_chart(fig2, height=190)


# ══════════════════════════════════════════════
# TAB 2 · INCIDENT LOG
# ══════════════════════════════════════════════
with tab_incidents:
    # Always refresh analytics fresh
    inc_analytics = db.get_incident_analytics()
    total_events  = inc_analytics["total_events"]
    max_rec       = inc_analytics["max_recorded_score"]
    avg_rec       = inc_analytics["average_recorded_score"]
    top_sources   = inc_analytics.get("top_incident_sources", [])

    # ── KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Incidents Logged</div>
            <div class="kpi-value">{total_events}</div>
            <div class="kpi-sub">SQLite audit trail</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Peak Recorded Score</div>
            <div class="kpi-value">{max_rec:.4f}</div>
            <div class="kpi-sub">All-time maximum</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Historical Average</div>
            <div class="kpi-value">{avg_rec:.4f}</div>
            <div class="kpi-sub">Mean anomaly score</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Video Sources Flagged</div>
            <div class="kpi-value">{len(top_sources)}</div>
            <div class="kpi-sub">Distinct sources</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Database Retrieval & Filter Controls
    f_row1_col1, f_row1_col2, f_row1_col3, f_row1_col4 = st.columns([1.5, 1.5, 1.2, 0.8])
    with f_row1_col1:
        search_query = st.text_input("🔍 Search Logs", placeholder="Search by video name or note...", key="inc_search")
    with f_row1_col2:
        score_range = st.slider("Score Range", 0.0, 1.0, (0.0, 1.0), step=0.05, key="inc_score_range")
    with f_row1_col3:
        order_choice = st.selectbox(
            "Sort Order",
            ["Newest First", "Highest Score First", "Lowest Score First", "Oldest First"],
            index=0,
            key="inc_order"
        )
    with f_row1_col4:
        limit = st.selectbox("Page Size", [25, 50, 100, 250, 500], index=1, key="inc_limit")

    order_map = {
        "Newest First": "id DESC",
        "Highest Score First": "anomaly_score DESC",
        "Lowest Score First": "anomaly_score ASC",
        "Oldest First": "id ASC"
    }

    # ── Total count matching criteria in DB
    total_matching = db.count_events(
        min_score=score_range[0] if score_range[0] > 0.0 else None,
        search_text=search_query if search_query.strip() else None
    )

    col_meta1, col_meta2 = st.columns([3, 1])
    with col_meta1:
        st.caption(f"Showing up to **{limit}** records out of **{total_matching}** matching database events.")
    with col_meta2:
        if st.button("🔄 Refresh Data", key="refresh_inc_btn", use_container_width=True):
            st.rerun()

    # ── Fetch events from Database
    recent_events = db.fetch_recent_events(
        limit=limit,
        min_score=score_range[0] if score_range[0] > 0.0 else None,
        max_score=score_range[1] if score_range[1] < 1.0 else None,
        search_text=search_query.strip() if search_query.strip() else None,
        order_by=order_map.get(order_choice, "id DESC")
    )

    if recent_events:
        df = pd.DataFrame(recent_events)

        display_df = df[["id", "timestamp", "video_source", "frame_number",
                          "anomaly_score", "classification", "threshold_used", "notes"]].copy()
        display_df.columns = ["ID", "Timestamp", "Video Source", "Frame #",
                               "Anomaly Score", "Classification", "Threshold", "Notes"]

        # ── FIX: Use Styler.map (pandas >= 2.1) instead of deprecated applymap
        def _score_color(val):
            if val >= 0.75:
                return "background-color: rgba(239,68,68,0.20); color: #fca5a5; font-weight:700"
            elif val >= 0.50:
                return "background-color: rgba(245,158,11,0.15); color: #fcd34d; font-weight:600"
            else:
                return "color: #94a3b8"

        styled = display_df.style.map(_score_color, subset=["Anomaly Score"])

        # FIX: use width="stretch" instead of deprecated use_container_width
        st.dataframe(styled, width="stretch", hide_index=True, height=420)

        # ── Inline trend mini-chart for incident scores
        if len(display_df) > 1:
            st.markdown("""
            <div class="section-header" style="margin-top:16px;">
                <div class="section-dot" style="background:#f59e0b;"></div>
                <h3>Recent Incident Score Trend</h3>
            </div>
            """, unsafe_allow_html=True)

            fig_inc = go.Figure()
            fig_inc.add_trace(go.Scatter(
                x=list(range(len(display_df))),
                y=display_df["Anomaly Score"].tolist(),
                mode="lines+markers",
                line=dict(color="#f59e0b", width=2),
                marker=dict(
                    size=7,
                    color=["#ef4444" if v >= threshold else "#10b981"
                           for v in display_df["Anomaly Score"].tolist()],
                    line=dict(width=1, color="rgba(0,0,0,0.3)")
                ),
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.06)",
                hovertemplate="Incident %{x}: score %{y:.4f}<extra></extra>"
            ))
            fig_inc.add_hline(
                y=threshold, line_dash="dash", line_color="#fbbf24", line_width=1.2,
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_font=dict(color="#fbbf24", size=10)
            )
            fig_inc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,10,24,0.0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Recent Incidents (newest first)",
                           gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                yaxis=dict(title="Score", range=[0, 1],
                           gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                showlegend=False,
            )
            _plotly_chart(fig_inc, height=200)

        # ── Export
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export CSV",
            data=csv_data,
            file_name=f"anomaly_incidents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

    else:
        st.markdown("""
        <div style="text-align:center; padding:48px 20px;">
            <div style="font-size:3rem;">🗂️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#c7d2fe; margin-top:12px;">No Incidents Logged</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:6px;">
                Run a video analysis in the <b style="color:#a5b4fc">Analysis tab</b> to start logging incidents.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Danger zone
    st.divider()
    with st.expander("⚠️ Danger Zone — Clear Incident Database"):
        st.warning("This permanently deletes all logged anomaly incidents.", icon="⚠️")
        if st.button("🗑️ Clear All Incident Logs", type="primary", key="clear_logs_btn"):
            db.clear_logs()
            st.success("All incident records cleared.")
            st.rerun()


# ══════════════════════════════════════════════
# TAB 3 · SESSIONS (Video Analysis History)
# ══════════════════════════════════════════════
with tab_sessions:
    sess_analytics = db.get_session_analytics()
    db_info        = db.get_db_info()

    # ── Session KPI cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Sessions Run</div>
            <div class="kpi-value">{sess_analytics['total_sessions']}</div>
            <div class="kpi-sub">Full video analyses</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Alert Sessions</div>
            <div class="kpi-value">{sess_analytics['alert_sessions']}</div>
            <div class="kpi-sub">Anomaly detected</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Normal Sessions</div>
            <div class="kpi-value">{sess_analytics['normal_sessions']}</div>
            <div class="kpi-sub">No anomaly found</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">DB Size</div>
            <div class="kpi-value">{db_info['size_kb']} KB</div>
            <div class="kpi-sub">{db_info['row_counts'].get('score_timeseries', 0)} score points stored</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DB Info box
    rc = db_info["row_counts"]
    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.20);
                border-radius:12px; padding:14px 20px; margin-bottom:20px; font-size:0.82rem;">
        <div style="color:#94a3b8; font-weight:700; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.7px; margin-bottom:10px;">📂 Database File Info</div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px;">
            <div><div style="color:#64748b;">Path</div>
                 <div style="color:#c7d2fe; font-weight:600; word-break:break-all;">{db_info['path']}</div></div>
            <div><div style="color:#64748b;">video_sessions</div>
                 <div style="color:#a5b4fc; font-weight:700;">{rc.get('video_sessions',0)} rows</div></div>
            <div><div style="color:#64748b;">anomaly_events</div>
                 <div style="color:#f87171; font-weight:700;">{rc.get('anomaly_events',0)} rows</div></div>
            <div><div style="color:#64748b;">score_timeseries</div>
                 <div style="color:#6ee7b7; font-weight:700;">{rc.get('score_timeseries',0)} rows</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters
    s_col1, s_col2, s_col3 = st.columns([1, 1, 2])
    with s_col1:
        sess_limit = st.selectbox("Show last", [10, 25, 50, 100], index=1, key="sess_limit")
    with s_col2:
        anomaly_only = st.toggle("Anomaly sessions only", value=False, key="sess_anom_only")
    with s_col3:
        source_search = st.text_input("Search by video name", placeholder="e.g. surveillance_clip", key="sess_source")

    sessions = db.fetch_sessions(
        limit=sess_limit,
        anomaly_only=anomaly_only,
        video_source=source_search if source_search else None
    )

    if sessions:
        df_sess = pd.DataFrame(sessions)

        # ── Human-friendly display
        display_sess = df_sess[[
            "id", "timestamp", "video_source", "spatial_mode",
            "total_frames", "total_clips", "anomaly_clips",
            "max_score", "avg_score", "anomaly_detected", "threshold_used"
        ]].copy()
        display_sess.columns = [
            "ID", "Timestamp", "Video Source", "Backbone",
            "Frames", "Clips", "Alert Clips",
            "Max Score", "Avg Score", "Anomaly?", "Threshold"
        ]
        display_sess["Anomaly?"] = display_sess["Anomaly?"].map({1: "🚨 YES", 0: "✅ NO"})
        display_sess["Max Score"] = display_sess["Max Score"].round(4)
        display_sess["Avg Score"] = display_sess["Avg Score"].round(4)

        def _sess_row_color(val):
            if val == "🚨 YES":
                return "background-color:rgba(239,68,68,0.15); color:#fca5a5; font-weight:700"
            return "color:#6ee7b7; font-weight:600"

        styled_sess = display_sess.style.map(_sess_row_color, subset=["Anomaly?"])
        st.dataframe(styled_sess, width="stretch", hide_index=True, height=380)

        # ── Drill-down: select session to view score timeseries
        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <div class="section-dot" style="background:#06b6d4;"></div>
            <h3>Score Timeseries Drill-Down</h3>
        </div>
        """, unsafe_allow_html=True)

        sess_ids = [str(s["id"]) for s in sessions]
        sess_labels = [
            f"Session {s['id']} · {s['video_source'][:30]} · {s['timestamp']}"
            for s in sessions
        ]
        selected_sess_label = st.selectbox(
            "Select a session to inspect its full clip-by-clip score trace:",
            sess_labels, key="sess_drill_select"
        )
        selected_sess_id = sessions[sess_labels.index(selected_sess_label)]["id"]
        sel_session = db.get_session_by_id(selected_sess_id)
        timeseries  = db.fetch_timeseries(selected_sess_id)

        if sel_session:
            anom_d = sel_session.get("anomaly_detected", 0)
            st.markdown(f"""
            <div style="background:rgba({'239,68,68' if anom_d else '16,185,129'},0.08);
                        border-left:4px solid {'#ef4444' if anom_d else '#10b981'};
                        border:1px solid rgba({'239,68,68' if anom_d else '16,185,129'},0.25);
                        border-radius:10px; padding:12px 18px; margin-bottom:14px; font-size:0.83rem;">
                <b style="color:{'#fca5a5' if anom_d else '#6ee7b7'};">{'🚨 ANOMALY' if anom_d else '✅ NORMAL'}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Video:</span> <b style="color:#c7d2fe;">{sel_session['video_source']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Backbone:</span> <b style="color:#a5b4fc;">{sel_session['spatial_mode']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Clips:</span> <b style="color:#e2e8f0;">{sel_session['total_clips']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Alert clips:</span> <b style="color:#f87171;">{sel_session['anomaly_clips']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Peak:</span> <b style="color:#fcd34d;">{sel_session['max_score']:.4f}</b>
            </div>
            """, unsafe_allow_html=True)

        if timeseries:
            df_ts = pd.DataFrame(timeseries)
            ts_thresh = sel_session["threshold_used"] if sel_session else threshold

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=df_ts["clip_index"].tolist(), y=df_ts["raw_score"].tolist(),
                mode="lines", name="Raw Score",
                line=dict(color="rgba(148,163,184,0.35)", width=1.2, dash="dot"),
                hovertemplate="Clip %{x} · Raw: %{y:.4f}<extra></extra>"
            ))
            ema_colors = ["#ef4444" if b else "#10b981" for b in df_ts["is_anomalous"].tolist()]
            fig_ts.add_trace(go.Scatter(
                x=df_ts["clip_index"].tolist(), y=df_ts["ema_score"].tolist(),
                mode="lines+markers", name="EMA Score",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=5, color=ema_colors, line=dict(width=1, color="rgba(0,0,0,0.3)")),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                hovertemplate="Clip %{x} · EMA: %{y:.4f}<extra></extra>"
            ))
            fig_ts.add_hline(
                y=ts_thresh, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                annotation_text=f"Threshold ({ts_thresh:.2f})",
                annotation_font=dict(color="#fbbf24", size=10)
            )
            for i, row in df_ts[df_ts["is_anomalous"] == 1].iterrows():
                fig_ts.add_vrect(
                    x0=row["clip_index"] - 0.5, x1=row["clip_index"] + 0.5,
                    fillcolor="rgba(239,68,68,0.07)", line_width=0, layer="below"
                )
            fig_ts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,10,24,0.0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Clip Index", gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                yaxis=dict(title="Score", range=[-0.05, 1.05], gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified"
            )
            _plotly_chart(fig_ts, height=300)
        else:
            st.info("No score timeseries data stored for this session yet. Run an analysis to populate.")

        # ── Export sessions CSV
        csv_sess = pd.DataFrame(sessions).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Sessions CSV", data=csv_sess,
            file_name=f"sessions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="sess_csv_btn"
        )

    else:
        st.markdown("""
        <div style="text-align:center; padding:48px 20px;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-size:1.1rem; font-weight:700; color:#c7d2fe; margin-top:12px;">No Sessions Yet</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:6px;">
                Analyze a video in the <b style="color:#a5b4fc">Analysis tab</b> to log the first session.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 · ANALYTICS
# ══════════════════════════════════════════════
with tab_analytics:
    an_analytics = db.get_incident_analytics()
    recent_all   = db.fetch_recent_events(limit=500)

    if not recent_all:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
            <div style="font-size:3.5rem;">📊</div>
            <div style="font-size:1.1rem; font-weight:700; color:#c7d2fe; margin-top:14px;">No Data Yet</div>
            <div style="color:#64748b; font-size:0.87rem; margin-top:6px;">
                Analyze videos first to populate the analytics dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_all = pd.DataFrame(recent_all)
        # FIX: robust timestamp parsing with error coercion
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["hour"] = df_all["timestamp"].dt.hour
        df_all["date"] = df_all["timestamp"].dt.date

        col_l, col_r = st.columns(2, gap="medium")

        with col_l:
            # ── Score trend over time
            st.markdown("""
            <div class="section-header">
                <div class="section-dot"></div>
                <h3>Score Trend Over Time</h3>
            </div>
            """, unsafe_allow_html=True)

            scores_list = df_all["anomaly_score"].tolist()
            # FIX: use per-point colors as a list (not a raw column in colorscale)
            point_colors = []
            for s in scores_list:
                if s >= 0.75:
                    point_colors.append("#ef4444")
                elif s >= 0.50:
                    point_colors.append("#fbbf24")
                else:
                    point_colors.append("#10b981")

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_all["timestamp"].tolist(),
                y=scores_list,
                mode="markers+lines",
                marker=dict(color=point_colors, size=7,
                            line=dict(width=1, color="rgba(0,0,0,0.3)")),
                line=dict(color="rgba(99,102,241,0.3)", width=1),
                hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>"
            ))
            fig_trend.add_hline(
                y=threshold, line_dash="dash", line_color="#fbbf24", line_width=1,
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_font=dict(color="#fbbf24", size=10)
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,10,24,0.0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False),
                yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False, range=[0, 1]),
                showlegend=False
            )
            _plotly_chart(fig_trend, height=280)

            # ── Hourly activity bar chart
            st.markdown("""
            <div class="section-header" style="margin-top:18px;">
                <div class="section-dot" style="background:#06b6d4;"></div>
                <h3>Incident Activity by Hour of Day</h3>
            </div>
            """, unsafe_allow_html=True)

            hourly = df_all.groupby("hour").size().reset_index(name="count")
            all_hours = pd.DataFrame({"hour": range(24)})
            hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
            hourly["count"] = hourly["count"].astype(int)

            # FIX: safe bar colors — avoid colorscale crash on uniform/single-value arrays
            h_colors = _safe_bar_colors(hourly["count"].tolist())

            fig_hour = go.Figure(go.Bar(
                x=hourly["hour"].tolist(),
                y=hourly["count"].tolist(),
                marker=dict(color=h_colors, line=dict(width=0)),
                hovertemplate="Hour %{x}:00  ·  %{y} incidents<extra></extra>"
            ))
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,10,24,0.0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Hour of Day", gridcolor="rgba(99,102,241,0.08)",
                           zeroline=False, dtick=2),
                yaxis=dict(title="# Incidents", gridcolor="rgba(99,102,241,0.08)",
                           zeroline=False),
                showlegend=False, bargap=0.15
            )
            _plotly_chart(fig_hour, height=220)

        with col_r:
            # ── Score distribution histogram
            st.markdown("""
            <div class="section-header">
                <div class="section-dot" style="background:#8b5cf6;"></div>
                <h3>Score Distribution</h3>
            </div>
            """, unsafe_allow_html=True)

            # FIX: manually build histogram bins so we can use per-bar colors
            score_vals = np.array(df_all["anomaly_score"].tolist(), dtype=float)
            h_counts, h_edges = np.histogram(score_vals, bins=25, range=(0.0, 1.0))
            h_mids   = [(h_edges[i] + h_edges[i+1]) / 2 for i in range(len(h_counts))]
            h_widths = [(h_edges[i+1] - h_edges[i]) * 0.92 for i in range(len(h_counts))]
            h_bcolors = []
            for mid in h_mids:
                if mid >= 0.75:
                    h_bcolors.append("#ef4444")
                elif mid >= 0.50:
                    h_bcolors.append("#fbbf24")
                else:
                    h_bcolors.append("#6366f1")

            fig_hist = go.Figure(go.Bar(
                x=h_mids, y=h_counts.tolist(),
                width=h_widths,
                marker=dict(color=h_bcolors, line=dict(width=0.5, color="rgba(0,0,0,0.2)")),
                opacity=0.90,
                hovertemplate="Score ~%{x:.2f}<br>Count: %{y}<extra></extra>"
            ))
            fig_hist.add_vline(
                x=threshold, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                annotation_text=f"Threshold {threshold:.2f}",
                annotation_font=dict(color="#fbbf24", size=10)
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,10,24,0.0)",
                font=dict(color="#94a3b8"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Anomaly Score", gridcolor="rgba(99,102,241,0.08)",
                           zeroline=False, range=[0, 1]),
                yaxis=dict(title="Frequency", gridcolor="rgba(99,102,241,0.08)",
                           zeroline=False),
                showlegend=False, bargap=0.04
            )
            _plotly_chart(fig_hist, height=280)

            # ── Top incident sources
            st.markdown("""
            <div class="section-header" style="margin-top:18px;">
                <div class="section-dot" style="background:#f59e0b;"></div>
                <h3>Top Flagged Video Sources</h3>
            </div>
            """, unsafe_allow_html=True)

            top_sources_data = an_analytics.get("top_incident_sources", [])
            if top_sources_data:
                df_top = pd.DataFrame(top_sources_data)
                df_top["short_name"] = df_top["video_source"].apply(
                    lambda x: ("..." + x[-26:]) if len(x) > 28 else x
                )
                count_list = df_top["count"].tolist()

                # FIX: safe per-bar colors to avoid colorscale crash on 1-row DataFrames
                bar_clrs = _safe_bar_colors(count_list)

                fig_bar = go.Figure(go.Bar(
                    x=count_list,
                    y=df_top["short_name"].tolist(),
                    orientation="h",
                    marker=dict(color=bar_clrs, line=dict(width=0)),
                    hovertemplate="%{y}<br>Incidents: %{x}<extra></extra>"
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,10,24,0.0)",
                    font=dict(color="#94a3b8"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="# Incidents", gridcolor="rgba(99,102,241,0.08)",
                               zeroline=False),
                    yaxis=dict(gridcolor="rgba(99,102,241,0.08)", zeroline=False,
                               autorange="reversed"),
                    showlegend=False, bargap=0.25
                )
                _plotly_chart(fig_bar, height=220)
            else:
                st.info("No flagged sources in database yet.")

        # ── Summary stats row
        st.divider()
        st.markdown("""
        <div class="section-header">
            <div class="section-dot" style="background:#10b981;"></div>
            <h3>Summary Statistics</h3>
        </div>
        """, unsafe_allow_html=True)

        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            with st.container(border=True):
                st.metric("Total Records", len(df_all))
        with stat_c2:
            with st.container(border=True):
                above = int((df_all["anomaly_score"] >= threshold).sum())
                st.metric("Above Threshold", above, delta=f"{above/max(len(df_all),1)*100:.1f}%")
        with stat_c3:
            with st.container(border=True):
                st.metric("Std Deviation", f"{df_all['anomaly_score'].std():.4f}")
        with stat_c4:
            with st.container(border=True):
                st.metric("Median Score", f"{df_all['anomaly_score'].median():.4f}")
