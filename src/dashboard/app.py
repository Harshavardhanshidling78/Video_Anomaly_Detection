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
import base64
import json
import streamlit.components.v1 as components

from src.config import (
    CHECKPOINT_DIR, DATA_DIR,
    ANOMALY_THRESHOLD, DEVICE
)
from src.pipeline.detector import VideoAnomalyDetector
from src.utils.database import AnomalyDatabase

# ─────────────────────────────────────────────
# PAGE CONFIG — Full-Width Zero-Sidebar Layout
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AnomalyShield Enterprise AI | Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# DESIGN SYSTEM: CSS & ENTERPRISE TOKENS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Hard Removal of Sidebar & Default Streamlit Chrome ── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarExpandButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

#MainMenu, header[data-testid="stHeader"], footer {
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Global Canvas & Base Hierarchy ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #060913 !important;
    background-image: 
        radial-gradient(circle at 10% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 90% 5%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 50% 95%, rgba(139, 92, 246, 0.05) 0%, transparent 50%) !important;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    min-height: 100vh;
}

.block-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding-top: 1.25rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Top Navigation Bar: SaaS Header ── */
.top-nav-bar {
    background: rgba(13, 19, 36, 0.78);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.nav-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nav-logo-badge {
    width: 46px;
    height: 46px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.15) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);
}

.nav-brand-info {
    display: flex;
    flex-direction: column;
}

.nav-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.2;
    text-wrap: balance;
}

.nav-badge-enterprise {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #c7d2fe;
}

.nav-subtitle {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 3px;
    letter-spacing: -0.01em;
    text-wrap: pretty;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.nav-status-chip {
    background: rgba(18, 26, 48, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 10px;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    transition: all 0.2s ease;
}
.nav-status-chip:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(18, 26, 48, 0.9);
}

.chip-title {
    color: #64748b;
    font-weight: 500;
}

.chip-val {
    color: #f1f5f9;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-indicator.online {
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: livePulse 2s infinite;
}

@keyframes livePulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* ── Modern Segmented Tabs Navigation ── */
[data-baseweb="tab-list"] {
    background: rgba(13, 19, 36, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    gap: 6px !important;
    margin-bottom: 24px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: -0.01em !important;
    padding: 9px 18px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.22) 0%, rgba(139, 92, 246, 0.16) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(99, 102, 241, 0.38) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* ── Enterprise Console Panels & Cards ── */
.console-panel {
    background: rgba(13, 19, 36, 0.72);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.panel-header-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
}

.panel-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #c7d2fe;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.panel-sub {
    font-size: 0.76rem;
    color: #64748b;
    letter-spacing: -0.01em;
}

/* ── High-Contrast KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
@media (max-width: 960px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 540px) {
    .kpi-grid { grid-template-columns: 1fr; }
}

.kpi-card {
    background: rgba(13, 19, 36, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.kpi-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), 0 0 16px rgba(99, 102, 241, 0.12);
}

.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.blue::after  { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.green::after { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.red::after   { background: linear-gradient(90deg, #f43f5e, #fb7185); }
.kpi-card.cyan::after  { background: linear-gradient(90deg, #06b6d4, #38bdf8); }

.kpi-label {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 6px 0 3px;
    line-height: 1.1;
    letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
}

.kpi-sub {
    font-size: 0.74rem;
    color: #64748b;
    font-variant-numeric: tabular-nums;
}
.kpi-alert-value { color: #f43f5e !important; text-shadow: 0 0 16px rgba(244, 63, 94, 0.3); }
.kpi-safe-value  { color: #34d399 !important; text-shadow: 0 0 16px rgba(52, 211, 153, 0.3); }

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.section-header h3 {
    font-size: 1.02rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.015em;
    margin: 0;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.7);
}

/* ── Threat Notification Banners ── */
.alert-box {
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.35);
    border-left: 5px solid #f43f5e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(244, 63, 94, 0.15);
}

.normal-box {
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.30);
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12);
}
.alert-icon { font-size: 1.75rem; flex-shrink: 0; }
.alert-text-title { font-weight: 800; font-size: 1.02rem; letter-spacing: -0.01em; }
.alert-text-sub   { font-size: 0.82rem; color: #cbd5e1; margin-top: 3px; font-variant-numeric: tabular-nums; }

/* ── Timeline Segment Heatmap ── */
.timeline-clip {
    display: inline-block;
    width: 13px; height: 32px;
    border-radius: 4px;
    margin: 1px;
    cursor: pointer;
    transition: transform 0.15s cubic-bezier(0.16, 1, 0.3, 1), filter 0.15s ease;
}
.timeline-clip:hover { transform: scaleY(1.35) scaleX(1.1); filter: brightness(1.3); z-index: 5; }

/* ── Tactile Enterprise Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: 1px solid rgba(165, 180, 252, 0.3) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 10px 24px !important;
    color: #ffffff !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2), 0 4px 18px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3), 0 6px 24px rgba(79, 70, 229, 0.6) !important;
    filter: brightness(1.08) !important;
}
[data-testid="stButton"] > button[kind="primary"]:active {
    transform: scale(0.98) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(18, 26, 48, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 7px 16px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(99, 102, 241, 0.45) !important;
    color: #ffffff !important;
    background: rgba(25, 36, 66, 0.85) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button[kind="secondary"]:active {
    transform: scale(0.98) !important;
}

/* ── Frame Image Polish ── */
img {
    border-radius: 12px !important;
    outline: 1px solid rgba(255, 255, 255, 0.08) !important;
    outline-offset: -1px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
}

/* ── Form Inputs, Selects & Sliders ── */
div[data-baseweb="select"] > div {
    background: rgba(13, 19, 36, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.84rem !important;
}

/* ── Native Segmented Control (st.segmented_control) ── */
[data-testid="stSegmentedControl"] {
    display: inline-flex !important;
    background: rgba(13, 19, 36, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 6px !important;
    width: 100% !important;
}

[data-testid="stSegmentedControl"] button {
    flex: 1 !important;
    border-radius: 8px !important;
    padding: 9px 18px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stSegmentedControl"] button:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.30) 0%, rgba(139, 92, 246, 0.22) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* ── Cyber Dropzone Styling ── */
[data-testid="stFileUploader"] {
    background: rgba(13, 19, 36, 0.6) !important;
    border: 1.5px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99, 102, 241, 0.7) !important;
    background: rgba(18, 26, 48, 0.75) !important;
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.18) !important;
}

/* ── Enterprise Footer ── */
.website-footer {
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    margin-top: 48px;
    padding-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 0.78rem;
    color: #64748b;
}
.footer-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 20px;
    padding: 3px 10px;
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _plotly_chart(fig, height=None, **kwargs):
    """Wrapper: always use width='stretch' + suppress displayModeBar with custom layout."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kwargs)


def _safe_bar_colors(values, low_color="#6366f1", high_color="#8b5cf6"):
    """Return per-bar colors via interpolation — avoids colorscale crash on 1-value arrays."""
    arr = list(values)
    mn, mx = min(arr), max(arr)
    if mx == mn:
        return [high_color] * len(arr)
    palette = [
        "rgba(99,102,241,0.35)", "rgba(99,102,241,0.50)", "rgba(99,102,241,0.65)",
        "rgba(120,100,241,0.75)", "rgba(139,92,246,0.85)", "#8b5cf6"
    ]
    colors = []
    for v in arr:
        idx = int((v - mn) / (mx - mn) * (len(palette) - 1))
        colors.append(palette[idx])
    return colors


def _render_radial_gauge(score: float, threshold: float):
    pct = max(0.0, min(1.0, float(score)))
    deg = int(pct * 360)
    if score >= 0.65:
        color = "#f43f5e"
        badge_text = "CRITICAL THREAT"
        badge_bg = "rgba(244, 63, 94, 0.16)"
        badge_border = "rgba(244, 63, 94, 0.45)"
        badge_color = "#fca5a5"
    elif score >= threshold:
        color = "#fbbf24"
        badge_text = "ELEVATED RISK"
        badge_bg = "rgba(251, 191, 36, 0.16)"
        badge_border = "rgba(251, 191, 36, 0.45)"
        badge_color = "#fcd34d"
    else:
        color = "#10b981"
        badge_text = "SAFE BASELINE"
        badge_bg = "rgba(16, 185, 129, 0.16)"
        badge_border = "rgba(16, 185, 129, 0.35)"
        badge_color = "#6ee7b7"

    gauge_html = f"""
    <div style="display:flex; align-items:center; gap:16px; background:rgba(18,26,48,0.75); border:1px solid rgba(148,163,184,0.14); border-radius:12px; padding:12px 18px; margin-bottom:12px;">
        <div style="position:relative; width:64px; height:64px; border-radius:50%; background:conic-gradient({color} {deg}deg, rgba(255,255,255,0.08) {deg}deg); display:flex; align-items:center; justify-content:center; box-shadow:0 0 16px {color}33; flex-shrink:0;">
            <div style="width:50px; height:50px; border-radius:50%; background:#0b1122; display:flex; align-items:center; justify-content:center; font-family:'JetBrains Mono', monospace; font-size:0.84rem; font-weight:800; color:#ffffff;">
                {int(pct * 100)}%
            </div>
        </div>
        <div style="flex:1;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; font-weight:600;">Frame Threat Level</span>
                <span style="font-size:0.68rem; font-weight:700; background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}; border-radius:6px; padding:2px 8px; letter-spacing:0.04em;">
                    {badge_text}
                </span>
            </div>
            <div style="font-size:1.15rem; font-weight:800; color:{color}; font-family:'JetBrains Mono', monospace;">
                {score:.4f} <span style="font-size:0.72rem; color:#64748b; font-weight:500;">/ 1.0000</span>
            </div>
        </div>
    </div>
    """
    return gauge_html


# ─────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def get_detector(model_path: str, spatial_mode: str = "vit"):
    import gc
    gc.collect()
    return VideoAnomalyDetector(
        model_path=model_path,
        spatial_mode=spatial_mode,
        threshold=ANOMALY_THRESHOLD
    )

@st.cache_resource
def get_database():
    return AnomalyDatabase()

db = get_database()

# ─────────────────────────────────────────────
# CACHED DB WRAPPERS (Eliminates tab-switching & rerun query lag)
# ─────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_incident_analytics():
    return db.get_incident_analytics()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_count_events(min_score=None, search_text=None):
    return db.count_events(min_score=min_score, search_text=search_text)

@st.cache_data(ttl=30, show_spinner=False)
def _cached_fetch_recent_events(limit=50, min_score=None, max_score=None, search_text=None, order_by="id DESC"):
    return db.fetch_recent_events(
        limit=limit,
        min_score=min_score,
        max_score=max_score,
        search_text=search_text,
        order_by=order_by
    )

@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_session_analytics():
    return db.get_session_analytics()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_db_info():
    return db.get_db_info()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_fetch_sessions(limit=25, anomaly_only=False, video_source=None):
    return db.fetch_sessions(
        limit=limit,
        anomaly_only=anomaly_only,
        video_source=video_source
    )

@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_session_by_id(session_id: int):
    return db.get_session_by_id(session_id)

@st.cache_data(ttl=30, show_spinner=False)
def _cached_fetch_timeseries(session_id: int):
    return db.fetch_timeseries(session_id)


# ─────────────────────────────────────────────
# ZERO-LATENCY FRAME ENCODING & CLIENT-SIDE STUDIO
# ─────────────────────────────────────────────
def _encode_frames_base64(frames, max_w=640, quality=75):
    """Pre-encodes frames into lightweight base64 JPEG strings for instantaneous browser switching."""
    encoded = []
    for f in frames:
        h, w = f.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            f = cv2.resize(f, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        encoded.append("data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii'))
    return encoded


def _render_instant_telemetry_studio(
    encoded_frames: list,
    scores: list,
    raw_scores: list,
    is_anom_list: list,
    anomaly_rois: dict,
    threshold: float,
    total_clips: int,
    weapon_clip_indices: set
):
    """Renders a 0ms client-side interactive telemetry studio with scrubbing, playback, and HUD."""
    payload = {
        "total_clips": total_clips,
        "threshold": float(threshold),
        "scores": [round(float(s), 4) for s in scores],
        "raw_scores": [round(float(s), 4) for s in raw_scores],
        "is_anom": [bool(f) for f in is_anom_list],
        "rois": {str(k): v for k, v in anomaly_rois.items()},
        "weapon_clips": list(weapon_clip_indices),
        "frames": encoded_frames
    }
    json_str = json.dumps(payload)

    player_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
    color: #e2e8f0;
    overflow: hidden;
    user-select: none;
}}
.studio-box {{
    background: rgba(13, 19, 36, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
/* Top Control & HUD Bar */
.hud-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    background: rgba(18, 26, 48, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 8px;
    padding: 6px 10px;
}}
.btn-group {{
    display: inline-flex;
    gap: 4px;
    align-items: center;
}}
.hud-btn {{
    background: rgba(30, 41, 69, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #cbd5e1;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    line-height: 1;
}}
.hud-btn:hover {{
    background: rgba(45, 62, 105, 0.95);
    color: #ffffff;
    border-color: rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
}}
.hud-btn:active {{
    transform: scale(0.96);
}}
.hud-btn.primary {{
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    border: 1px solid rgba(165, 180, 252, 0.35);
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
}}
.hud-btn.primary.playing {{
    background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%);
    box-shadow: 0 2px 8px rgba(244, 63, 94, 0.45);
}}
.hud-btn.threat {{
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid rgba(244, 63, 94, 0.35);
    color: #fca5a5;
}}
.hud-btn.threat:hover {{
    background: rgba(244, 63, 94, 0.3);
    color: #ffffff;
}}
.hud-select {{
    background: rgba(18, 26, 48, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #a5b4fc;
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 0.71rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    cursor: pointer;
}}
/* Status Readout in HUD */
.hud-status {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
}}
.mini-gauge {{
    position: relative;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: all 0.2s ease;
}}
.mini-gauge-inner {{
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #0b1122;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 800;
    color: #ffffff;
}}
.hud-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 800;
    color: #f8fafc;
}}
.hud-badge {{
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 5px;
    padding: 3px 8px;
    letter-spacing: 0.03em;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}}
/* Video Viewport */
.video-frame-container {{
    position: relative;
    width: 100%;
    height: 350px;
    background: #000000;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
}}
.video-img {{
    max-width: 100%;
    max-height: 350px;
    width: auto;
    height: auto;
    display: block;
    object-fit: contain;
}}
/* Spatial ROI Overlay */
.roi-float-overlay {{
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(11, 15, 25, 0.85);
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 4px 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    z-index: 10;
}}
.roi-float-badge {{
    background: rgba(245, 158, 11, 0.25);
    color: #fbbf24;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 1px 5px;
    border-radius: 4px;
}}
.roi-float-text {{
    font-size: 0.72rem;
    color: #fef3c7;
    font-family: 'JetBrains Mono', monospace;
}}
/* Slim Timeline & Scrubber */
.scrub-container {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: rgba(18, 26, 48, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 8px;
    padding: 5px 8px;
}}
.pips-row {{
    display: flex;
    gap: 2px;
    align-items: center;
    height: 7px;
    border-radius: 3px;
    overflow: hidden;
}}
.pip-item {{
    flex: 1;
    height: 100%;
    cursor: pointer;
    transition: opacity 0.15s ease, transform 0.15s ease;
}}
.pip-item:hover {{
    filter: brightness(1.4);
}}
.pip-item.active-pip {{
    box-shadow: 0 0 6px #38bdf8;
    transform: scaleY(1.3);
    z-index: 5;
}}
.scrubber-slider {{
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 3px;
    background: rgba(30, 41, 59, 0.8);
    outline: none;
    cursor: pointer;
    margin: 2px 0 0 0;
}}
.scrubber-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #38bdf8;
    cursor: pointer;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.8);
    border: 2px solid #ffffff;
    transition: transform 0.1s ease;
}}
.scrubber-slider::-webkit-slider-thumb:hover {{
    transform: scale(1.3);
}}
/* Bottom Telemetry Info Strip */
.telemetry-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    color: #94a3b8;
    padding: 0 4px;
}}
.telemetry-left {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.telemetry-hint {{
    font-size: 0.68rem;
    color: #64748b;
}}
</style>
</head>
<body>
<div class="studio-box">
    <!-- Top HUD Toolbar -->
    <div class="hud-bar">
        <div class="btn-group">
            <button id="btnPrevThreat" class="hud-btn threat" title="Previous Threat (Up Arrow)">⏮ Anomaly</button>
            <button id="btnPrev" class="hud-btn" title="Previous Frame (Left Arrow)">◀</button>
            <button id="btnPlay" class="hud-btn primary" title="Play/Pause (Spacebar)">▶ Play</button>
            <button id="btnNext" class="hud-btn" title="Next Frame (Right Arrow)">▶</button>
            <button id="btnNextThreat" class="hud-btn threat" title="Next Threat (Down Arrow)">Anomaly ⏭</button>
            <select id="speedSelect" class="hud-select" title="Speed">
                <option value="150">0.7x</option>
                <option value="100" selected>1.0x (10 FPS)</option>
                <option value="60">1.6x (16 FPS)</option>
                <option value="40">2.5x (25 FPS)</option>
            </select>
        </div>

        <div class="hud-status">
            <div id="gaugeRing" class="mini-gauge">
                <div id="gaugeInner" class="mini-gauge-inner">0%</div>
            </div>
            <span id="threatScoreVal" class="hud-score">0.0000</span>
            <span id="threatBadge" class="hud-badge">SAFE BASELINE</span>
        </div>
    </div>

    <!-- Main Video Stage -->
    <div class="video-frame-container">
        <img id="mainFrameImg" class="video-img" src="" alt="CCTV Telemetry">
        <div id="roiOverlay" class="roi-float-overlay" style="display:none;">
            <span class="roi-float-badge">🎯 ROI</span>
            <span id="roiText" class="roi-float-text">TARGET: ALTERCATION [0, 0, 0, 0]</span>
        </div>
    </div>

    <!-- Slim Unified Timeline & Scrubber -->
    <div class="scrub-container">
        <div class="pips-row" id="timelineRibbon"></div>
        <input type="range" id="clipScrubber" class="scrubber-slider" min="0" max="{total_clips - 1}" value="0" step="1">
    </div>

    <!-- Bottom Telemetry Info Strip -->
    <div class="telemetry-footer">
        <div class="telemetry-left">
            <span id="frameCounter">Frame 1 / {total_clips} (Clip #1)</span>
            <span>·</span>
            <span id="timecodeDisplay">00:00 / 00:00</span>
        </div>
        <div class="telemetry-hint">
            ⌨️ [Space] Play/Pause · [← / →] Step · [↑ / ↓] Threats
        </div>
    </div>
</div>

<script>
const data = {json_str};
let currentIdx = 0;
let isPlaying = false;
let playInterval = null;
let playSpeed = 100;

// Build Pips
const ribbonEl = document.getElementById("timelineRibbon");
const weaponSet = new Set(data.weapon_clips || []);
const totalClips = data.total_clips || 1;
const totalSec = Math.round(totalClips * 16 / 30);
const totalTimeStr = `${{String(Math.floor(totalSec/60)).padStart(2, '0')}}:${{String(totalSec%60).padStart(2, '0')}}`;

for (let i = 0; i < totalClips; i++) {{
    const pip = document.createElement("div");
    pip.className = "pip-item";
    const s = data.scores[i] || 0.0;
    const isAnom = data.is_anom[i] || (s >= data.threshold);
    const isWep = weaponSet.has(i);

    if (isWep) {{
        pip.style.background = "#e11d48";
    }} else if (isAnom) {{
        pip.style.background = "#f43f5e";
    }} else {{
        pip.style.background = "#10b981";
    }}

    pip.title = `Clip ${{i+1}}: ${{s.toFixed(3)}}`;
    pip.addEventListener("click", () => updateFrame(i));
    ribbonEl.appendChild(pip);
}}

function updateFrame(idx) {{
    if (idx < 0) idx = 0;
    if (idx >= totalClips) idx = totalClips - 1;
    currentIdx = idx;

    // Scrubber
    document.getElementById("clipScrubber").value = idx;

    // Image
    const imgEl = document.getElementById("mainFrameImg");
    if (data.frames && data.frames[idx]) {{
        imgEl.src = data.frames[idx];
    }}

    // Score & Flag
    const score = (data.scores && data.scores[idx] !== undefined) ? data.scores[idx] : 0.0;
    const isAnom = (data.is_anom && data.is_anom[idx] !== undefined) ? data.is_anom[idx] : (score >= data.threshold);
    const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
    const deg = Math.round((pct / 100) * 360);

    let color = "#10b981";
    let badgeText = "SAFE BASELINE";
    let badgeBg = "rgba(16, 185, 129, 0.16)";
    let badgeBorder = "rgba(16, 185, 129, 0.35)";
    let badgeColor = "#6ee7b7";

    if (score >= 0.65) {{
        color = "#f43f5e";
        badgeText = "CRITICAL THREAT";
        badgeBg = "rgba(244, 63, 94, 0.2)";
        badgeBorder = "rgba(244, 63, 94, 0.5)";
        badgeColor = "#fca5a5";
    }} else if (score >= data.threshold) {{
        color = "#fbbf24";
        badgeText = "ELEVATED RISK";
        badgeBg = "rgba(251, 191, 36, 0.2)";
        badgeBorder = "rgba(251, 191, 36, 0.5)";
        badgeColor = "#fcd34d";
    }}

    // Gauge Ring
    const ringEl = document.getElementById("gaugeRing");
    ringEl.style.background = `conic-gradient(${{color}} ${{deg}}deg, rgba(255,255,255,0.1) ${{deg}}deg)`;
    document.getElementById("gaugeInner").innerText = pct + "%";

    // Score & Badge
    document.getElementById("threatScoreVal").innerText = score.toFixed(4);
    document.getElementById("threatScoreVal").style.color = color;

    const badgeEl = document.getElementById("threatBadge");
    badgeEl.innerText = badgeText;
    badgeEl.style.background = badgeBg;
    badgeEl.style.border = `1px solid ${{badgeBorder}}`;
    badgeEl.style.color = badgeColor;

    // ROI Float Overlay
    const roiOverlay = document.getElementById("roiOverlay");
    const roi = data.rois ? (data.rois[idx] || data.rois[String(idx)]) : null;
    if (isAnom && roi && roi.box) {{
        roiOverlay.style.display = "flex";
        const label = roi.label || "ANOMALY";
        document.getElementById("roiText").innerText = `${{label}} [${{roi.box[0]}},${{roi.box[1]}} → ${{roi.box[2]}},${{roi.box[3]}}]`;
    }} else {{
        roiOverlay.style.display = "none";
    }}

    // Footer Info & Timecode
    const curSec = Math.round(idx * 16 / 30);
    const curTimeStr = `${{String(Math.floor(curSec/60)).padStart(2, '0')}}:${{String(curSec%60).padStart(2, '0')}}`;
    document.getElementById("frameCounter").innerText = `Frame ${{idx + 1}} / ${{totalClips}} (Clip #${{idx + 1}})`;
    document.getElementById("timecodeDisplay").innerText = `${{curTimeStr}} / ${{totalTimeStr}}`;

    // Active Pip
    const pips = ribbonEl.children;
    for (let i = 0; i < pips.length; i++) {{
        if (i === idx) {{
            pips[i].classList.add("active-pip");
        }} else {{
            pips[i].classList.remove("active-pip");
        }}
    }}
}}

function togglePlay() {{
    isPlaying = !isPlaying;
    const btn = document.getElementById("btnPlay");
    if (isPlaying) {{
        btn.innerText = "⏸ Pause";
        btn.classList.add("playing");
        startPlayback();
    }} else {{
        btn.innerText = "▶ Play";
        btn.classList.remove("playing");
        stopPlayback();
    }}
}}

function startPlayback() {{
    stopPlayback();
    playInterval = setInterval(() => {{
        let nextIdx = currentIdx + 1;
        if (nextIdx >= totalClips) {{
            nextIdx = 0;
        }}
        updateFrame(nextIdx);
    }}, playSpeed);
}}

function stopPlayback() {{
    if (playInterval) {{
        clearInterval(playInterval);
        playInterval = null;
    }}
}}

function jumpNextAnomaly() {{
    for (let i = currentIdx + 1; i < totalClips; i++) {{
        const s = data.scores[i] || 0;
        if (data.is_anom[i] || s >= data.threshold) {{
            updateFrame(i);
            return;
        }}
    }}
    for (let i = 0; i <= currentIdx; i++) {{
        const s = data.scores[i] || 0;
        if (data.is_anom[i] || s >= data.threshold) {{
            updateFrame(i);
            return;
        }}
    }}
}}

function jumpPrevAnomaly() {{
    for (let i = currentIdx - 1; i >= 0; i--) {{
        const s = data.scores[i] || 0;
        if (data.is_anom[i] || s >= data.threshold) {{
            updateFrame(i);
            return;
        }}
    }}
    for (let i = totalClips - 1; i >= currentIdx; i--) {{
        const s = data.scores[i] || 0;
        if (data.is_anom[i] || s >= data.threshold) {{
            updateFrame(i);
            return;
        }}
    }}
}}

document.getElementById("clipScrubber").addEventListener("input", (e) => {{
    updateFrame(parseInt(e.target.value, 10));
}});

document.getElementById("btnPrev").addEventListener("click", () => updateFrame(currentIdx - 1));
document.getElementById("btnNext").addEventListener("click", () => updateFrame(currentIdx + 1));
document.getElementById("btnPlay").addEventListener("click", togglePlay);
document.getElementById("btnNextThreat").addEventListener("click", jumpNextAnomaly);
document.getElementById("btnPrevThreat").addEventListener("click", jumpPrevAnomaly);

document.getElementById("speedSelect").addEventListener("change", (e) => {{
    playSpeed = parseInt(e.target.value, 10);
    if (isPlaying) startPlayback();
}});

window.addEventListener("keydown", (e) => {{
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft" || e.key === "a") {{
        e.preventDefault();
        updateFrame(currentIdx - 1);
    }} else if (e.key === "ArrowRight" || e.key === "d") {{
        e.preventDefault();
        updateFrame(currentIdx + 1);
    }} else if (e.key === " ") {{
        e.preventDefault();
        togglePlay();
    }} else if (e.key === "ArrowUp") {{
        e.preventDefault();
        jumpPrevAnomaly();
    }} else if (e.key === "ArrowDown") {{
        e.preventDefault();
        jumpNextAnomaly();
    }}
}});

updateFrame(0);
</script>
</body>
</html>"""
    if hasattr(st, "iframe"):
        st.iframe(player_html, height=485, width="stretch")
    else:
        components.html(player_html, height=485, scrolling=False)


# ─────────────────────────────────────────────
# HARDWARE & MODEL STATUS SETUP
# ─────────────────────────────────────────────
device_label = "🟢 GPU Active (CUDA)" if DEVICE == "cuda" else "🟡 CPU Optimized"
ckpt_path = CHECKPOINT_DIR / "best_model.pth"
has_ckpt = ckpt_path.exists()
ckpt_label = "✅ Weights Loaded" if has_ckpt else "⚠️ Initialized Weights"
model_file = str(ckpt_path) if has_ckpt else None

# Session state initialization
if "threshold_val" not in st.session_state:
    st.session_state["threshold_val"] = float(ANOMALY_THRESHOLD)
if "selected_source_type" not in st.session_state:
    st.session_state["selected_source_type"] = "📂 Upload Video File"


# ─────────────────────────────────────────────
# REAL-WORLD TOP NAVIGATION BAR (NO SIDEBAR)
# ─────────────────────────────────────────────
st.markdown(f"""
<header class="top-nav-bar">
    <div class="nav-left">
        <div class="nav-logo-badge">🛡️</div>
        <div class="nav-brand-info">
            <div class="nav-title">
                AnomalyShield <span class="nav-badge-enterprise">Enterprise AI</span>
            </div>
            <div class="nav-subtitle">
                Autonomous CCTV Surveillance · ViT-B/16 Spatial Transformer + LSTM Temporal Threat Analysis
            </div>
        </div>
    </div>
    <div class="nav-right">
        <div class="nav-status-chip">
            <span class="status-indicator online"></span>
            <span class="chip-val" style="color: #6ee7b7;">System Operational</span>
        </div>
        <div class="nav-status-chip">
            <span class="chip-title">Hardware:</span>
            <span class="chip-val">{device_label}</span>
        </div>
        <div class="nav-status-chip">
            <span class="chip-title">Weights:</span>
            <span class="chip-val" style="color: {'#6ee7b7' if has_ckpt else '#fcd34d'}">{ckpt_label}</span>
        </div>
        <div class="nav-status-chip">
            <span class="chip-title">Backbone:</span>
            <span class="chip-val" style="color: #c7d2fe;">ViT-B/16 + Bi-LSTM</span>
        </div>
    </div>
</header>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN NAVIGATION TABS (Strictly Preserving Tabs)
# ─────────────────────────────────────────────
tab_analyze, tab_incidents, tab_sessions, tab_analytics = st.tabs([
    "🎯  Live Analysis Studio",
    "📋  Threat Incident Audit",
    "🎬  Video Session Archive",
    "📊  System Analytics & Telemetry"
])


# ══════════════════════════════════════════════
# TAB 1 · LIVE ANALYSIS STUDIO
# ══════════════════════════════════════════════
with tab_analyze:
    st.markdown("""
    <div class="console-panel">
        <div class="panel-header-strip">
            <div class="panel-title">
                <span>🎛️ Video Ingestion &amp; Neural Detection Console</span>
            </div>
            <div class="panel-sub">
                Configure source footage, model sensitivity threshold, and temporal noise filters
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ingest, col_params = st.columns([1.25, 1.0], gap="large")

    video_source_path = None
    uploaded_video_name = ""

    with col_ingest:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#a5b4fc; letter-spacing:0.04em; text-transform:uppercase; margin-bottom:10px;'>📹 Video Ingestion Source</div>", unsafe_allow_html=True)
        
        feed_mode = st.segmented_control(
            "Video Source Selector",
            options=["📂 Upload Video File", "📊 Training Dataset Archive"],
            default="📂 Upload Video File",
            label_visibility="collapsed",
            key="feed_mode_selector"
        )
        if not feed_mode:
            feed_mode = "📂 Upload Video File"

        if feed_mode == "📂 Upload Video File":
            uploaded_file = st.file_uploader(
                "Upload Surveillance Footage (MP4, AVI, MOV)",
                type=["mp4", "avi", "mov"],
                help="Supports standard CCTV formats up to 200MB."
            )
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                video_source_path = tfile.name
                uploaded_video_name = uploaded_file.name
                st.success(f"📎 File ready: **{uploaded_file.name}** ({len(uploaded_file.getbuffer())/1024/1024:.1f} MB)")

        elif feed_mode == "📊 Training Dataset Archive":
            cat_c1, cat_c2 = st.columns([1, 1.8], gap="medium")
            with cat_c1:
                sample_type = st.segmented_control(
                    "Folder Category",
                    options=["Normal", "Anomalous"],
                    default="Normal",
                    key="sample_cat_selector"
                )
                if not sample_type:
                    sample_type = "Normal"
            sample_folder = DATA_DIR / "train" / sample_type.lower()
            with cat_c2:
                if sample_folder.exists():
                    sample_vids = list(sample_folder.glob("*.avi")) + list(sample_folder.glob("*.mp4"))
                    if sample_vids:
                        sample_names = [v.name for v in sample_vids[:35]]
                        chosen_sample = st.selectbox(
                            "Select Surveillance Video",
                            sample_names,
                            index=0
                        )
                        video_source_path = str(sample_folder / chosen_sample)
                        uploaded_video_name = f"{sample_type}_{chosen_sample}"
                    else:
                        st.warning(f"No {sample_type} videos found.")
                else:
                    st.warning("Training directory not found.")

    with col_params:
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#a5b4fc; letter-spacing:0.04em; text-transform:uppercase; margin-bottom:10px;'>⚙️ Detection Parameters</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:6px;'>Sensitivity Presets:</div>", unsafe_allow_html=True)
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            if st.button("⚡ Calibrated (0.35)", use_container_width=True):
                st.session_state["threshold_val"] = 0.35
        with p_c2:
            if st.button("⚖️ Balanced (0.50)", use_container_width=True):
                st.session_state["threshold_val"] = 0.50
        with p_c3:
            if st.button("🛡️ Strict (0.65)", use_container_width=True):
                st.session_state["threshold_val"] = 0.65

        threshold = st.slider(
            "Anomaly Score Threshold",
            min_value=0.10, max_value=0.95,
            value=float(st.session_state["threshold_val"]), step=0.05,
            help="Clips scoring at or above this threshold trigger security alerts."
        )
        st.session_state["threshold_val"] = threshold

        use_ema = st.toggle(
            "EMA Temporal Filter",
            value=True,
            help="Exponential Moving Average reduces false-positive single-frame spikes."
        )

        scan_weapons = st.toggle(
            "⚔️ CCTV Weapon Detection",
            value=True,
            help="Detects handguns, blades, and acute threats with geometric false-alarm filtering."
        )

        spatial_mode = "vit"
        st.markdown("""
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.22); border-radius:8px; padding:8px 12px; margin-top:8px; font-size:0.74rem; color:#c7d2fe;">
            🔬 <b>Active Spatial Backbone:</b> ViT-B/16 (Torchvision Pretrained) · Dense 768-D Embeddings
        </div>
        """, unsafe_allow_html=True)

    # Initialize Detector
    detector = get_detector(model_file, spatial_mode)
    detector.set_threshold(threshold)
    detector.set_spatial_mode(spatial_mode)

    # Action Trigger Row
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if video_source_path:
        act_col1, act_col2 = st.columns([1.2, 3])
        with act_col1:
            run_btn = st.button("🚀  Run Neural Anomaly Analysis", type="primary", use_container_width=True)
        with act_col2:
            st.markdown(f"""
            <div style="background:rgba(18,26,48,0.7); border:1px solid rgba(148,163,184,0.14); border-radius:10px; padding:8px 16px; display:flex; align-items:center; gap:12px; height:100%;">
                <span style="font-size:1.2rem;">🎬</span>
                <div>
                    <span style="font-size:0.86rem; font-weight:700; color:#f1f5f9;">{uploaded_video_name}</span>
                    <span style="font-size:0.75rem; color:#94a3b8; margin-left:8px; font-variant-numeric:tabular-nums;">
                        Threshold: <b style="color:#fcd34d;">{threshold:.2f}</b> · EMA Temporal Filter: <b style="color:#6ee7b7;">{'ON' if use_ema else 'OFF'}</b>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if run_btn:
            prog = st.progress(0, text="Initializing Vision Transformer & Bi-LSTM inference...")

            def cb(pct, msg):
                prog.progress(int(pct * 100), text=msg)

            with st.spinner("Processing video stream through deep neural pipeline..."):
                results = detector.process_video(
                    video_source_path,
                    log_to_db=True,
                    use_ema=use_ema,
                    use_adaptive_threshold=False,
                    scan_weapons=scan_weapons,
                    progress_callback=cb
                )
            prog.empty()

            if "error" in results:
                st.error(f"Detection failed: {results['error']}")
            else:
                st.session_state["results"] = results
                st.session_state["video_name"] = uploaded_video_name
                # Pre-encode frames into base64 JPEGs once for zero-latency browser scrubbing
                ann_frames = results.get("annotated_frames", [])
                st.session_state["encoded_frames"] = _encode_frames_base64(ann_frames)
                st.session_state.pop("cached_waveform_fig", None)
                st.session_state.pop("cached_hist_fig", None)
                st.session_state.pop("cached_chart_key", None)
                st.cache_data.clear()
                st.toast("Neural Anomaly Analysis Complete!", icon="✅")

    # ── Display Results (If Available) or Real-World Empty State ──
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
        anomaly_rois = res.get("anomaly_rois", {})
        anom_count   = sum(is_anom_list)
        video_name   = st.session_state.get("video_name", "—")

        if "encoded_frames" not in st.session_state or len(st.session_state["encoded_frames"]) != len(ann_frames):
            st.session_state["encoded_frames"] = _encode_frames_base64(ann_frames)
        encoded_frames = st.session_state["encoded_frames"]

        has_weapon   = res.get("weapon_detected", False)
        anom_type    = res.get("anomaly_type", "Anomalous Activity" if is_anom else "Normal Activity")
        weapon_logs  = res.get("weapon_detections", [])

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # Threat Verdict Banner
        if has_weapon:
            st.markdown(f"""
            <div class="alert-box" style="border: 2px solid #f43f5e; background: rgba(244, 63, 94, 0.16);">
                <div class="alert-icon">🚨</div>
                <div>
                    <div class="alert-text-title" style="color:#f43f5e;">CRITICAL SECURITY THREAT: WEAPON DETECTED</div>
                    <div class="alert-text-sub">Spatial scanner confirmed <b>{len(weapon_logs)} weapon sighting(s)</b> · Peak Threat Score: <b>{max_score:.4f}</b> · Bounding boxes generated on annotated telemetry frames</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif is_anom:
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-icon">⚠️</div>
                <div>
                    <div class="alert-text-title" style="color:#fca5a5;">ANOMALOUS ACTIVITY IDENTIFIED</div>
                    <div class="alert-text-sub"><b>{anom_count} of {total_clips} temporal windows</b> exceeded anomaly threshold {threshold:.2f} · Peak Score: <b>{max_score:.4f}</b> · Classification: <b>{anom_type}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="normal-box">
                <div class="alert-icon">✅</div>
                <div>
                    <div class="alert-text-title" style="color:#6ee7b7;">VERIFIED NORMAL ROUTINE</div>
                    <div class="alert-text-sub">All <b>{total_clips} temporal clips</b> remained within safe baseline (Threshold {threshold:.2f}) · Peak Score: <b>{max_score:.4f}</b> · No weapon or violent signatures found</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # KPI Metric Cards
        avg_pct = f"{avg_score * 100:.1f}%"
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card {'red' if is_anom else 'green'}">
                <div class="kpi-label">Detection Verdict</div>
                <div class="kpi-value {'kpi-alert-value' if is_anom else 'kpi-safe-value'}">
                    {'🚨 THREAT DETECTED' if is_anom else '✅ NORMAL SCENE'}
                </div>
                <div class="kpi-sub">{video_name[:28]}</div>
            </div>
            <div class="kpi-card blue">
                <div class="kpi-label">Peak Anomaly Score</div>
                <div class="kpi-value">{max_score:.4f}</div>
                <div class="kpi-sub">Threshold delta: {max_score - threshold:+.3f}</div>
            </div>
            <div class="kpi-card cyan">
                <div class="kpi-label">Average Score</div>
                <div class="kpi-value">{avg_score:.4f}</div>
                <div class="kpi-sub">{avg_pct} of maximum scale</div>
            </div>
            <div class="kpi-card {'red' if anom_count else 'green'}">
                <div class="kpi-label">Flagged Windows</div>
                <div class="kpi-value">{anom_count} / {total_clips}</div>
                <div class="kpi-sub">{total_frames} total processed frames</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clip Timeline Heatmap
        if scores:
            st.markdown("""
            <div class="section-header">
                <div class="section-dot"></div>
                <h3>Temporal Window Threat Heatmap</h3>
            </div>
            """, unsafe_allow_html=True)

            weapon_clip_indices = set(w.get("clip_idx", -1) for w in weapon_logs)
            normal_count = total_clips - anom_count

            blocks_html = ""
            for i, (s, flag) in enumerate(zip(scores, is_anom_list)):
                intensity = int(s * 255)
                is_weapon_clip = i in weapon_clip_indices
                sec = int(i * 16 / 30)
                tc_str = f"{sec//60:02d}:{sec%60:02d}"
                if is_weapon_clip:
                    color = "#e11d48"
                    border_style = "border: 2px solid #fbbf24; box-shadow: 0 0 8px #f59e0b;"
                    title = f"Clip {i+1} ({tc_str}): score={s:.3f} · CRITICAL THREAT: WEAPON DETECTED"
                elif flag or s >= threshold:
                    color = "#f43f5e"
                    border_style = "border: 1px solid rgba(244, 63, 94, 0.45);"
                    title = f"Clip {i+1} ({tc_str}): score={s:.3f} · ANOMALOUS EVENT"
                else:
                    color = "#10b981"
                    border_style = "border: 1px solid rgba(16, 185, 129, 0.25);"
                    title = f"Clip {i+1} ({tc_str}): score={s:.3f} · NORMAL ROUTINE"
                
                blocks_html += f'<div class="timeline-clip" style="background:{color}; {border_style}" title="{title}"></div>'

            weapon_badge = (
                f" &nbsp;·&nbsp; <span style='color:#fcd34d; font-weight:700;'>⚔️ Weapon Alerts ({len(weapon_clip_indices)})</span>"
                if weapon_clip_indices else ""
            )

            total_sec = int(total_clips * 16 / 30)
            t_start = "00:00"
            t_mid = f"{total_sec//2//60:02d}:{total_sec//2%60:02d}"
            t_end = f"{total_sec//60:02d}:{total_sec%60:02d}"

            timeline_html = (
                f'<div style="background:rgba(13,19,36,0.72); border:1px solid rgba(148,163,184,0.14); '
                f'border-radius:14px; padding:16px 20px; margin-bottom:24px; box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
                f'<div style="font-size:0.8rem; color:#94a3b8; font-weight:600;">'
                f'Temporal Status: &nbsp;'
                f'<span style="color:#6ee7b7; font-weight:700;">🟢 Normal ({normal_count})</span> &nbsp;·&nbsp;'
                f'<span style="color:#f43f5e; font-weight:700;">🔴 Flagged ({anom_count})</span>'
                f'{weapon_badge}'
                f'</div>'
                f'<div style="font-size:0.74rem; color:#64748b; font-variant-numeric:tabular-nums;">'
                f'{total_clips} clips · ~{total_sec}s runtime · Hover for timecode'
                f'</div>'
                f'</div>'
                f'<div style="display:flex; flex-wrap:wrap; gap:4px; align-items:center;">'
                f'{blocks_html}'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; margin-top:8px; font-family:\'JetBrains Mono\', monospace; font-size:0.7rem; color:#64748b;">'
                f'<span>{t_start}</span><span>{t_mid}</span><span>{t_end}</span>'
                f'</div>'
                f'<div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; padding-top:10px; border-top:1px solid rgba(148,163,184,0.08); font-size:0.72rem;">'
                f'<span style="display:inline-flex; align-items:center; gap:5px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); border-radius:6px; padding:3px 9px; color:#6ee7b7; font-weight:600;"><span style="width:8px; height:8px; border-radius:2px; background:#10b981;"></span> Normal Baseline (&lt; {threshold:.2f})</span>'
                f'<span style="display:inline-flex; align-items:center; gap:5px; background:rgba(244,63,94,0.1); border:1px solid rgba(244,63,94,0.25); border-radius:6px; padding:3px 9px; color:#fca5a5; font-weight:600;"><span style="width:8px; height:8px; border-radius:2px; background:#f43f5e;"></span> Anomalous Event (≥ {threshold:.2f})</span>'
                f'{"<span style=\"display:inline-flex; align-items:center; gap:5px; background:rgba(225,29,72,0.15); border:1px solid #fbbf24; border-radius:6px; padding:3px 9px; color:#fcd34d; font-weight:700;\"><span style=\"width:8px; height:8px; border-radius:2px; background:#dc2626; border:1px solid #fbbf24;\"></span> ⚔️ Weapon Sighting</span>" if weapon_clip_indices else ""}'
                f'</div>'
                f'</div>'
            )
            with st.expander("🗺️ Comprehensive Clip Timeline Breakdown", expanded=False):
                st.markdown(timeline_html, unsafe_allow_html=True)

        # Dual-Pane Telemetry Studio
        col_vid, col_chart = st.columns([1, 1.05], gap="large")

        with col_vid:
            st.markdown("""
            <div class="section-header">
                <div class="section-dot" style="background:#06b6d4; box-shadow:0 0 8px rgba(6,182,212,0.6);"></div>
                <h3>Frame Telemetry &amp; Detection Inspector</h3>
            </div>
            """, unsafe_allow_html=True)

            if encoded_frames:
                _render_instant_telemetry_studio(
                    encoded_frames=encoded_frames,
                    scores=scores,
                    raw_scores=raw_scores,
                    is_anom_list=is_anom_list,
                    anomaly_rois=anomaly_rois,
                    threshold=threshold,
                    total_clips=total_clips,
                    weapon_clip_indices=set(w.get("clip_idx", -1) for w in weapon_logs)
                )
            else:
                st.info("No annotated frames available for telemetry.")

        with col_chart:
            st.markdown("""
            <div class="section-header">
                <div class="section-dot" style="background:#8b5cf6; box-shadow:0 0 8px rgba(139,92,246,0.6);"></div>
                <h3>Neural Score Waveform &amp; Distribution</h3>
            </div>
            """, unsafe_allow_html=True)

            chart_cache_key = (str(st.session_state.get("video_name", "")), float(threshold), len(scores))
            if st.session_state.get("cached_chart_key") != chart_cache_key:
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
                            fillcolor="rgba(244,63,94,0.08)",
                            line_width=0, layer="below"
                        )

                # EMA / Smoothed score
                line_color = "#f43f5e" if is_anom else "#10b981"
                fig.add_trace(go.Scatter(
                    x=clip_x, y=scores,
                    mode="lines+markers",
                    name="EMA Score",
                    line=dict(color=line_color, width=2.5),
                    marker=dict(
                        size=6,
                        color=["#f43f5e" if f else "#10b981" for f in is_anom_list],
                        line=dict(width=1, color="rgba(0,0,0,0.4)")
                    ),
                    fill="tozeroy",
                    fillcolor=f"rgba({'244,63,94' if is_anom else '16,185,129'},0.06)",
                    hovertemplate="Clip %{x} · Score: %{y:.4f}<extra></extra>"
                ))

                fig.add_hline(
                    y=threshold,
                    line_dash="dash", line_color="#fbbf24", line_width=1.5,
                    annotation_text=f"Threshold ({threshold:.2f})",
                    annotation_position="top left",
                    annotation_font=dict(color="#fbbf24", size=11, family="Inter, sans-serif")
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13,19,36,0.5)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    xaxis=dict(title="Temporal Clip Index (T=16 frames)",
                               gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                               tickfont=dict(size=10, family="JetBrains Mono, monospace")),
                    yaxis=dict(title="Anomaly Probability", range=[-0.05, 1.05],
                               gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                               tickfont=dict(size=10, family="JetBrains Mono, monospace")),
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                                font=dict(size=11)),
                    hovermode="x unified",
                )

                # Mini score histogram
                score_arr = np.array(scores)
                bins = np.linspace(0, 1, 22)
                counts, edges = np.histogram(score_arr, bins=bins)
                bar_colors_hist = ["#f43f5e" if (edges[i] + edges[i+1]) / 2 >= threshold else "#6366f1"
                                   for i in range(len(counts))]
                fig2 = go.Figure()
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
                    annotation_font=dict(color="#fbbf24", size=10, family="Inter, sans-serif")
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(13,19,36,0.5)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    margin=dict(l=10, r=10, t=24, b=10),
                    xaxis=dict(title="Score", gridcolor="rgba(148,163,184,0.08)", zeroline=False, range=[0, 1],
                               tickfont=dict(family="JetBrains Mono, monospace")),
                    yaxis=dict(title="# Clips", gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                               tickfont=dict(family="JetBrains Mono, monospace")),
                    showlegend=False, bargap=0.05
                )

                st.session_state["cached_waveform_fig"] = fig
                st.session_state["cached_hist_fig"] = fig2
                st.session_state["cached_chart_key"] = chart_cache_key
            else:
                fig = st.session_state["cached_waveform_fig"]
                fig2 = st.session_state["cached_hist_fig"]

            _plotly_chart(fig, height=295)
            _plotly_chart(fig2, height=165)

    elif not video_source_path:
        st.markdown("""
        <div style="background:rgba(13,19,36,0.65); border:1px solid rgba(148,163,184,0.14); border-radius:16px; padding:48px 32px; text-align:center; margin-top:20px; box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);">
            <div style="font-size:3.5rem; margin-bottom:12px;">🛡️</div>
            <div style="font-size:1.45rem; font-weight:800; color:#f1f5f9; margin-bottom:8px; letter-spacing:-0.02em;">
                Autonomous CCTV Video Intelligence Ready
            </div>
            <div style="color:#94a3b8; font-size:0.92rem; max-width:620px; margin:0 auto 28px; line-height:1.5;">
                Select a surveillance recording from the console above or drag &amp; drop a video file.
                The neural pipeline extracts dense spatial features via Vision Transformers and scores temporal anomaly dynamics in real-time.
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; max-width:880px; margin:0 auto; text-align:left;">
                <div style="background:rgba(18,26,48,0.7); border:1px solid rgba(148,163,184,0.14); border-radius:12px; padding:18px;">
                    <div style="font-size:1.35rem; margin-bottom:6px;">🔬</div>
                    <div style="font-weight:700; color:#c7d2fe; font-size:0.86rem; margin-bottom:4px;">ViT-B/16 Spatial Vision</div>
                    <div style="color:#64748b; font-size:0.75rem; line-height:1.4;">Dense 768-D patch embeddings for fine-grained spatial and weapon feature extraction.</div>
                </div>
                <div style="background:rgba(18,26,48,0.7); border:1px solid rgba(148,163,184,0.14); border-radius:12px; padding:18px;">
                    <div style="font-size:1.35rem; margin-bottom:6px;">⏱️</div>
                    <div style="font-weight:700; color:#c7d2fe; font-size:0.86rem; margin-bottom:4px;">Bi-LSTM Temporal Dynamics</div>
                    <div style="color:#64748b; font-size:0.75rem; line-height:1.4;">16-frame sliding temporal window capturing violent motions, sudden altercations, and anomalies.</div>
                </div>
                <div style="background:rgba(18,26,48,0.7); border:1px solid rgba(148,163,184,0.14); border-radius:12px; padding:18px;">
                    <div style="font-size:1.35rem; margin-bottom:6px;">🗄️</div>
                    <div style="font-weight:700; color:#c7d2fe; font-size:0.86rem; margin-bottom:4px;">Forensic Audit Logging</div>
                    <div style="color:#64748b; font-size:0.75rem; line-height:1.4;">All anomalies and video sessions logged to SQLite audit database with timestamped telemetry.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 · INCIDENT AUDIT LOG
# ══════════════════════════════════════════════
with tab_incidents:
    inc_analytics = _cached_get_incident_analytics()
    total_events  = inc_analytics["total_events"]
    max_rec       = inc_analytics["max_recorded_score"]
    avg_rec       = inc_analytics["average_recorded_score"]
    top_sources   = inc_analytics.get("top_incident_sources", [])

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Logged Incidents</div>
            <div class="kpi-value">{total_events}</div>
            <div class="kpi-sub">SQLite forensic audit trail</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Peak Recorded Score</div>
            <div class="kpi-value">{max_rec:.4f}</div>
            <div class="kpi-sub">All-time historical maximum</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Historical Average</div>
            <div class="kpi-value">{avg_rec:.4f}</div>
            <div class="kpi-sub">Mean anomaly severity</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Flagged Video Sources</div>
            <div class="kpi-value">{len(top_sources)}</div>
            <div class="kpi-sub">Distinct CCTV cameras/clips</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(13,19,36,0.72); border:1px solid rgba(148,163,184,0.14); border-radius:14px; padding:16px 20px; margin-bottom:20px; box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);">
    """, unsafe_allow_html=True)

    f_row1_col1, f_row1_col2, f_row1_col3, f_row1_col4 = st.columns([1.6, 1.4, 1.2, 0.8])
    with f_row1_col1:
        search_query = st.text_input("🔍 Search Audit Records", placeholder="Filter by video name or note...", key="inc_search")
    with f_row1_col2:
        score_range = st.slider("Anomaly Score Range", 0.0, 1.0, (0.0, 1.0), step=0.05, key="inc_score_range")
    with f_row1_col3:
        order_choice = st.selectbox(
            "Sort Order",
            ["Newest First", "Highest Score First", "Lowest Score First", "Oldest First"],
            index=0, key="inc_order"
        )
    with f_row1_col4:
        limit = st.selectbox("Page Size", [25, 50, 100, 250, 500], index=1, key="inc_limit")

    order_map = {
        "Newest First": "id DESC",
        "Highest Score First": "anomaly_score DESC",
        "Lowest Score First": "anomaly_score ASC",
        "Oldest First": "id ASC"
    }

    total_matching = _cached_count_events(
        min_score=score_range[0] if score_range[0] > 0.0 else None,
        search_text=search_query if search_query.strip() else None
    )

    col_meta1, col_meta2 = st.columns([3, 1])
    with col_meta1:
        st.caption(f"Showing up to **{limit}** records out of **{total_matching}** matching database events.")
    with col_meta2:
        if st.button("🔄 Refresh Audit Data", key="refresh_inc_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    recent_events = _cached_fetch_recent_events(
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

        def _score_color(val):
            if val >= 0.75:
                return "background-color: rgba(244,63,94,0.22); color: #fca5a5; font-weight:700"
            elif val >= 0.50:
                return "background-color: rgba(245,158,11,0.18); color: #fcd34d; font-weight:600"
            else:
                return "color: #94a3b8"

        styled = display_df.style.map(_score_color, subset=["Anomaly Score"])
        st.dataframe(styled, width="stretch", hide_index=True, height=400)

        if len(display_df) > 1:
            st.markdown("""
            <div class="section-header" style="margin-top:18px;">
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
                    color=["#f43f5e" if v >= threshold else "#10b981"
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
                annotation_font=dict(color="#fbbf24", size=10, family="Inter, sans-serif")
            )
            fig_inc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,19,36,0.5)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Recent Incidents (newest first)",
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                           tickfont=dict(family="JetBrains Mono, monospace")),
                yaxis=dict(title="Score", range=[0, 1],
                           gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                           tickfont=dict(family="JetBrains Mono, monospace")),
                showlegend=False,
            )
            _plotly_chart(fig_inc, height=200)

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Incident Audit Trail (CSV)",
            data=csv_data,
            file_name=f"anomaly_incidents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:48px 20px; background:rgba(13,19,36,0.5); border-radius:14px; border:1px solid rgba(148,163,184,0.12);">
            <div style="font-size:3rem;">🗂️</div>
            <div style="font-size:1.15rem; font-weight:700; color:#c7d2fe; margin-top:12px;">No Incidents Logged Yet</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:6px;">
                Execute video analysis in the <b style="color:#a5b4fc">Live Analysis Studio</b> to begin logging events.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    with st.expander("⚠️ Database Operations — Maintenance & Clear"):
        st.warning("Clearing logs permanently removes all historical anomaly event records.", icon="⚠️")
        if st.button("🗑️ Clear All Incident Records", type="secondary", key="clear_logs_btn"):
            db.clear_logs()
            st.cache_data.clear()
            st.success("All incident records cleared safely.")
            st.rerun()


# ══════════════════════════════════════════════
# TAB 3 · VIDEO SESSION ARCHIVE
# ══════════════════════════════════════════════
with tab_sessions:
    sess_analytics = _cached_get_session_analytics()
    db_info        = _cached_get_db_info()

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Sessions Executed</div>
            <div class="kpi-value">{sess_analytics['total_sessions']}</div>
            <div class="kpi-sub">Surveillance video evaluations</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Alert Sessions</div>
            <div class="kpi-value">{sess_analytics['alert_sessions']}</div>
            <div class="kpi-sub">Threats or anomalies flagged</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Normal Sessions</div>
            <div class="kpi-value">{sess_analytics['normal_sessions']}</div>
            <div class="kpi-sub">Verified safe routines</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Audit DB Storage</div>
            <div class="kpi-value">{db_info['size_kb']} KB</div>
            <div class="kpi-sub">{db_info['row_counts'].get('score_timeseries', 0)} timeseries data points</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rc = db_info["row_counts"]
    st.markdown(f"""
    <div style="background:rgba(13,19,36,0.72); border:1px solid rgba(148,163,184,0.14);
                border-radius:14px; padding:16px 20px; margin-bottom:20px; font-size:0.82rem; box-shadow:inset 0 1px 0 rgba(255,255,255,0.03);">
        <div style="color:#94a3b8; font-weight:700; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.06em; margin-bottom:12px;">📂 Persistent Database File Info</div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px;">
            <div><div style="color:#64748b;">DB File</div>
                 <div style="color:#c7d2fe; font-weight:600; font-family:'JetBrains Mono', monospace; font-size:0.76rem; word-break:break-all;">{db_info['path']}</div></div>
            <div><div style="color:#64748b;">video_sessions</div>
                 <div style="color:#a5b4fc; font-weight:700; font-variant-numeric:tabular-nums;">{rc.get('video_sessions',0)} records</div></div>
            <div><div style="color:#64748b;">anomaly_events</div>
                 <div style="color:#f43f5e; font-weight:700; font-variant-numeric:tabular-nums;">{rc.get('anomaly_events',0)} records</div></div>
            <div><div style="color:#64748b;">score_timeseries</div>
                 <div style="color:#6ee7b7; font-weight:700; font-variant-numeric:tabular-nums;">{rc.get('score_timeseries',0)} records</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2, s_col3 = st.columns([1, 1, 2])
    with s_col1:
        sess_limit = st.selectbox("Show Latest", [10, 25, 50, 100], index=1, key="sess_limit")
    with s_col2:
        anomaly_only = st.toggle("Alert Sessions Only", value=False, key="sess_anom_only")
    with s_col3:
        source_search = st.text_input("Filter by Video Name", placeholder="e.g. surveillance_feed_01", key="sess_source")

    sessions = _cached_fetch_sessions(
        limit=sess_limit,
        anomaly_only=anomaly_only,
        video_source=source_search if source_search else None
    )

    if sessions:
        df_sess = pd.DataFrame(sessions)

        display_sess = df_sess[[
            "id", "timestamp", "video_source", "spatial_mode",
            "total_frames", "total_clips", "anomaly_clips",
            "max_score", "avg_score", "anomaly_detected", "threshold_used"
        ]].copy()
        display_sess.columns = [
            "ID", "Timestamp", "Video Source", "Backbone",
            "Frames", "Clips", "Alert Clips",
            "Max Score", "Avg Score", "Threat Flag", "Threshold"
        ]
        display_sess["Threat Flag"] = display_sess["Threat Flag"].map({1: "🚨 THREAT", 0: "✅ SAFE"})
        display_sess["Max Score"] = display_sess["Max Score"].round(4)
        display_sess["Avg Score"] = display_sess["Avg Score"].round(4)

        def _sess_row_color(val):
            if val == "🚨 THREAT":
                return "background-color:rgba(244,63,94,0.18); color:#fca5a5; font-weight:700"
            return "color:#6ee7b7; font-weight:600"

        styled_sess = display_sess.style.map(_sess_row_color, subset=["Threat Flag"])
        st.dataframe(styled_sess, width="stretch", hide_index=True, height=360)

        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <div class="section-dot" style="background:#06b6d4;"></div>
            <h3>Historical Score Timeseries Drill-Down</h3>
        </div>
        """, unsafe_allow_html=True)

        sess_labels = [
            f"Session {s['id']} · {s['video_source'][:30]} · {s['timestamp']}"
            for s in sessions
        ]
        selected_sess_label = st.selectbox(
            "Inspect full temporal waveform for session:",
            sess_labels, key="sess_drill_select"
        )
        selected_sess_id = sessions[sess_labels.index(selected_sess_label)]["id"]
        sel_session = _cached_get_session_by_id(selected_sess_id)
        timeseries  = _cached_fetch_timeseries(selected_sess_id)

        if sel_session:
            anom_d = sel_session.get("anomaly_detected", 0)
            st.markdown(f"""
            <div style="background:rgba({'244,63,94' if anom_d else '16,185,129'},0.08);
                        border-left:4px solid {'#f43f5e' if anom_d else '#10b981'};
                        border:1px solid rgba({'244,63,94' if anom_d else '16,185,129'},0.25);
                        border-radius:10px; padding:12px 18px; margin-bottom:14px; font-size:0.83rem;">
                <b style="color:{'#fca5a5' if anom_d else '#6ee7b7'};">{'🚨 THREAT DETECTED' if anom_d else '✅ NORMAL ROUTINE'}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Source:</span> <b style="color:#c7d2fe;">{sel_session['video_source']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Backbone:</span> <b style="color:#a5b4fc;">{sel_session['spatial_mode']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Total Clips:</span> <b style="color:#e2e8f0; font-variant-numeric:tabular-nums;">{sel_session['total_clips']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Flagged Clips:</span> <b style="color:#f43f5e; font-variant-numeric:tabular-nums;">{sel_session['anomaly_clips']}</b>
                &nbsp;·&nbsp; <span style="color:#94a3b8;">Peak Score:</span> <b style="color:#fcd34d; font-variant-numeric:tabular-nums;">{sel_session['max_score']:.4f}</b>
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
            ema_colors = ["#f43f5e" if b else "#10b981" for b in df_ts["is_anomalous"].tolist()]
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
                annotation_font=dict(color="#fbbf24", size=10, family="Inter, sans-serif")
            )
            for i, row in df_ts[df_ts["is_anomalous"] == 1].iterrows():
                fig_ts.add_vrect(
                    x0=row["clip_index"] - 0.5, x1=row["clip_index"] + 0.5,
                    fillcolor="rgba(244,63,94,0.08)", line_width=0, layer="below"
                )
            fig_ts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,19,36,0.5)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Clip Index", gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                           tickfont=dict(family="JetBrains Mono, monospace")),
                yaxis=dict(title="Score", range=[-0.05, 1.05], gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                           tickfont=dict(family="JetBrains Mono, monospace")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified"
            )
            _plotly_chart(fig_ts, height=300)

        csv_sess = pd.DataFrame(sessions).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Sessions Archive (CSV)", data=csv_sess,
            file_name=f"sessions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", key="sess_csv_btn"
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:48px 20px; background:rgba(13,19,36,0.5); border-radius:14px; border:1px solid rgba(148,163,184,0.12);">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-size:1.15rem; font-weight:700; color:#c7d2fe; margin-top:12px;">No Recorded Sessions</div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:6px;">
                Analyze a video in the <b style="color:#a5b4fc">Live Analysis Studio</b> to record your first evaluation session.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 · SYSTEM ANALYTICS & TELEMETRY
# ══════════════════════════════════════════════
with tab_analytics:
    an_analytics = _cached_get_incident_analytics()
    recent_all   = _cached_fetch_recent_events(limit=500)

    if not recent_all:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; background:rgba(13,19,36,0.5); border-radius:16px; border:1px solid rgba(148,163,184,0.14);">
            <div style="font-size:3.5rem;">📊</div>
            <div style="font-size:1.2rem; font-weight:700; color:#c7d2fe; margin-top:14px;">No Telemetry Data Available</div>
            <div style="color:#64748b; font-size:0.87rem; margin-top:6px;">
                Run evaluations on CCTV footage in the Live Analysis Studio to populate real-time intelligence analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_all = pd.DataFrame(recent_all)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["hour"] = df_all["timestamp"].dt.hour
        df_all["date"] = df_all["timestamp"].dt.date

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("""
            <div class="section-header">
                <div class="section-dot"></div>
                <h3>Anomaly Score Trend Over Time</h3>
            </div>
            """, unsafe_allow_html=True)

            scores_list = df_all["anomaly_score"].tolist()
            point_colors = []
            for s in scores_list:
                if s >= 0.75:
                    point_colors.append("#f43f5e")
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
                annotation_font=dict(color="#fbbf24", size=10, family="Inter, sans-serif")
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,19,36,0.5)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                           tickfont=dict(family="JetBrains Mono, monospace")),
                yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=False, range=[0, 1],
                           tickfont=dict(family="JetBrains Mono, monospace")),
                showlegend=False
            )
            _plotly_chart(fig_trend, height=280)

            st.markdown("""
            <div class="section-header" style="margin-top:20px;">
                <div class="section-dot" style="background:#06b6d4;"></div>
                <h3>Incident Activity by Hour of Day</h3>
            </div>
            """, unsafe_allow_html=True)

            hourly = df_all.groupby("hour").size().reset_index(name="count")
            all_hours = pd.DataFrame({"hour": range(24)})
            hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
            hourly["count"] = hourly["count"].astype(int)

            h_colors = _safe_bar_colors(hourly["count"].tolist())

            fig_hour = go.Figure(go.Bar(
                x=hourly["hour"].tolist(),
                y=hourly["count"].tolist(),
                marker=dict(color=h_colors, line=dict(width=0)),
                hovertemplate="Hour %{x}:00  ·  %{y} incidents<extra></extra>"
            ))
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,19,36,0.5)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Hour of Day", gridcolor="rgba(148,163,184,0.08)",
                           zeroline=False, dtick=2, tickfont=dict(family="JetBrains Mono, monospace")),
                yaxis=dict(title="# Incidents", gridcolor="rgba(148,163,184,0.08)",
                           zeroline=False, tickfont=dict(family="JetBrains Mono, monospace")),
                showlegend=False, bargap=0.15
            )
            _plotly_chart(fig_hour, height=220)

        with col_r:
            st.markdown("""
            <div class="section-header">
                <div class="section-dot" style="background:#8b5cf6;"></div>
                <h3>Score Distribution Across All Events</h3>
            </div>
            """, unsafe_allow_html=True)

            score_vals = np.array(df_all["anomaly_score"].tolist(), dtype=float)
            h_counts, h_edges = np.histogram(score_vals, bins=25, range=(0.0, 1.0))
            h_mids   = [(h_edges[i] + h_edges[i+1]) / 2 for i in range(len(h_counts))]
            h_widths = [(h_edges[i+1] - h_edges[i]) * 0.92 for i in range(len(h_counts))]
            h_bcolors = []
            for mid in h_mids:
                if mid >= 0.75:
                    h_bcolors.append("#f43f5e")
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
                annotation_font=dict(color="#fbbf24", size=10, family="Inter, sans-serif")
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,19,36,0.5)",
                font=dict(color="#94a3b8", family="Inter, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Anomaly Score", gridcolor="rgba(148,163,184,0.08)",
                           zeroline=False, range=[0, 1], tickfont=dict(family="JetBrains Mono, monospace")),
                yaxis=dict(title="Frequency", gridcolor="rgba(148,163,184,0.08)",
                           zeroline=False, tickfont=dict(family="JetBrains Mono, monospace")),
                showlegend=False, bargap=0.04
            )
            _plotly_chart(fig_hist, height=280)

            st.markdown("""
            <div class="section-header" style="margin-top:20px;">
                <div class="section-dot" style="background:#f59e0b;"></div>
                <h3>Top Flagged Surveillance Sources</h3>
            </div>
            """, unsafe_allow_html=True)

            top_sources_data = an_analytics.get("top_incident_sources", [])
            if top_sources_data:
                df_top = pd.DataFrame(top_sources_data)
                df_top["short_name"] = df_top["video_source"].apply(
                    lambda x: ("..." + x[-26:]) if len(x) > 28 else x
                )
                count_list = df_top["count"].tolist()
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
                    plot_bgcolor="rgba(13,19,36,0.5)",
                    font=dict(color="#94a3b8", family="Inter, sans-serif"),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="# Incidents", gridcolor="rgba(148,163,184,0.08)",
                               zeroline=False, tickfont=dict(family="JetBrains Mono, monospace")),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                               autorange="reversed", tickfont=dict(family="Inter, sans-serif")),
                    showlegend=False, bargap=0.25
                )
                _plotly_chart(fig_bar, height=220)
            else:
                st.info("No flagged sources in database yet.")

        st.divider()
        st.markdown("""
        <div class="section-header">
            <div class="section-dot" style="background:#10b981;"></div>
            <h3>Summary Statistical Distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            with st.container(border=True):
                st.metric("Total Event Records", len(df_all))
        with stat_c2:
            with st.container(border=True):
                above = int((df_all["anomaly_score"] >= threshold).sum())
                st.metric("Above Active Threshold", above, delta=f"{above/max(len(df_all),1)*100:.1f}%")
        with stat_c3:
            with st.container(border=True):
                st.metric("Score Standard Dev", f"{df_all['anomaly_score'].std():.4f}")
        with stat_c4:
            with st.container(border=True):
                st.metric("Median Anomaly Score", f"{df_all['anomaly_score'].median():.4f}")


# ─────────────────────────────────────────────
# REAL-WORLD ENTERPRISE FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<footer class="website-footer">
    <div>
        <b>AnomalyShield Enterprise Platform v2.4</b> · Production Video Intelligence &amp; Surveillance Threat Detection
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
        <span class="footer-badge">⚡ PyTorch + Torchvision</span>
        <span class="footer-badge">🧠 ViT-B/16 + Bi-LSTM</span>
        <span class="footer-badge">🗄️ SQLite Audit Trail</span>
    </div>
    <div style="color:#475569; font-size:0.72rem;">
        Enterprise Security &amp; Forensic Analytics · All Rights Reserved
    </div>
</footer>
""", unsafe_allow_html=True)
