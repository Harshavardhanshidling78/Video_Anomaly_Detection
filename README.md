# Enterprise Video Anomaly Detection System (ViT + LSTM/GRU)

An intelligent surveillance system for automated real-time video anomaly detection using Vision Transformers (ViT) for spatial feature extraction, Recurrent Neural Networks (LSTM/GRU) for temporal sequence modeling, a REST API, and an interactive Streamlit dashboard.

---

## 📌 Project Overview
Continuous manual CCTV surveillance monitoring is prone to human fatigue and errors. This system provides an automated deep learning pipeline capable of:
- Ingesting live webcam/CCTV feeds or pre-recorded surveillance videos (MP4/AVI).
- Frame extraction, spatial transformation, and patch attention embedding via a **Vision Transformer (ViT)**.
- Capturing motion progression and temporal transitions across frame sequences using an **LSTM / GRU** recurrent architecture.
- Predicting continuous **Anomaly Scores** $[0.0, 1.0]$ and classifying events as **Normal** vs **Anomalous**.
- Displaying real-time video telemetry, interactive score-over-time graphs (Plotly), and logging alert incidents to an **SQLite** database.

---

## 🏗️ System Architecture Flow
```
Video Input → Frame Extraction → Preprocessing → ViT Feature Extraction → LSTM/GRU Temporal Analysis → Anomaly Classification → Anomaly Score → Dashboard / API
```

1. **Video Processing & Sampling**: Frames resized to $224 \times 224$, normalized, and grouped into sequence clips ($T=16$).
2. **Spatial Feature Extraction**: Pre-trained Vision Transformer (`ViT-B/16`) extracts 768-dimensional feature representations for each video frame.
3. **Temporal Modeling**: Recurrent LSTM/GRU model with self-attention temporal pooling aggregates sequence vectors into a temporal context representation.
4. **Classification & Scoring**: Linear classification head produces anomaly probability score $S \in [0.0, 1.0]$.
5. **Dashboard & Alert System**: Streamlit frontend displaying visual bounding status banners, real-time score graph, adjustable alert threshold slider, and SQLite incident event history.

---

## 🛠️ Requirements & Installation

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+
- OpenCV (`opencv-python`)
- Streamlit, FastAPI, Uvicorn, Plotly, Pandas, NumPy, Scikit-Learn

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### Step 1: Generate Demo Videos
Create synthetic surveillance videos (Normal walking vs Anomalous chaotic disturbance) for instant testing:
```bash
python scripts/generate_demo_data.py
```

### Step 2: Train & Fine-Tune Model
Train the recurrent temporal anomaly detector on sequence features from your dataset (`data/train/normal` and `data/train/anomalous`):
```bash
python scripts/train.py --epochs 20 --batch_size 16 --spatial_mode vit --cell_type LSTM
```

### Step 3: Evaluate Model Performance
Evaluate performance metrics (Precision, Recall, F1-Score, ROC-AUC) on test clips:
```bash
python scripts/evaluate.py --spatial_mode vit --cell_type LSTM
```

### Step 4: Benchmark Latency & Throughput
Measure inference throughput (FPS) for cold and warm feature-cached passes:
```bash
python scripts/benchmark_performance.py
```

### Step 5: Launch Interactive Web Dashboard
Run the Streamlit web dashboard application:
```bash
python -m streamlit run src/dashboard/app.py
```

### Step 6: Launch Production REST API Server
Start the FastAPI server for REST API endpoints:
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

---

## 📂 Enterprise Directory Structure

```
VideoAnomalyDetection/
├── README.md                           # Enterprise Documentation & Setup Guide
├── requirements.txt                    # Python Dependencies
├── checkpoints/                        # Model Weights Artifacts (.pth)
│   └── best_model.pth
├── data/                               # Standardized Dataset Hierarchy
│   ├── demo_videos/                    # Sample Surveillance Video Clips
│   ├── train/                          # Custom Training Dataset (2,000+ Videos)
│   │   ├── normal/                     # Normal Surveillance Videos
│   │   └── anomalous/                  # Anomaly Surveillance Videos
│   ├── test/                           # Custom Evaluation Dataset
│   │   ├── normal/
│   │   └── anomalous/
│   └── anomaly_events.db               # SQLite Audit Database
├── scripts/                            # CLI Execution Scripts
│   ├── train.py                        # Training & Fine-Tuning CLI
│   ├── evaluate.py                     # Evaluation Metrics CLI
│   ├── benchmark_performance.py        # Inference Throughput Benchmark
│   └── generate_demo_data.py           # Synthetic Demo Data Generator
├── src/                                # Core Modular Package Source Code
│   ├── config.py                       # Hyperparameters & Configurations
│   ├── api/                            # Production REST API Service
│   │   └── main.py                     # FastAPI Endpoints & App
│   ├── dashboard/                      # Production Dashboard Service
│   │   └── app.py                      # Streamlit Real-Time Dashboard
│   ├── models/                         # PyTorch Deep Learning Models
│   │   ├── vit_extractor.py            # Vision Transformer (ViT-B/16 & MobileNet)
│   │   ├── temporal_model.py           # LSTM/GRU Recurrent Sequence Network
│   │   └── anomaly_classifier.py       # End-to-End Detector Model
│   ├── preprocessing/                  # Data Ingestion Layer
│   │   └── video_processor.py          # OpenCV Frame Sampler & Clip Builder
│   ├── pipeline/                       # Pipeline Orchestrator
│   │   └── detector.py                 # Real-time Video Detector & EMA Filter
│   └── utils/                          # DB Helpers & Analytics
│       └── database.py                 # SQLite Audit Logger & Analytics
└── tests/                              # Automated Test Suite
    ├── test_models.py
    └── test_api.py
```

---

## 📝 License
Department of Computer Science & Engineering, SDM College of Engineering and Technology, Dharwad | 2026–27.
