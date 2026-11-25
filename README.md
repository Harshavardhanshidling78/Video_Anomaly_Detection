# Enterprise Video Anomaly Detection System (ViT + Bi-LSTM/GRU + YOLOv8)

An intelligent surveillance system for automated real-time video anomaly and weapon threat detection. It combines a **Vision Transformer (`ViT-B/16`)** for spatial feature extraction, a **Bidirectional LSTM / GRU** with temporal attention for sequence modeling, an optional **YOLOv8** spatial weapon/context detector, a **FastAPI** REST service, and an interactive **Streamlit** dashboard backed by **PostgreSQL / SQLite**.

---

## 📌 Project Overview

Continuous manual CCTV surveillance monitoring is prone to human fatigue and missed events. This system provides an automated dual-stream deep learning pipeline capable of:
- **Video Ingestion & Preprocessing**: Ingesting surveillance videos (MP4/AVI) via OpenCV with aspect-ratio-preserving letterboxing (`224 × 224`) and strided clip sampling (`CLIP_LEN = 16`, `FRAME_STRIDE = 4`).
- **Stream 1 — Temporal Behavioral Anomaly Detection**: Extracting 768-dimensional spatial patch embeddings via **ViT-B/16** and modeling motion progression across frame sequences using a **Bidirectional LSTM / GRU** with temporal attention pooling.
- **Stream 2 — Spatial Weapon & Threat Scanning (Optional)**: Detecting handheld weapons/threats via **YOLOv8** (`checkpoints/weapon_yolo/best.pt`) with COCO context & vehicle/hand false-positive filtering (`checkpoints/yolov8n.pt`).
- **Actor Localization & Temporal Smoothing**: Localizing anomalous regions with bounding boxes, applying EMA temporal noise filtering (`ema_alpha = 0.35`), and verifying sustained anomalies against a calibrated threshold (`ANOMALY_THRESHOLD = 0.35`).
- **Interactive Dashboard & Audit Persistence**: Providing live analysis, clip-level timeline playback, Plotly telemetry charts, and persistent incident/session logging to **PostgreSQL** (with automatic **SQLite** fallback).

---

## 🏗️ System Architecture Flow

```text
Video Input → OpenCV Frame Sampling (224×224, T=16, Stride=4)
                     │
                     ├──► Stream 1: ViT-B/16 (768-D) → Bi-LSTM/GRU + Attention → Anomaly Score [0.0, 1.0]
                     │
                     └──► Stream 2: YOLOv8 Weapon Detector + COCO Context Veto → Weapon Threat Alerts
                                           │
                                           ▼
                     EMA Smoothing + Actor Localization → FastAPI / Streamlit Dashboard → PostgreSQL / SQLite
```

1. **Video Processing & Sampling**: Frames are resized to $224 \times 224$ with letterboxing, normalized, and grouped into 16-frame clips (`FRAME_STRIDE = 4`).
2. **Spatial Feature Extraction**: Pre-trained Vision Transformer (`ViT-B/16`) extracts 768-dimensional feature representations per frame.
3. **Temporal Modeling**: Recurrent Bidirectional LSTM/GRU with self-attention temporal pooling aggregates sequence vectors into a temporal context representation.
4. **Classification, Smoothing & Localization**: Linear classification head produces anomaly probability scores $S \in [0.0, 1.0]$, smoothed via EMA and paired with spatial actor localization.
5. **Dual-Stream Weapon Scanning**: Optional YOLOv8 spatial scanner flags weapon sightings per clip and filters out benign vehicle/hand false positives.
6. **Dashboard, API & Alert Persistence**: FastAPI REST endpoints and a 4-tab Streamlit UI (*Live Analysis Studio*, *Threat Incident Audit*, *Video Session Archive*, *System Analytics & Telemetry*) backed by PostgreSQL or SQLite.

---

## 🛠️ Requirements & Installation

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+ & TorchVision 0.15+
- OpenCV (`opencv-python`), Streamlit, FastAPI, Uvicorn, Plotly, Pandas, NumPy, Scikit-Learn
- *(Optional)* `ultralytics` for YOLOv8 weapon & COCO context detection
- *(Optional)* PostgreSQL (`psycopg2-binary`) — falls back to local SQLite (`data/anomaly_events.db`) if `DATABASE_URL` is not set

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start Guide

### Option A: 1-Click Full-Stack Launch (Recommended)
Start both the **FastAPI** backend (`http://127.0.0.1:8000`) and **Streamlit** dashboard (`http://localhost:8501`) with automatic health checks and browser launch:
```bash
python run_all.py
```
On Windows, you can also double-click or run:
```bat
run_all.bat
```
Optional flags: `--host`, `--port-api`, `--port-ui`, `--no-browser`, `--timeout`.

### Option B: Launch Services Individually
- **Streamlit Web Dashboard**:
  ```bash
  python -m streamlit run src/dashboard/app.py
  ```
- **FastAPI REST Server** (Interactive Swagger docs at `http://127.0.0.1:8000/docs`):
  ```bash
  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
  ```

---

## 🧪 Training, Evaluation & CLI Workflows

### 1. Prepare Train/Test Dataset Split
Split raw videos in `data/` into `data/train` and `data/test` (generates `data/dataset_split_manifest.json`):
```bash
python scripts/split_train_test.py --test_count 250
```

### 2. Train & Fine-Tune the Temporal Model
Train the ViT + Bi-LSTM/GRU anomaly detector on `data/train/normal` and `data/train/anomalous`:
```bash
python scripts/train.py --epochs 30 --batch_size 64 --lr 3e-4 --spatial_mode vit --cell_type LSTM
```
*(Use `--no_bidirectional` for unidirectional RNNs, `--force_reextract` to rebuild cached ViT features, or `--max_videos` to cap dataset size.)*

### 3. Train / Fine-Tune the YOLOv8 Weapon Detector (Optional)
```bash
python scripts/train_weapon_detector.py --epochs 20 --batch 16 --imgsz 640
```

### 4. Evaluate Model Performance
- **Clip-level metrics** (Precision, Recall, F1-Score, ROC-AUC):
  ```bash
  python scripts/evaluate.py --split test --spatial_mode vit --cell_type LSTM
  ```
- **Per-video evaluation** (with optional `--scan-weapons` and CSV export):
  ```bash
  python scripts/evaluate_individual_videos.py --scan-weapons
  ```
- **Single video detection & checkpoint/audit inspection**:
  ```bash
  python scripts/detect.py --video path/to/video.mp4
  python scripts/detect.py --info
  python scripts/detect.py --audit
  ```
- **Batch directory evaluation**:
  ```bash
  python scripts/batch_evaluate.py
  ```

### 5. Benchmark Inference Throughput
Measure FPS for cold and warm feature-cached passes:
```bash
python scripts/benchmark_performance.py
```

### 6. (Optional) Configure PostgreSQL & Migrate SQLite Data
1. Set `DATABASE_URL` in `.env`:
   ```env
   DATABASE_URL=postgresql://username:password@localhost:5432/video_anomaly_db
   ```
2. Migrate existing historical records from SQLite to PostgreSQL:
   ```bash
   python scripts/migrate_to_postgres.py
   ```

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service health, model status, and device info |
| `POST` | `/api/v1/detect` | Upload/analyze a video for anomalies & weapon threats |
| `GET` | `/api/v1/incidents` | Retrieve logged anomaly incidents |
| `GET` | `/api/v1/sessions` | List processed video analysis sessions |
| `GET` | `/api/v1/sessions/{session_id}` | Retrieve detailed telemetry for a specific session |

---

## ✅ Running Tests

Run the automated test suite with `pytest`:
```bash
pytest
```

---

## 📂 Repository Structure

```text
VideoAnomalyDetection/
├── README.md                           # Project Documentation & Setup Guide
├── CONTRIBUTING.md                     # Contribution Guidelines
├── requirements.txt                    # Python Dependencies
├── pytest.ini                          # Pytest Configuration
├── run_all.py                          # Full-Stack Launcher (FastAPI + Streamlit)
├── run_all.bat                         # Windows 1-Click Launch Script
├── checkpoints/                        # Model Weights Artifacts
│   ├── best_model.pth                  # Trained ViT + Bi-LSTM/GRU Checkpoint
│   ├── yolov8n.pt                      # YOLOv8 COCO Context Detector
│   └── weapon_yolo/best.pt             # YOLOv8 Weapon Threat Detector
├── data/                               # Dataset & Persistence Artifacts
│   ├── dataset_split_manifest.json     # Train/Test Split Manifest
│   ├── train/                          # Training Videos (normal/ & anomalous/)
│   ├── test/                           # Evaluation Videos (normal/ & anomalous/)
│   ├── processed/                      # Cached ViT Feature Tensors (.pt)
│   └── anomaly_events.db               # SQLite Audit Database (when PostgreSQL unset)
├── scripts/                            # CLI Execution Scripts
│   ├── split_train_test.py             # Dataset Train/Test Splitter
│   ├── train.py                        # ViT + LSTM/GRU Training CLI
│   ├── train_weapon_detector.py        # YOLOv8 Weapon Detector Training CLI
│   ├── evaluate.py                     # Dataset Metrics Evaluation CLI
│   ├── evaluate_individual_videos.py   # Per-Video Evaluation & CSV Reporter
│   ├── batch_evaluate.py               # Batch Video Directory Scanner
│   ├── detect.py                       # Single Video Detection & Inspection CLI
│   ├── benchmark_performance.py        # Inference Throughput Benchmark
│   └── migrate_to_postgres.py          # SQLite to PostgreSQL Migration CLI
├── src/                                # Core Modular Package Source Code
│   ├── config.py                       # Hyperparameters & Environment Config
│   ├── api/
│   │   └── main.py                     # FastAPI REST Service
│   ├── dashboard/
│   │   └── app.py                      # Streamlit 4-Tab Real-Time Dashboard
│   ├── models/
│   │   ├── vit_extractor.py            # Vision Transformer (ViT-B/16 Spatial Extractor)
│   │   ├── temporal_model.py           # Bi-LSTM/GRU Recurrent Sequence Network
│   │   └── anomaly_classifier.py       # End-to-End Detector Model
│   ├── preprocessing/
│   │   └── video_processor.py          # OpenCV Frame Sampler & Clip Builder
│   ├── pipeline/
│   │   └── detector.py                 # Dual-Stream Detector, Localization & EMA Filter
│   └── utils/
│       └── database.py                 # PostgreSQL / SQLite Audit Logger & Analytics
└── tests/                              # Automated Test Suite
    ├── test_models.py
    ├── test_api.py
    ├── test_actor_localization.py
    ├── test_clip_timeline.py
    └── test_weapon_detection.py
```

---

## 📝 License
Department of Computer Science & Engineering, SDM College of Engineering and Technology, Dharwad | 2026–27.
