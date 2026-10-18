import os
from pathlib import Path

import sys

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
DEMO_DIR = DATA_DIR / "demo_videos"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DB_PATH = DATA_DIR / "anomaly_events.db"


# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Video Processing Configurations
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
CLIP_LEN = 16          # Number of frames per clip sequence (T)
FRAME_STRIDE = 4       # Optimized sampling stride between frames (4x faster)

# Model Parameters
SPATIAL_FEATURE_DIM = 768   # ViT Base embedding output size
TEMPORAL_HIDDEN_DIM = 256  # LSTM/GRU hidden dimension
TEMPORAL_NUM_LAYERS = 2    # Number of recurrent layers
RECURRENT_TYPE = "LSTM"    # "LSTM" or "GRU"
ANOMALY_THRESHOLD = 0.50   # Default anomaly decision boundary score [0.0 - 1.0]

# Device Setup
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
