import sys
import argparse
import torch
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.config import DEVICE, CHECKPOINT_DIR, ANOMALY_THRESHOLD
from src.models.anomaly_classifier import VideoAnomalyDetectorModel
from scripts.train import extract_dataset_features

def evaluate_model(checkpoint_path: str = None, spatial_mode: str = None, cell_type: str = None, max_videos: int = 50):
    if not checkpoint_path:
        checkpoint_path = str(CHECKPOINT_DIR / "best_model.pth")

    ckpt_obj = None
    if Path(checkpoint_path).exists():
        try:
            ckpt_obj = torch.load(checkpoint_path, map_location=DEVICE)
            if isinstance(ckpt_obj, dict):
                if spatial_mode is None:
                    spatial_mode = ckpt_obj.get("spatial_mode", "vit")
                if cell_type is None:
                    cell_type = ckpt_obj.get("cell_type", "LSTM")
        except Exception as e:
            print(f"[Warning] Failed reading checkpoint ({e})")

    spatial_mode = spatial_mode or "vit"
    cell_type = cell_type or "LSTM"

    print(f"--- Evaluating Video Anomaly Detection Model ({spatial_mode.upper()} + {cell_type.upper()}) ---")
    model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode, cell_type=cell_type).to(DEVICE)

    if ckpt_obj is not None:
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            model.load_state_dict(ckpt_obj["state_dict"])
        else:
            model.load_state_dict(ckpt_obj)
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print("Using initialized model weights.")

    model.eval()

    # Extract or load cached test features from video clips
    test_feats, true_labels = extract_dataset_features(spatial_mode=spatial_mode, stride=4, max_videos=max_videos)
    test_feats = test_feats.to(DEVICE)
    true_labels = true_labels.numpy().flatten()

    with torch.no_grad():
        scores, preds = model(test_feats)
        scores = scores.cpu().numpy().flatten()
        preds = preds.cpu().numpy().flatten()

    print("\n--- Model Evaluation Results ---")
    print(classification_report(true_labels, preds, target_names=["Normal", "Anomalous"], zero_division=0))
    try:
        auc = roc_auc_score(true_labels, scores)
        print(f"ROC-AUC Score: {auc:.4f}")
    except Exception:
        pass
    
    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Anomaly Detector")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint pth file")
    parser.add_argument("--spatial_mode", type=str, default="fast", choices=["fast", "vit"], help="Spatial feature extractor mode")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to evaluate")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, spatial_mode=args.spatial_mode, cell_type=args.cell_type, max_videos=args.max_videos)
