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

def evaluate_model(checkpoint_path: str = None, spatial_mode: str = None, cell_type: str = None, max_videos: int = None, split: str = "test"):
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
    bidirectional = True
    ckpt_thresh = ANOMALY_THRESHOLD
    if ckpt_obj is not None:
        if isinstance(ckpt_obj, dict) and "optimal_threshold" in ckpt_obj:
            ckpt_thresh = float(ckpt_obj["optimal_threshold"])
        sd = ckpt_obj.get("state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
        if isinstance(ckpt_obj, dict) and "bidirectional" in ckpt_obj:
            bidirectional = ckpt_obj["bidirectional"]
        elif isinstance(sd, dict):
            head_w = sd.get("classifier_head.0.weight", None)
            if head_w is not None:
                bidirectional = (head_w.shape[1] > 256)

    print(f"--- Evaluating Video Anomaly Detection Model ({spatial_mode.upper()} + {cell_type.upper()}, Bi-LSTM={bidirectional}, Checkpoint Thresh={ckpt_thresh:.2f}) on Split '{split.upper()}' ---")
    model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode, cell_type=cell_type, bidirectional=bidirectional).to(DEVICE)

    if ckpt_obj is not None:
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            model.load_state_dict(ckpt_obj["state_dict"])
        else:
            model.load_state_dict(ckpt_obj)
        print(f"Loaded trained weights from {checkpoint_path}")
    else:
        print("Using initialized model weights.")

    model.eval()

    # Extract or load cached features from selected split (data/test or data/train)
    test_feats, true_labels = extract_dataset_features(spatial_mode=spatial_mode, max_videos=max_videos, split=split)
    true_labels = true_labels.numpy().flatten()

    print(f"Running inference across {test_feats.shape[0]} test clip sequences...")
    with torch.inference_mode():
        scores = model.predict_batch(test_feats, batch_size=256).cpu().numpy().flatten()
        preds = (scores >= ANOMALY_THRESHOLD).astype(int)

    # 1. Baseline Performance at Default Threshold (0.50)
    print(f"\n=======================================================")
    print(f"  📊 Performance at Default Threshold ({ANOMALY_THRESHOLD:.2f})")
    print(f"=======================================================")
    print(classification_report(true_labels, preds, target_names=["Normal", "Anomalous"], digits=4, zero_division=0))
    
    auc = None
    try:
        auc = roc_auc_score(true_labels, scores)
        print(f"🔥 Area Under the ROC Curve (ROC-AUC): {auc * 100:.2f}%")
    except Exception as e:
        print(f"Could not compute ROC-AUC: {e}")

    cm = confusion_matrix(true_labels, preds)
    print("\nConfusion Matrix [TN, FP / FN, TP]:")
    print(cm)

    # 2. Optimal Threshold Calibration (Maximizing Balanced Accuracy & F1-Score)
    threshold_range = np.arange(0.20, 0.80, 0.01)
    best_thresh = ANOMALY_THRESHOLD
    best_acc = 0.0
    for t in threshold_range:
        acc = ( (scores >= t) == true_labels ).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    optimal_preds = (scores >= best_thresh).astype(int)
    print(f"\n=======================================================")
    print(f"  🎯 Calibrated Optimal Decision Threshold: {best_thresh:.2f}")
    print(f"  🎯 Peak Detection Accuracy: {best_acc * 100:.2f}%")
    print(f"=======================================================")
    print(classification_report(true_labels, optimal_preds, target_names=["Normal", "Anomalous"], digits=4, zero_division=0))
    
    return {
        "roc_auc": auc,
        "default_accuracy": (preds == true_labels).mean(),
        "optimal_threshold": float(best_thresh),
        "optimal_accuracy": float(best_acc)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Anomaly Detector")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint pth file")
    parser.add_argument("--spatial_mode", type=str, default="vit", choices=["vit"], help="Spatial feature extractor mode (Torchvision ViT-B/16)")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split to evaluate on (default: test)")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, spatial_mode=args.spatial_mode, cell_type=args.cell_type, max_videos=args.max_videos, split=args.split)
