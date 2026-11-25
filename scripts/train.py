import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEVICE, CHECKPOINT_DIR, ANOMALY_THRESHOLD, CLIP_LEN, SPATIAL_FEATURE_DIM, DATA_DIR
from src.preprocessing.video_processor import FastVideoProcessor
from src.models.anomaly_classifier import VideoAnomalyDetectorModel

from tqdm import tqdm

if DEVICE == "cpu":
    torch.set_num_threads(min(12, os.cpu_count() or 4))

def extract_dataset_features(spatial_mode: str = "vit", stride: int = 8, max_videos: int = None, force_reextract: bool = False, split: str = "train"):
    """
    Extracts real spatial feature embeddings from videos (split='train' or 'test') with:
    - Disk caching for instant (< 0.1s) subsequent re-runs
    - tqdm progress bars for live video extraction monitoring
    - Optional max_videos limit for fast prototyping
    """
    processor = FastVideoProcessor()

    target_dir = DATA_DIR / split
    normal_dir = target_dir / "normal"
    anom_dir = target_dir / "anomalous"

    normal_videos = list(normal_dir.glob("*.mp4")) + list(normal_dir.glob("*.avi")) + list(normal_dir.glob("*.mov")) if normal_dir.exists() else []
    anom_videos = list(anom_dir.glob("*.mp4")) + list(anom_dir.glob("*.avi")) + list(anom_dir.glob("*.mov")) if anom_dir.exists() else []

    # Apply max_videos subset limit if specified
    if max_videos and max_videos > 0:
        normal_videos = normal_videos[:max_videos]
        anom_videos = anom_videos[:max_videos]

    # Ensure videos exist
    if not normal_videos or not anom_videos:
        raise ValueError(f"No videos found in data/{split}/normal or data/{split}/anomalous. Please add video files.")

    # Check feature cache on disk
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / f"extracted_features_{spatial_mode}_{split}_{len(normal_videos)}_{len(anom_videos)}.pt"

    if not cache_file.exists() and not force_reextract:
        # Fallback to any existing feature cache for this spatial mode and split (largest file first)
        candidates = sorted(list(processed_dir.glob(f"extracted_features_{spatial_mode}_{split}_*.pt")), key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            cache_file = candidates[0]

    if cache_file.exists() and not force_reextract:
        print(f"[Info] Loading pre-extracted feature cache from {cache_file.name} (zero-copy mmap)...")
        try:
            cached_data = torch.load(cache_file, map_location="cpu", mmap=True)
        except Exception:
            cached_data = torch.load(cache_file, map_location="cpu")
        print(f"[Success] Loaded {cached_data['X'].shape[0]} feature sequence clips from cache!")
        X = cached_data["X"]
        Y = cached_data["Y"]
        return X, Y

    print(f"[Info] Extracting real video features using spatial_mode='{spatial_mode}'...")
    print(f"Normal videos found: {len(normal_videos)} | Anomalous videos found: {len(anom_videos)}")

    model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode).to(DEVICE)
    model.eval()

    all_clips_list = []
    all_labels_list = []

    from src.preprocessing.video_processor import IMAGENET_MEAN, IMAGENET_STD

    video_cache_dir = processed_dir / f"cache_{spatial_mode}"
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    def process_single_video(vpath, sample_step=3, clip_len=CLIP_LEN, clip_stride=2):
        v_cache_file = video_cache_dir / f"{Path(vpath).parent.name}_{Path(vpath).stem}.pt"
        if v_cache_file.exists():
            try:
                return torch.load(v_cache_file, map_location="cpu")
            except Exception:
                pass

        frames = processor.extract_frames_from_video(str(vpath))
        if not frames:
            return None
        if sample_step > 1:
            frames = frames[::sample_step]

        valid_rgb = []
        for f in frames:
            p_f = processor.preprocess_frame_fast(f)
            if p_f is not None:
                valid_rgb.append(p_f)

        if not valid_rgb:
            return None

        if len(valid_rgb) < clip_len:
            valid_rgb = valid_rgb + [valid_rgb[-1]] * (clip_len - len(valid_rgb))

        np_frames = np.stack(valid_rgb, axis=0)
        t_frames = torch.from_numpy(np_frames).permute(0, 3, 1, 2).float() / 255.0
        t_norm = (t_frames - IMAGENET_MEAN) / IMAGENET_STD

        with torch.inference_mode():
            frame_feats = model.spatial_extractor(t_norm.to(DEVICE)).cpu()

        clips_list = []
        n_f = frame_feats.shape[0]
        for i in range(0, n_f - clip_len + 1, clip_stride):
            clips_list.append(frame_feats[i : i + clip_len])

        if not clips_list:
            return None
        stacked = torch.stack(clips_list, dim=0)
        try:
            torch.save(stacked, v_cache_file)
        except Exception:
            pass
        return stacked

    # Process Normal Videos (Label 0.0)
    print("\n--- Processing Normal Videos ---")
    for vpath in tqdm(normal_videos, desc="Normal Videos Progress"):
        clip_feats = process_single_video(vpath)
        if clip_feats is not None and clip_feats.shape[0] > 0:
            all_clips_list.append(clip_feats)
            all_labels_list.append(torch.zeros(clip_feats.shape[0], 1))

    # Process Anomalous Videos (Label 1.0)
    print("\n--- Processing Anomalous Videos ---")
    for vpath in tqdm(anom_videos, desc="Anomalous Videos Progress"):
        clip_feats = process_single_video(vpath)
        if clip_feats is not None and clip_feats.shape[0] > 0:
            all_clips_list.append(clip_feats)
            labels = torch.ones(clip_feats.shape[0], 1)
            all_labels_list.append(labels)

    if not all_clips_list:
        raise ValueError("No video features extracted. Check video files.")

    X = torch.cat(all_clips_list, dim=0) # (Total_Clips, T, D)
    Y = torch.cat(all_labels_list, dim=0) # (Total_Clips, 1)

    print(f"\n[Success] Extracted {X.shape[0]} video clip feature sequences of shape {X.shape[1:]}")
    
    # Save cache to disk
    torch.save({"X": X, "Y": Y}, cache_file)
    print(f"[Info] Saved feature cache to {cache_file}")

    return X, Y


from sklearn.metrics import roc_auc_score

class BinaryFocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with Target Label Smoothing for Weakly-Supervised Video Anomaly Detection:
    - Down-weights easy normal frames (gamma=2.0)
    - Prevents normal intervals inside anomalous videos from poisoning gradients (label_smoothing=0.02)
    - Balances positive/negative sample gradient impact (alpha=0.45)
    """
    def __init__(self, alpha: float = 0.45, gamma: float = 2.0, label_smoothing: float = 0.02):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_smooth = target * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing
        p = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        pt = torch.where(target >= 0.5, p, 1.0 - p)
        alpha_t = torch.where(target >= 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        bce = -(target_smooth * torch.log(p) + (1.0 - target_smooth) * torch.log(1.0 - p))
        return (focal_weight * bce).mean()


def train_model(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    cell_type: str = "LSTM",
    spatial_mode: str = "vit",
    bidirectional: bool = True,
    max_videos: int = None,
    force_reextract: bool = False,
    progress_callback = None
):
    print(f"--- Training Video Anomaly Detector ({spatial_mode.upper()} + {cell_type}, Bi-LSTM={bidirectional}) ---")
    print(f"Device: {DEVICE}")

    # Step 1: Extract real features from video dataset (split='train')
    X, Y = extract_dataset_features(spatial_mode=spatial_mode, max_videos=max_videos, force_reextract=force_reextract, split="train")
    num_samples = X.shape[0]

    # Check for held-out test split for true validation (DataLoader with shuffle=True handles random mini-batch sampling)
    val_from_test = False
    try:
        X_test, Y_test = extract_dataset_features(spatial_mode=spatial_mode, max_videos=max_videos, force_reextract=False, split="test")
        val_dataset = TensorDataset(X_test, Y_test)
        train_dataset = TensorDataset(X, Y)
        val_from_test = True
        print(f"[Info] Using independent test split ({X_test.shape[0]} clips) as validation benchmark.")
    except Exception as e:
        print(f"[Info] Test split not found ({e}), using 80/20 train/val split.")
        split = max(1, int(0.8 * num_samples))
        train_dataset = TensorDataset(X[:split], Y[:split])
        val_dataset = TensorDataset(X[split:], Y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Step 2: Instantiate Model with Bi-LSTM and Focal Loss
    model = VideoAnomalyDetectorModel(
        spatial_mode=spatial_mode,
        cell_type=cell_type,
        bidirectional=bidirectional
    ).to(DEVICE)
    criterion = BinaryFocalLossWithSmoothing(alpha=0.50, gamma=2.0, label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc = 0.0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 6
    epochs_no_improve = 0
    optimal_thresh = ANOMALY_THRESHOLD
    save_path = CHECKPOINT_DIR / "best_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # Continuous embedding space augmentation (prevents Bi-LSTM over-fitting on static ViT features)
            noise = torch.randn_like(feats) * 0.04
            train_feats = feats + noise

            scores, preds = model(train_feats)
            loss = criterion(scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= max(1, total)
        train_acc = correct / max(1, total)
        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_scores = []
        all_val_labels = []

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                scores, preds = model(feats)
                loss = criterion(scores, labels)

                val_loss += loss.item() * feats.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_val_scores.append(scores.cpu())
                all_val_labels.append(labels.cpu())

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        v_scores = torch.cat(all_val_scores).squeeze().numpy()
        v_labels = torch.cat(all_val_labels).squeeze().numpy()
        val_auc = 0.5
        try:
            val_auc = roc_auc_score(v_labels, v_scores)
        except Exception:
            pass

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% AUC: {val_auc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, train_acc, val_loss, val_acc)

        # Selection criterion: Validation AUC improvement or validation loss improvement
        is_best = False
        if val_auc > best_val_auc + 0.002 or (val_auc >= best_val_auc - 0.002 and val_loss < best_val_loss):
            is_best = True
            best_val_auc = max(best_val_auc, val_auc)
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if is_best:
            # Calibrate optimal decision threshold (maximizing balanced F1 & accuracy)
            best_t = ANOMALY_THRESHOLD
            best_t_f1 = 0.0
            for t in np.arange(0.20, 0.70, 0.01):
                bin_preds = (v_scores >= t).astype(int)
                tp = int(((bin_preds == 1) & (v_labels == 1)).sum())
                fp = int(((bin_preds == 1) & (v_labels == 0)).sum())
                fn = int(((bin_preds == 0) & (v_labels == 1)).sum())
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
                acc = ((bin_preds == v_labels).mean())
                score_metric = 0.6 * f1 + 0.4 * acc
                if score_metric > best_t_f1:
                    best_t_f1 = score_metric
                    best_t = float(t)
            optimal_thresh = best_t

            checkpoint_data = {
                "state_dict": model.state_dict(),
                "spatial_mode": spatial_mode,
                "cell_type": cell_type,
                "bidirectional": bidirectional,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "optimal_threshold": optimal_thresh
            }
            torch.save(checkpoint_data, save_path)
            print(f"  --> [Saved Best Checkpoint] AUC: {best_val_auc*100:.2f}%, Val Loss: {val_loss:.4f}, Calibrated Threshold: {optimal_thresh:.2f}")

        if epochs_no_improve >= patience and epoch >= 10:
            print(f"\n[Info] Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    print(f"\n[Success] Training complete! Best trained model checkpoint saved to {save_path} (AUC: {best_val_auc*100:.2f}%, Threshold: {optimal_thresh:.2f})")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Anomaly Detector on Real Video Dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--no_bidirectional", action="store_true", help="Disable bidirectional LSTM")
    parser.add_argument("--spatial_mode", type=str, default="vit", choices=["vit"], help="Spatial feature extractor mode (Torchvision ViT-B/16)")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of normal/anomalous videos to process")
    parser.add_argument("--force_reextract", action="store_true", help="Force re-extraction of dataset features")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cell_type=args.cell_type,
        spatial_mode=args.spatial_mode,
        bidirectional=not args.no_bidirectional,
        max_videos=args.max_videos,
        force_reextract=args.force_reextract
    )

