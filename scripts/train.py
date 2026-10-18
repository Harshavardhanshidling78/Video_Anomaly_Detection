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

from src.config import DEVICE, CHECKPOINT_DIR, ANOMALY_THRESHOLD, CLIP_LEN, SPATIAL_FEATURE_DIM, DEMO_DIR, DATA_DIR
from src.preprocessing.video_processor import FastVideoProcessor
from src.models.anomaly_classifier import VideoAnomalyDetectorModel

from tqdm import tqdm

if DEVICE == "cpu":
    torch.set_num_threads(min(12, os.cpu_count() or 4))

def extract_dataset_features(spatial_mode: str = "fast", stride: int = 8, max_videos: int = None, force_reextract: bool = False):
    """
    Extracts real spatial feature embeddings from training videos with:
    - Disk caching for instant (< 0.1s) subsequent re-runs
    - tqdm progress bars for live video extraction monitoring
    - Optional max_videos limit for fast CPU prototyping
    """
    processor = FastVideoProcessor()

    train_dir = DATA_DIR / "train"
    normal_dir = train_dir / "normal"
    anom_dir = train_dir / "anomalous"

    normal_videos = list(normal_dir.glob("*.mp4")) + list(normal_dir.glob("*.avi")) + list(normal_dir.glob("*.mov")) if normal_dir.exists() else []
    anom_videos = list(anom_dir.glob("*.mp4")) + list(anom_dir.glob("*.avi")) + list(anom_dir.glob("*.mov")) if anom_dir.exists() else []

    # Apply max_videos subset limit if specified
    if max_videos and max_videos > 0:
        normal_videos = normal_videos[:max_videos]
        anom_videos = anom_videos[:max_videos]

    # Ensure training videos exist
    if not normal_videos or not anom_videos:
        raise ValueError("No training videos found in data/train/normal or data/train/anomalous. Please add video files.")


    # Check feature cache on disk
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / f"extracted_features_{spatial_mode}_{len(normal_videos)}_{len(anom_videos)}.pt"


    if cache_file.exists() and not force_reextract:
        print(f"[Info] Loading pre-extracted feature cache from {cache_file.name}...")
        cached_data = torch.load(cache_file, map_location="cpu")
        print(f"[Success] Loaded {cached_data['X'].shape[0]} feature sequence clips from cache!")
        return cached_data["X"], cached_data["Y"]

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
            if vpath == DEMO_DIR / "anomalous_event_demo.mp4":
                cutoff = int(0.3 * clip_feats.shape[0])
                labels[:cutoff] = 0.0
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


def train_model(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-4,
    cell_type: str = "LSTM",
    spatial_mode: str = "fast",
    max_videos: int = None,
    force_reextract: bool = False,
    progress_callback = None
):
    print(f"--- Training Video Anomaly Detector ({spatial_mode.upper()} + {cell_type}) ---")
    print(f"Device: {DEVICE}")

    # Step 1: Extract real features from video dataset
    X, Y = extract_dataset_features(spatial_mode=spatial_mode, max_videos=max_videos, force_reextract=force_reextract)
    num_samples = X.shape[0]

    # Shuffle dataset
    perm = torch.randperm(num_samples)
    X = X[perm]
    Y = Y[perm]

    # Split train/val (80% train, 20% val)
    split = max(1, int(0.8 * num_samples))
    train_dataset = TensorDataset(X[:split], Y[:split])
    val_dataset = TensorDataset(X[split:], Y[split:])

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, split), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(1, num_samples - split)), shuffle=False)

    # Step 2: Instantiate Model
    model = VideoAnomalyDetectorModel(spatial_mode=spatial_mode, cell_type=cell_type).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float('inf')
    save_path = CHECKPOINT_DIR / "best_model.pth"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            scores, preds = model(feats)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= max(1, total)
        train_acc = correct / max(1, total)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                scores, preds = model(feats)
                loss = criterion(scores, labels)

                val_loss += loss.item() * feats.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, train_acc, val_loss, val_acc)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            checkpoint_data = {
                "state_dict": model.state_dict(),
                "spatial_mode": spatial_mode,
                "cell_type": cell_type,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            torch.save(checkpoint_data, save_path)

    print(f"\n[Success] Training complete! Best trained model checkpoint saved to {save_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Anomaly Detector on Real Video Dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--spatial_mode", type=str, default="fast", choices=["fast", "vit"], help="Spatial feature extractor mode")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of normal/anomalous videos to process")
    parser.add_argument("--force_reextract", action="store_true", help="Force re-extraction of dataset features")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cell_type=args.cell_type,
        spatial_mode=args.spatial_mode,
        max_videos=args.max_videos,
        force_reextract=args.force_reextract
    )

