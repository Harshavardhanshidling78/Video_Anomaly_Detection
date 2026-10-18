import sys
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.train import train_model, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Video Anomaly Detector on Real Video Dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--spatial_mode", type=str, default="fast", choices=["fast", "vit"], help="Spatial feature extractor mode")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of normal/anomalous videos to process")
    parser.add_argument("--force_reextract", action="store_true", help="Force re-extraction of video features")
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

