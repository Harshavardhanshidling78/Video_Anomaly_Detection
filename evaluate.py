import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.evaluate import evaluate_model, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Video Anomaly Detector")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint pth file")
    parser.add_argument("--spatial_mode", type=str, default="fast", choices=["fast", "vit"], help="Spatial feature extractor mode")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="Temporal RNN cell type")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to evaluate")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, spatial_mode=args.spatial_mode, cell_type=args.cell_type, max_videos=args.max_videos)
