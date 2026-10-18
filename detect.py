import sys
import argparse
from pathlib import Path

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from src.config import CHECKPOINT_DIR, ANOMALY_THRESHOLD, DEVICE
from src.pipeline.detector import VideoAnomalyDetector
from src.utils.database import AnomalyDatabase


def print_banner(title: str):
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def inspect_checkpoint():
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    print_banner("MODEL CHECKPOINT INSPECTION")
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        return

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"File Path:        {ckpt_path}")
    print(f"File Size:        {size_mb:.1f} MB")

    try:
        data = torch.load(ckpt_path, map_location="cpu")
        if isinstance(data, dict):
            print(f"Spatial Backbone: {data.get('spatial_mode', 'N/A').upper()}")
            print(f"Temporal Cell:    {data.get('cell_type', 'N/A').upper()}")
            val_loss = data.get('val_loss')
            val_acc = data.get('val_acc')
            print(f"Validation Loss:  {val_loss:.4f}" if val_loss is not None else "Validation Loss:  N/A")
            print(f"Val Accuracy:     {val_acc * 100:.2f}%" if val_acc is not None else "Val Accuracy:     N/A")
            num_params = sum(p.numel() for p in data.get('state_dict', {}).values())
            print(f"Total Parameters: {num_params:,}")
        else:
            print("Format:           Raw PyTorch state_dict")
    except Exception as e:
        print(f"[Error] Could not inspect checkpoint: {e}")
    print("=" * 65)


def view_audit_logs(limit: int = 15):
    db = AnomalyDatabase()
    print_banner("SQLITE INCIDENT AUDIT TRAIL")
    analytics = db.get_incident_analytics()
    print(f"Total Logged Events:    {analytics['total_events']}")
    print(f"Max Recorded Score:     {analytics['max_recorded_score']:.3f}")
    print(f"Average Recorded Score: {analytics['average_recorded_score']:.3f}")
    print("-" * 65)

    events = db.fetch_recent_events(limit=limit)
    if not events:
        print("No incident events logged in database yet.")
    else:
        print(f"{'ID':<5} {'Timestamp':<20} {'Source':<20} {'Score':<8} {'Status'}")
        print("-" * 65)
        for ev in events:
            score = ev['anomaly_score']
            status = "ANOMALY" if score >= ev.get('threshold_used', 0.5) else "NORMAL"
            src = Path(ev['video_source']).name
            if len(src) > 18:
                src = src[:15] + "..."
            print(f"{ev['id']:<5} {ev['timestamp']:<20} {src:<20} {score:<8.3f} {status}")
    print("=" * 65)


def run_detection(video_path: str, threshold: float = ANOMALY_THRESHOLD):
    vpath = Path(video_path)
    if not vpath.exists():
        print(f"[Error] Video file not found: {video_path}")
        return

    print_banner(f"ANALYZING VIDEO: {vpath.name}")
    print(f"Hardware Device:    {DEVICE.upper()}")
    print(f"Decision Threshold: {threshold:.2f}")

    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    model_file = str(ckpt_path) if ckpt_path.exists() else None

    detector = VideoAnomalyDetector(
        model_path=model_file,
        threshold=threshold
    )

    print("\n[1/2] Ingesting frames & evaluating sequence clips...")

    from tqdm import tqdm
    pbar = tqdm(total=100, desc="Processing Video", bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")

    def progress_cb(pct, msg):
        pbar.n = int(pct * 100)
        pbar.set_postfix_str(msg)
        pbar.refresh()

    results = detector.process_video(str(vpath), log_to_db=True, progress_callback=progress_cb)
    pbar.n = 100
    pbar.set_postfix_str("Complete!")
    pbar.refresh()
    pbar.close()

    if "error" in results:
        print(f"[Error] {results['error']}")
        return

    scores = results["scores"]
    total_frames = results["total_frames"]
    total_clips = results["total_clips"]
    max_score = results["max_score"]
    avg_score = results["avg_score"]
    is_anomalous = results["anomaly_detected"]

    print("\n[2/2] Sequence Clip Telemetry:")
    print("-" * 65)
    print(f"{'Clip #':<8} {'Center Frame':<15} {'Anomaly Score':<18} {'Visual Bar'}")
    print("-" * 65)

    bar_width = 25
    for i, s in enumerate(scores):
        frame_idx = results["center_indices"][i] if i < len(results["center_indices"]) else i * 4
        filled = int(s * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        tag = " [ALERT]" if s >= threshold else ""
        print(f"{i + 1:<8} {frame_idx:<15} {s:<6.3f} {tag:<10} [{bar}]")

    print("=" * 65)
    print("DETECTION SUMMARY:")
    print(f"Total Video Frames:    {total_frames}")
    print(f"Total Sequence Clips:  {total_clips}")
    print(f"Average Anomaly Score: {avg_score:.3f}")
    print(f"Peak Anomaly Score:    {max_score:.3f}")
    print("-" * 65)

    if is_anomalous:
        print("[!] VERDICT: [ANOMALOUS EVENT DETECTED!]")
        print("    Alert recorded to SQLite database (data/anomaly_events.db).")
    else:
        print("[*] VERDICT: [NORMAL SURVEILLANCE ACTIVITY]")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Video Anomaly Detector & Inspector")
    parser.add_argument("--video", "-v", type=str, default="", help="Path to video file for anomaly detection")
    parser.add_argument("--threshold", "-t", type=float, default=ANOMALY_THRESHOLD, help="Anomaly alert threshold [0.0 - 1.0]")
    parser.add_argument("--info", "-i", action="store_true", help="Inspect trained model checkpoint metadata")
    parser.add_argument("--audit", "-a", action="store_true", help="View recent SQLite audit incident history")
    args = parser.parse_args()

    if args.info:
        inspect_checkpoint()
    elif args.audit:
        view_audit_logs()
    elif args.video:
        run_detection(args.video, threshold=args.threshold)
    else:
        # Default behavior when run with no arguments: inspect checkpoint
        inspect_checkpoint()
        print("\nCLI Usage Examples:")
        print("  python detect.py --video data/train/normal/-1l5631l3fg_0.avi")
        print("  python detect.py --video data/train/anomalous/-1l5631l3fg_0.avi")
        print("  python detect.py --audit")
        print("  python evaluate.py")
