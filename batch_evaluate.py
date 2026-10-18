import sys
import argparse
import csv
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from src.config import CHECKPOINT_DIR, ANOMALY_THRESHOLD, DEVICE
from src.pipeline.detector import VideoAnomalyDetector

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}


def collect_videos(folder: Path):
    """Recursively collect all video files from a directory."""
    videos = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(folder.rglob(f"*{ext}"))
        videos.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(videos))


def print_header():
    print()
    print("=" * 90)
    print("   VIDEO ANOMALY DETECTION - BATCH EVALUATION REPORT")
    print("=" * 90)
    print(f"  {'#':<6} {'VIDEO NAME':<45} {'SCORE':<8} {'VERDICT':<12} {'TIME(s)'}")
    print("-" * 90)


def print_row(idx: int, name: str, score: float, verdict: str, elapsed: float):
    # Truncate long names
    if len(name) > 43:
        name = name[:40] + "..."
    verdict_display = "[ANOMALY]" if verdict == "ANOMALY" else "[ NORMAL ]"
    print(f"  {idx:<6} {name:<45} {score:<8.3f} {verdict_display:<12} {elapsed:.1f}s")


def print_summary(results: list, total_time: float, output_csv: str = None):
    anomaly_count = sum(1 for r in results if r["verdict"] == "ANOMALY")
    normal_count  = sum(1 for r in results if r["verdict"] == "NORMAL")
    error_count   = sum(1 for r in results if r["verdict"] == "ERROR")
    total         = len(results)

    print("-" * 90)
    print()
    print("=" * 90)
    print("   BATCH SUMMARY")
    print("=" * 90)
    print(f"  Total Videos Scanned : {total}")
    print(f"  Anomalous Videos      : {anomaly_count}  ({anomaly_count/max(1,total)*100:.1f}%)")
    print(f"  Normal Videos         : {normal_count}  ({normal_count/max(1,total)*100:.1f}%)")
    if error_count:
        print(f"  Errors / Skipped      : {error_count}")
    print(f"  Total Time Taken      : {total_time:.1f}s  (~{total_time/max(1,total):.1f}s per video)")
    print("=" * 90)

    if output_csv:
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "video_name", "full_path", "score", "verdict", "time_sec"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  [Saved] Results exported to: {output_csv}")
        except Exception as e:
            print(f"\n  [Warning] Could not save CSV: {e}")
    print()


def run_batch(folder: str, threshold: float, output_csv: str = None, limit: int = None):
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"[Error] Folder not found: {folder}")
        sys.exit(1)

    print(f"\n  Scanning folder : {folder_path.resolve()}")
    videos = collect_videos(folder_path)

    if not videos:
        print(f"[Error] No video files found in: {folder_path}")
        sys.exit(1)

    if limit:
        videos = videos[:limit]

    print(f"  Videos Found    : {len(videos)}")
    print(f"  Threshold       : {threshold:.2f}")
    print(f"  Device          : {DEVICE.upper()}")

    # Load model once - reuse for all videos
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    model_file = str(ckpt_path) if ckpt_path.exists() else None
    if not model_file:
        print(f"\n[Error] No trained model found at: {ckpt_path}")
        print("  Please run: python train.py  first.")
        sys.exit(1)

    print(f"  Model Checkpoint: {ckpt_path.name}  ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)")

    detector = VideoAnomalyDetector(model_path=model_file, threshold=threshold)

    print_header()

    all_results = []
    batch_start = time.time()

    for idx, vpath in enumerate(videos, start=1):
        video_name = vpath.name
        t0 = time.time()

        try:
            results = detector.process_video(str(vpath), log_to_db=False)
            elapsed = time.time() - t0

            if "error" in results:
                verdict = "ERROR"
                score   = 0.0
                print_row(idx, video_name, score, "ERROR", elapsed)
            else:
                score   = results["max_score"]
                verdict = "ANOMALY" if results["anomaly_detected"] else "NORMAL"
                print_row(idx, video_name, score, verdict, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            verdict = "ERROR"
            score   = 0.0
            print_row(idx, video_name + f" [ERR: {str(e)[:20]}]", score, "ERROR", elapsed)

        all_results.append({
            "index":      idx,
            "video_name": video_name,
            "full_path":  str(vpath),
            "score":      round(score, 4),
            "verdict":    verdict,
            "time_sec":   round(time.time() - t0, 2)
        })

    total_time = time.time() - batch_start
    print_summary(all_results, total_time, output_csv=output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Video Anomaly Evaluator - Scan an entire folder of videos",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        required=True,
        help="Path to folder containing videos (scans subfolders too)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=ANOMALY_THRESHOLD,
        help=f"Anomaly decision threshold [0.0 - 1.0] (default: {ANOMALY_THRESHOLD})"
    )
    parser.add_argument(
        "--csv", "-c",
        type=str,
        default=None,
        help="Optional: export results to a CSV file (e.g. results.csv)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Optional: limit to first N videos (for quick testing)"
    )
    args = parser.parse_args()
    run_batch(args.folder, args.threshold, output_csv=args.csv, limit=args.limit)
