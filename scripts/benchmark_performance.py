import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import DEMO_DIR
from src.pipeline.detector import VideoAnomalyDetector

def benchmark():
    video_path = DEMO_DIR / "anomalous_event_demo.mp4"
    if not video_path.exists():
        from scripts.generate_demo_data import create_demo_videos
        create_demo_videos()



    detector = VideoAnomalyDetector(spatial_mode="fast")
    
    print(f"--- Benchmarking High-Performance Video Anomaly Backend ---")
    print(f"Target Video: {video_path}")
    
    # 1. Cold Pass (Reading video + fast feature extraction + model evaluation)
    t0 = time.perf_counter()
    results1 = detector.process_video(str(video_path), log_to_db=False, stride=6)
    t1 = time.perf_counter()

    elapsed1 = t1 - t0
    total_frames = results1["total_frames"]
    fps1 = total_frames / elapsed1 if elapsed1 > 0 else 0

    print("\n--- Cold Pass Performance Results ---")
    print(f"Total Video Frames Processed: {total_frames}")
    print(f"Total Sequence Clips Evaluated: {results1['total_clips']}")
    print(f"Total Cold Pass Time: {elapsed1:.3f} seconds")
    print(f"Cold Throughput: {fps1:.1f} FPS")

    # 2. Warm Pass (Using feature cache)
    t2 = time.perf_counter()
    results2 = detector.process_video(str(video_path), log_to_db=False, stride=6)
    t3 = time.perf_counter()

    elapsed2 = t3 - t2
    fps2 = total_frames / elapsed2 if elapsed2 > 0 else 0

    print("\n--- Warm Cached Pass Performance Results ---")
    print(f"Total Cached Pass Time: {elapsed2:.4f} seconds")
    print(f"Cached Throughput: {fps2:.1f} FPS")

if __name__ == "__main__":
    benchmark()
