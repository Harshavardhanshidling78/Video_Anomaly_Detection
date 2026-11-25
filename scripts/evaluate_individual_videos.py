import sys
import argparse
import time
import csv
import gc
from pathlib import Path
from typing import Dict, List, Any

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import DATA_DIR, CHECKPOINT_DIR, ANOMALY_THRESHOLD, DEVICE
from src.pipeline.detector import VideoAnomalyDetector

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def collect_test_dataset(test_dir: Path) -> List[Dict[str, Any]]:
    """Collect all test videos with ground-truth labels."""
    dataset = []
    normal_dir = test_dir / "normal"
    anom_dir = test_dir / "anomalous"

    if normal_dir.exists():
        for ext in SUPPORTED_EXTENSIONS:
            for p in sorted(normal_dir.glob(f"*{ext}")):
                dataset.append({"path": p, "true_label": "NORMAL", "label_id": 0})

    if anom_dir.exists():
        for ext in SUPPORTED_EXTENSIONS:
            for p in sorted(anom_dir.glob(f"*{ext}")):
                dataset.append({"path": p, "true_label": "ANOMALOUS", "label_id": 1})

    return dataset


def print_banner(total_videos: int, normal_cnt: int, anom_cnt: int, threshold: float):
    print("=" * 110)
    print("       VIDEO ANOMALY DETECTION - INDIVIDUAL VIDEO BENCHMARK & ACCURACY REPORT")
    print("=" * 110)
    print(f"  Total Test Videos : {total_videos} ({normal_cnt} Normal, {anom_cnt} Anomalous)")
    print(f"  Decision Threshold: {threshold:.2f}")
    print(f"  Inference Device  : {DEVICE.upper()}")
    print("=" * 110)
    print(f"  {'#':<5} {'VIDEO FILENAME':<38} {'GROUND TRUTH':<14} {'PREDICTION':<13} {'MAX SCORE':<11} {'MATCH':<9} {'TIME'}")
    print("-" * 110)


def print_video_row(idx: int, filename: str, true_label: str, pred_label: str, max_score: float, is_correct: bool, elapsed: float):
    short_name = filename if len(filename) <= 36 else filename[:33] + "..."
    match_str = "[OK]" if is_correct else "[FAIL]"
    print(f"  {idx:<5} {short_name:<38} {true_label:<14} {pred_label:<13} {max_score:<11.4f} {match_str:<9} {elapsed:.1f}s", flush=True)


def evaluate_individual_videos(
    test_dir: str = None,
    threshold: float = ANOMALY_THRESHOLD,
    output_csv: str = None,
    limit_per_class: int = None,
    resume: bool = True,
    scan_weapons: bool = False
):
    test_path = Path(test_dir) if test_dir else DATA_DIR / "test"
    if not test_path.exists():
        print(f"[Error] Test directory does not exist: {test_path}")
        sys.exit(1)

    csv_path = Path(output_csv) if output_csv else DATA_DIR / "individual_video_eval_results.csv"

    # Collect dataset
    all_items = collect_test_dataset(test_path)
    if not all_items:
        print(f"[Error] No videos found in {test_path}/normal or {test_path}/anomalous")
        sys.exit(1)

    # Filter by limit_per_class if requested
    if limit_per_class and limit_per_class > 0:
        normals = [x for x in all_items if x["label_id"] == 0][:limit_per_class]
        anomalies = [x for x in all_items if x["label_id"] == 1][:limit_per_class]
        all_items = normals + anomalies

    normal_cnt = sum(1 for x in all_items if x["label_id"] == 0)
    anom_cnt = sum(1 for x in all_items if x["label_id"] == 1)

    # Check for resume support (skip only successful evaluations, retry errors)
    completed_results = {}
    if resume and csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("classification_type") != "ERROR":
                        completed_results[row["video_name"]] = {
                            "index": int(row.get("index", 0)),
                            "video_name": row["video_name"],
                            "true_label": row["true_label"],
                            "pred_label": row["pred_label"],
                            "true_id": int(row["true_id"]),
                            "pred_id": int(row["pred_id"]),
                            "is_correct": row["is_correct"] == "True",
                            "classification_type": row["classification_type"],
                            "max_score": float(row["max_score"]),
                            "avg_score": float(row["avg_score"]),
                            "anom_ratio": float(row["anom_ratio"]),
                            "time_sec": float(row["time_sec"])
                        }
            if completed_results:
                print(f"[Resume] Found {len(completed_results)} successfully analyzed videos in {csv_path.name}")
        except Exception as e:
            print(f"[Warning] Could not parse existing CSV for resume: {e}")

    # Prepare CSV file
    csv_fields = [
        "index", "video_name", "true_label", "pred_label", "true_id", "pred_id",
        "is_correct", "classification_type", "max_score", "avg_score", "anom_ratio", "time_sec"
    ]
    if not csv_path.exists() or not resume or len(completed_results) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    # Load Model Pipeline
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    model_file = str(ckpt_path) if ckpt_path.exists() else None
    if not model_file:
        print(f"[Error] Checkpoint not found at {ckpt_path}. Run training first.")
        sys.exit(1)

    print(f"\nInitializing Video Anomaly Detection Pipeline ({ckpt_path.name})...")
    detector = VideoAnomalyDetector(model_path=model_file, threshold=threshold)

    print_banner(len(all_items), normal_cnt, anom_cnt, threshold)

    all_results = []
    start_time = time.time()

    for idx, item in enumerate(all_items, start=1):
        vpath = item["path"]
        vname = vpath.name
        true_label = item["true_label"]
        true_id = item["label_id"]

        if vname in completed_results:
            res = completed_results[vname]
            res["index"] = idx
            all_results.append(res)
            print_video_row(idx, vname, true_label, res["pred_label"], res["max_score"], res["is_correct"], res["time_sec"])
            continue

        t0 = time.time()
        try:
            analysis = detector.process_video(
                str(vpath),
                log_to_db=False,
                scan_weapons=scan_weapons,
                use_ema=True,
                return_annotated=False,
                cache_features=False
            )
            detector.clear_cache()
            gc.collect()
            elapsed = time.time() - t0

            if "error" in analysis:
                pred_label = "ERROR"
                pred_id = -1
                max_s = 0.0
                avg_s = 0.0
                anom_r = 0.0
                is_correct = False
                c_type = "ERROR"
            else:
                anom_detected = bool(analysis.get("anomaly_detected", False))
                pred_label = "ANOMALOUS" if anom_detected else "NORMAL"
                pred_id = 1 if anom_detected else 0
                max_s = float(analysis.get("max_score", 0.0))
                avg_s = float(analysis.get("avg_score", 0.0))
                
                scores = analysis.get("scores", [])
                anom_clips = sum(1 for s in scores if s >= threshold)
                anom_r = anom_clips / max(1, len(scores))
                
                is_correct = (pred_id == true_id)
                if true_id == 1 and pred_id == 1:
                    c_type = "TP"
                elif true_id == 0 and pred_id == 0:
                    c_type = "TN"
                elif true_id == 0 and pred_id == 1:
                    c_type = "FP"
                else:
                    c_type = "FN"

        except Exception as e:
            detector.clear_cache()
            gc.collect()
            elapsed = time.time() - t0
            print(f"\n[Warning] Error on {vname}: {e}")
            pred_label = f"ERR"
            pred_id = -1
            max_s = 0.0
            avg_s = 0.0
            anom_r = 0.0
            is_correct = False
            c_type = "ERROR"

        row_data = {
            "index": idx,
            "video_name": vname,
            "true_label": true_label,
            "pred_label": pred_label,
            "true_id": true_id,
            "pred_id": pred_id,
            "is_correct": is_correct,
            "classification_type": c_type,
            "max_score": round(max_s, 4),
            "avg_score": round(avg_s, 4),
            "anom_ratio": round(anom_r, 4),
            "time_sec": round(elapsed, 2)
        }
        all_results.append(row_data)

        # Write incrementally to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row_data)

        print_video_row(idx, vname, true_label, pred_label, max_s, is_correct, elapsed)

    # Clean and consolidate final CSV
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
    except Exception as e:
        print(f"[Warning] Failed consolidating CSV: {e}")

    total_elapsed = time.time() - start_time
    print_final_summary(all_results, total_elapsed, str(csv_path))


def print_final_summary(results: List[Dict[str, Any]], total_time: float, csv_file: str):
    valid_results = [r for r in results if r["pred_id"] in (0, 1)]
    if not valid_results:
        print("\n[Warning] No valid inference results to summarize.")
        return

    y_true = [r["true_id"] for r in valid_results]
    y_pred = [r["pred_id"] for r in valid_results]

    tp = sum(1 for r in valid_results if r["classification_type"] == "TP")
    tn = sum(1 for r in valid_results if r["classification_type"] == "TN")
    fp = sum(1 for r in valid_results if r["classification_type"] == "FP")
    fn = sum(1 for r in valid_results if r["classification_type"] == "FN")

    accuracy = (tp + tn) / max(1, len(valid_results))
    normal_acc = tn / max(1, (tn + fp)) # Specificity
    anom_acc = tp / max(1, (tp + fn))   # Sensitivity / Recall

    print("\n" + "=" * 80)
    print("                     OVERALL VIDEO-LEVEL ACCURACY REPORT")
    print("=" * 80)
    print(f"  Total Videos Evaluated        : {len(valid_results)}")
    print(f"  Overall Video Accuracy        : {accuracy * 100:.2f}% ({tp + tn}/{len(valid_results)})")
    print(f"  Normal Detection (Specificity): {normal_acc * 100:.2f}% ({tn}/{tn + fp})")
    print(f"  Anomaly Detection (Recall)    : {anom_acc * 100:.2f}% ({tp}/{tp + fn})")
    print(f"  False Alarms (False Positives): {fp} ({fp / max(1, tn + fp) * 100:.1f}%)")
    print(f"  Missed Anomalies (False Neg): {fn} ({fn / max(1, tp + fn) * 100:.1f}%)")
    print(f"  Total Benchmark Time          : {total_time:.1f}s (~{total_time / max(1, len(valid_results)):.1f}s/video)")
    print("-" * 80)
    print("  CONFUSION MATRIX [TN, FP / FN, TP]:")
    print(f"    [[TN: {tn:<5}  FP: {fp:<5}]")
    print(f"     [FN: {fn:<5}  TP: {tp:<5}]]")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"], digits=4))
    print(f"  [Output] Full individual video results saved to:\n  -> {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection on Individual Videos")
    parser.add_argument("--test-dir", "-d", type=str, default=None, help="Directory containing test/normal and test/anomalous")
    parser.add_argument("--threshold", "-t", type=float, default=ANOMALY_THRESHOLD, help="Anomaly decision threshold")
    parser.add_argument("--csv", "-c", type=str, default=None, help="Output CSV path")
    parser.add_argument("--limit-per-class", "-l", type=int, default=None, help="Limit videos per class (e.g. 5 or 25 for quick check)")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch and overwrite existing CSV")
    parser.add_argument("--scan-weapons", action="store_true", help="Also run YOLO weapon threat detection")
    args = parser.parse_args()

    evaluate_individual_videos(
        test_dir=args.test_dir,
        threshold=args.threshold,
        output_csv=args.csv,
        limit_per_class=args.limit_per_class,
        resume=not args.no_resume,
        scan_weapons=args.scan_weapons
    )
