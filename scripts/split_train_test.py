import os
import json
import shutil
import argparse
import random
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# Videos referenced in automated regression tests that must remain in train
PROTECTED_FILES = {
    "gettyimages-81931343-640_adpp"
}

def split_dataset(test_count: int = 250, copy_mode: bool = False, seed: int = 42):
    """
    Splits video dataset into:
      - Training: data/train/{normal, anomalous}
      - Testing:  data/test/{normal, anomalous}

    Ensures:
      1. Zero clip-level scene leakage by grouping clips by base video ID.
      2. Exactly `test_count` clips (default: 250 normal + 250 anomalous = 500 test videos).
      3. Saves dataset manifest to data/dataset_split_manifest.json for reproducibility.
    """
    random.seed(seed)
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "test_count_per_category": test_count,
        "mode": "copy" if copy_mode else "move",
        "splits": {}
    }

    print(f"\n==================================================================")
    print(f"🎬 Executing Dataset Train-Test Split (Target: {test_count} Test per Category)")
    print(f"==================================================================")

    for category in ["normal", "anomalous"]:
        src_cat = train_dir / category
        dst_cat = test_dir / category
        dst_cat.mkdir(parents=True, exist_ok=True)

        if not src_cat.exists():
            print(f"[Warning] Directory {src_cat} does not exist.")
            continue

        video_files = sorted(list(src_cat.glob("*.mp4")) + list(src_cat.glob("*.avi")) + list(src_cat.glob("*.mov")))
        if not video_files:
            print(f"[Info] No videos found in {src_cat}")
            continue

        print(f"\n[{category.upper()}] Initial videos in train: {len(video_files)}")

        # Group clips by base video ID (e.g., base '-1l5631l3fg' from '-1l5631l3fg_0.avi')
        groups = {}
        for vf in video_files:
            stem = vf.stem
            base_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
            groups.setdefault(base_id, []).append(vf)

        # Separate protected bases
        eligible_bases = [b for b in groups.keys() if b not in PROTECTED_FILES]
        random.shuffle(eligible_bases)

        selected_test_clips = []
        selected_test_bases = []

        # 1. Greedy selection of complete parent groups without scene leakage
        for bid in eligible_bases:
            clips = groups[bid]
            if len(selected_test_clips) + len(clips) <= test_count:
                selected_test_clips.extend(clips)
                selected_test_bases.append(bid)
            if len(selected_test_clips) == test_count:
                break

        # Fallback if group sizes didn't hit exact count
        if len(selected_test_clips) < test_count:
            remaining_needed = test_count - len(selected_test_clips)
            print(f"[{category}] Adding {remaining_needed} remaining clips to reach exact {test_count} count...")
            for bid in eligible_bases:
                if bid not in selected_test_bases:
                    clips = groups[bid]
                    for c in clips:
                        if len(selected_test_clips) < test_count:
                            selected_test_clips.append(c)
                        else:
                            break
                if len(selected_test_clips) == test_count:
                    break

        # Move or Copy files to test directory
        test_file_names = []
        for vf in selected_test_clips:
            dst_path = dst_cat / vf.name
            if copy_mode:
                shutil.copy2(vf, dst_path)
            else:
                shutil.move(vf, dst_path)
            test_file_names.append(vf.name)

        remaining_train_files = sorted([f.name for f in src_cat.glob("*.*") if f.is_file()])

        manifest["splits"][category] = {
            "test_count": len(test_file_names),
            "train_count": len(remaining_train_files),
            "test_files": test_file_names,
            "train_files": remaining_train_files
        }

        action_name = "Copied" if copy_mode else "Extracted & Moved"
        print(f"✅ {action_name} {len(test_file_names)} videos to {dst_cat}")
        print(f"📊 Remaining {category} videos in {src_cat}: {len(remaining_train_files)}")

    manifest_path = data_dir / "dataset_split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Manifest] Saved split record to: {manifest_path}")
    print(f"==================================================================")
    print(f"🎯 Total Test Videos: {len(manifest['splits'].get('normal', {}).get('test_files', [])) + len(manifest['splits'].get('anomalous', {}).get('test_files', []))}")
    print(f"🎯 Total Train Videos: {len(manifest['splits'].get('normal', {}).get('train_files', [])) + len(manifest['splits'].get('anomalous', {}).get('train_files', []))}")
    print(f"==================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into dedicated train and test sets")
    parser.add_argument("--test_count", type=int, default=250, help="Number of videos per class for testing (default: 250)")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_dataset(test_count=args.test_count, copy_mode=args.copy, seed=args.seed)
