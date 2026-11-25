import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import numpy as np
from src.pipeline.detector import VideoAnomalyDetector

class TestClipTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = VideoAnomalyDetector(model_path="checkpoints/best_model.pth")
        cls.test_video = "data/train/anomalous/gettyimages-81931343-640_adpp.mp4"

    def test_calibrated_clip_level_timeline(self):
        res = self.detector.process_video(self.test_video, log_to_db=False)
        total = res["total_clips"]
        is_anom = res["is_anomalous"]
        scores = res["scores"]
        anom_count = sum(is_anom)
        normal_count = total - anom_count

        print("\n" + "="*60)
        print("VERIFICATION RESULTS FOR gettyimages-81931343-640_adpp.mp4")
        print(f"Total clips: {total}")
        print(f"Normal clips (Green): {normal_count} ({normal_count/total*100:.1f}%)")
        print(f"Anomalous clips (Red): {anom_count} ({anom_count/total*100:.1f}%)")
        print(f"Peak score: {res['max_score']:.4f}")
        print(f"Average score: {res['avg_score']:.4f}")
        print(f"Video Anomaly Detected: {res['anomaly_detected']}")
        print(f"Weapon Detected: {res['weapon_detected']}")
        print(f"Anomaly Classification: {res['anomaly_type']}")
        print("="*60)

        # 1. Total clips should match video duration
        self.assertEqual(total, 61)

        # 2. Timeline must NOT be saturated (must contain both green and red clips)
        self.assertGreater(normal_count, 10, "Timeline should have normal clips identified as green")
        self.assertGreater(anom_count, 10, "Timeline should have anomalous clips identified as red")

        # 3. Overall video verdict must remain anomalous
        self.assertTrue(res["anomaly_detected"])
        self.assertIn(res["anomaly_type"], ["Abnormal / Violent Motion", "Weapon Threat Detected"])

        # 4. Scores of normal clips must be below threshold, anomalous above threshold
        for score, flag in zip(scores, is_anom):
            if flag:
                self.assertGreaterEqual(score, self.detector.threshold)
            else:
                self.assertLess(score, self.detector.threshold)

        # 5. Concluding clips (after altercation ends and scene calms down) must be normal (Green)
        for i in range(50, total):
            self.assertFalse(is_anom[i], f"Clip {i+1} at end of video should be normal green, got score {scores[i]}")

if __name__ == "__main__":
    unittest.main()
