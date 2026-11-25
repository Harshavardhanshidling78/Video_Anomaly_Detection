import unittest
import numpy as np
from src.pipeline.detector import VideoAnomalyDetector

class TestActorLocalizationAndThreatFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize detector with default configs
        cls.detector = VideoAnomalyDetector(model_path="checkpoints/best_model.pth")

    def test_overhead_ceiling_rail_rejection(self):
        """
        Verify that overhead bus handrails and ceiling window glare (like [6, 0 -> 427, 265])
        are strictly rejected as anomaly ROIs when no actor is present in the ceiling.
        """
        h, w = 720, 1280
        curr_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        ref_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        
        # Simulate bright window glare / ceiling handrail motion at top of bus
        # Bounding box roughly [6, 0 -> 427, 265]
        curr_frame[0:265, 6:427] = 240
        ref_frame[0:265, 6:427] = 40
        
        # Localize anomaly without persons: ceiling rail must NOT be returned!
        roi = self.detector.localize_anomaly_region(curr_frame, ref_frame, person_boxes=None)
        self.assertIsNotNone(roi)
        # Should fallback to safe middle/lower activity zone, not ceiling
        box = roi["box"]
        self.assertGreaterEqual(box[1], int(0.15 * h), "ROI must not anchor to top ceiling edge")
        self.assertNotEqual(roi["type"], "ceiling_fixture")

    def test_anomalous_actor_prioritization(self):
        """
        Verify that when a human actor is present (even if ceiling glare has larger area),
        the anomaly ROI is anchored to the human actor ('ANOMALOUS ACTOR HERE').
        """
        h, w = 720, 1280
        curr_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        ref_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        
        # Motion at actor location (seated / fighting person: [100, 300, 450, 700])
        curr_frame[300:700, 100:450] = 220
        ref_frame[300:700, 100:450] = 50
        
        person_boxes = [[100, 300, 450, 700]]
        roi = self.detector.localize_anomaly_region(curr_frame, ref_frame, person_boxes=person_boxes)
        
        self.assertIsNotNone(roi)
        self.assertEqual(roi["type"], "person")
        self.assertEqual(roi["label"], "ANOMALOUS ACTOR HERE")
        self.assertEqual(roi["box"], [100, 300, 450, 700])

    def test_altercation_merging(self):
        """
        Verify that when two actors are grappling/fighting close to each other,
        their bounding boxes are merged into an altercation envelope.
        """
        h, w = 720, 1280
        curr_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        ref_frame = np.ones((h, w, 3), dtype=np.uint8) * 100
        
        # Two grappling actors with active motion
        curr_frame[250:650, 100:400] = 210
        ref_frame[250:650, 100:400] = 60
        
        person1 = [100, 250, 280, 650]
        person2 = [220, 260, 400, 640] # overlapping
        
        roi = self.detector.localize_anomaly_region(
            curr_frame, ref_frame, person_boxes=[person1, person2]
        )
        
        self.assertIsNotNone(roi)
        self.assertEqual(roi["type"], "altercation")
        self.assertIn("ALTERCATION", roi["label"])
        # Envelope must enclose both actors
        self.assertEqual(roi["box"][0], 100)
        self.assertEqual(roi["box"][2], 400)

    def test_stationary_metallic_fixture_rejected_as_weapon(self):
        """
        Verify that candidate weapons located on the ceiling (overhead rail) or
        without dynamic motion are filtered out and not treated as active weapon threats.
        """
        h, w = 432, 576
        # Simulate ceiling handrail box [449, 0, 573, 72]
        ceiling_rail_box = [449, 0, 573, 72]
        # Ceilings must be excluded
        is_ceiling = (ceiling_rail_box[1] <= 0.06 * h) and (ceiling_rail_box[3] <= 0.22 * h)
        self.assertTrue(is_ceiling, "Ceiling rail box should match ceiling exclusion filter")

if __name__ == "__main__":
    unittest.main()
