import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import torch
from src.models.vit_extractor import SpatialFeatureExtractor
from src.models.temporal_model import RecurrentTemporalModel
from src.models.anomaly_classifier import VideoAnomalyDetectorModel

class TestDeepLearningModels(unittest.TestCase):
    def test_spatial_extractor_vit(self):
        extractor = SpatialFeatureExtractor(mode="vit", pretrained=False)
        extractor.eval()
        # Test forward pass with inference mode
        with torch.no_grad():
            x = torch.randn(1, 4, 3, 224, 224)
            feats = extractor(x)
            self.assertEqual(feats.shape, (1, 4, 768))

    def test_recurrent_temporal_model(self):
        temporal_net = RecurrentTemporalModel(input_dim=768, hidden_dim=256, cell_type="LSTM", bidirectional=True)
        temporal_net.eval()
        with torch.no_grad():
            x = torch.randn(2, 16, 768)
            context = temporal_net(x)
            self.assertEqual(context.shape, (2, 512))

            # Test unidirectional compatibility
            temporal_uni = RecurrentTemporalModel(input_dim=768, hidden_dim=256, cell_type="LSTM", bidirectional=False)
            self.assertEqual(temporal_uni(x).shape, (2, 256))

    def test_video_anomaly_detector_model(self):
        model = VideoAnomalyDetectorModel(spatial_mode="vit", pretrained_vit=False, cell_type="LSTM")
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 4, 3, 224, 224)
            scores, preds = model(x)
            self.assertEqual(scores.shape, (1, 1))
            self.assertEqual(preds.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()
