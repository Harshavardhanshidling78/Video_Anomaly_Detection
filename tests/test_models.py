import unittest
import torch
from src.models.vit_extractor import SpatialFeatureExtractor
from src.models.temporal_model import RecurrentTemporalModel
from src.models.anomaly_classifier import VideoAnomalyDetectorModel

class TestDeepLearningModels(unittest.TestCase):
    def test_spatial_extractor_fast(self):
        extractor = SpatialFeatureExtractor(mode="fast")
        # Dummy batch clip: (B=2, T=16, C=3, H=224, W=224)
        x = torch.randn(2, 16, 3, 224, 224)
        feats = extractor(x)
        self.assertEqual(feats.shape, (2, 16, 768))

    def test_recurrent_temporal_model(self):
        temporal_net = RecurrentTemporalModel(input_dim=768, hidden_dim=256, cell_type="LSTM")
        x = torch.randn(2, 16, 768)
        context = temporal_net(x)
        self.assertEqual(context.shape, (2, 256))

    def test_video_anomaly_detector_model(self):
        model = VideoAnomalyDetectorModel(spatial_mode="fast", cell_type="LSTM")
        x = torch.randn(2, 16, 3, 224, 224)
        scores, preds = model(x)
        self.assertEqual(scores.shape, (2, 1))
        self.assertEqual(preds.shape, (2, 1))

if __name__ == "__main__":
    unittest.main()
